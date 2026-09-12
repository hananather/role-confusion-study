#!/usr/bin/env python3
"""I audit my saved native probes without fitting, predicting, or loading a model.

I deserialize only my own trusted training artifacts. Native cuML deserialization
may initialize its CUDA handle; this helper reads existing estimator state and
copies coefficient arrays to host memory. It never calls fit, predict, forward,
downloads, or provider APIs. Its only file writes are the requested JSON report.

Usage: python3 audit_saved_probes.py --probe-run /workspace/results/R/probes-full \
           --out /workspace/results/R/probes-full/native-probe-audit.json
       python3 audit_saved_probes.py --self-test

Saved iteration counts and objectives do not establish convergence. In
particular, stopping before max_iter can also follow a failed line search.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
from pathlib import Path
import pickle
import sys


ROLE_SPACES = (
    ('user', 'assistant'), ('user', 'assistant', 'tool'),
    ('user', 'cot', 'assistant'), ('user', 'cot', 'assistant', 'tool'),
    ('system', 'user', 'assistant'), ('system', 'user', 'assistant', 'tool'),
    ('system', 'user', 'cot', 'assistant'), ('system', 'user', 'cot', 'assistant', 'tool'),
)
ROLE_CHAR = {'system': 's', 'user': 'u', 'cot': 'c', 'assistant': 'a', 'tool': 't'}
EXPECTED_PARAMS = {'C': 0.005, 'penalty': 'l2', 'max_iter': 5000,
                   'linesearch_max_iter': 100, 'fit_intercept': True}
EXPECTED_KEYS = {(layer, roles) for layer in range(24) for roles in ROLE_SPACES}
CONVERGENCE_BOUNDARY = (
    'Convergence is unestablished by this audit. Saved iterations below 5000, '
    'finite coefficients, finite objectives, and held-out metrics do not prove '
    'optimizer convergence or successful line search.'
)


def host_array(value):
    """Read existing NumPy, CuPy, or cuML array state without evaluating a probe."""
    import numpy as np
    if hasattr(value, 'get'):
        value = value.get()
    elif hasattr(value, 'to_output'):
        value = value.to_output('numpy')
    elif hasattr(value, 'to_numpy'):
        value = value.to_numpy()
    return np.asarray(value)


def json_value(value):
    """Keep the report finite JSON, without object reprs or large array payloads."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(float(value)) else {'nonfinite': str(float(value))}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, 'shape'):
        array = host_array(value)
        if array.size <= 64:
            return json_value(array.tolist())
        return {'shape': list(array.shape), 'dtype': str(array.dtype), 'values_omitted': True}
    return {'object_type': type(value).__module__ + '.' + type(value).__name__}


def optional_state(estimator, path):
    try:
        value = estimator
        for name in path.split('.'):
            value = getattr(value, name)
    except AttributeError:
        return {'available': False, 'reason': 'not_saved_or_not_exposed'}
    except Exception as error:
        return {'available': False, 'reason': 'read_failed', 'error_type': type(error).__name__}
    try:
        return {'available': True, 'value': json_value(value)}
    except Exception as error:
        return {'available': False, 'reason': 'host_conversion_failed', 'error_type': type(error).__name__}


def space_label(roles):
    return ''.join(ROLE_CHAR.get(role, '?') for role in roles)


def key_record(key):
    layer, roles = key
    return {'layer_ix': layer, 'role_space': list(roles), 'role_space_abbreviation': space_label(roles)}


def audit_probe(record, index):
    import numpy as np
    result = {'index': index, 'defects': [], 'convergence': 'unestablished'}
    defects = result['defects']
    if not isinstance(record, dict):
        defects.append('record_is_not_a_dictionary')
        return result, None
    layer = record.get('layer_ix')
    roles = record.get('role_space')
    layer_valid = isinstance(layer, numbers.Integral) and not isinstance(layer, bool) and 0 <= layer < 24
    roles_valid = isinstance(roles, (list, tuple)) and tuple(roles) in ROLE_SPACES
    if not layer_valid:
        defects.append('layer_outside_0_to_23_or_invalid')
    if not roles_valid:
        defects.append('unexpected_role_space_or_order')
    result.update(layer_ix=json_value(layer), role_space=json_value(roles))
    if not roles_valid:
        return result, None
    roles = tuple(roles)
    result['role_space_abbreviation'] = space_label(roles)
    expected_map = {role: i for i, role in enumerate(roles)}
    actual_map = record.get('roles_map')
    result['roles_map'] = json_value(actual_map)
    if actual_map != expected_map:
        defects.append('roles_map_differs_from_declared_role_order')
    key = (int(layer), roles) if layer_valid else None
    for name in ('acc', 'nll', 'n_inputs'):
        if name in record:
            result[name] = json_value(record[name])
    for name in ('acc', 'nll'):
        value = record.get(name)
        if value is not None:
            try:
                finite = math.isfinite(float(value))
                valid = finite and (0 <= float(value) <= 1 if name == 'acc' else float(value) >= 0)
                if not valid:
                    defects.append(name + '_is_nonfinite_or_out_of_range')
            except (TypeError, ValueError):
                defects.append(name + '_is_not_numeric')
    pipeline = record.get('probe')
    result['pipeline_type'] = type(pipeline).__module__ + '.' + type(pipeline).__name__
    if result['pipeline_type'] != 'sklearn.pipeline.Pipeline':
        defects.append('expected_sklearn_pipeline')
    try:
        steps = pipeline.named_steps
        result['pipeline_steps'] = list(steps)
        if list(steps) != ['clf']:
            defects.append('pipeline_has_scaler_or_unexpected_steps')
        clf = steps['clf']
    except (AttributeError, KeyError, TypeError):
        defects.append('classifier_unavailable')
        return result, key
    result['estimator_type'] = type(clf).__module__ + '.' + type(clf).__name__
    if not type(clf).__module__.startswith('cuml.') or type(clf).__name__ != 'LogisticRegression':
        defects.append('expected_native_cuml_logistic_regression')
    try:
        params = clf.get_params()
        result['params'] = json_value(params)
        for name, expected in EXPECTED_PARAMS.items():
            actual = params.get(name)
            if name == 'C':
                try:
                    matches = math.isclose(float(actual), expected, rel_tol=1e-8, abs_tol=1e-12)
                except (TypeError, ValueError):
                    matches = False
            elif name == 'fit_intercept':
                matches = isinstance(actual, (bool, np.bool_)) and bool(actual)
            else:
                matches = actual == expected
            if not matches:
                defects.append('unexpected_parameter:' + name)
        if params.get('solver', 'qn') != 'qn':
            defects.append('unexpected_parameter:solver')
        if params.get('class_weight') is not None:
            defects.append('unexpected_parameter:class_weight')
    except Exception as error:
        result['params_read_error'] = type(error).__name__
        defects.append('estimator_parameters_unavailable')
    expected_rows = 1 if len(roles) == 2 else len(roles)
    result['expected_coef_shape'] = [expected_rows, 2880]
    result['expected_intercept_elements'] = expected_rows
    for field in ('coef_', 'intercept_'):
        try:
            array = host_array(getattr(clf, field))
            finite = bool(np.isfinite(array).all())
            result[field] = {'shape': list(array.shape), 'dtype': str(array.dtype), 'finite': finite}
            if not finite:
                defects.append(field + '_contains_nonfinite_values')
            if field == 'coef_' and array.shape != (expected_rows, 2880):
                defects.append('coefficient_shape_or_feature_count_mismatch')
            if field == 'intercept_' and (array.size != expected_rows or array.ndim not in (1, 2)):
                defects.append('intercept_shape_or_class_count_mismatch')
        except Exception as error:
            result[field] = {'available': False, 'error_type': type(error).__name__}
            defects.append(field + '_unavailable_or_invalid')
    try:
        classes = host_array(clf.classes_)
        result['classes_'] = json_value(classes)
        if classes.ndim != 1 or classes.tolist() != list(range(len(roles))):
            defects.append('classifier_class_order_differs_from_roles_map')
    except Exception as error:
        result['classes_'] = {'available': False, 'error_type': type(error).__name__}
        defects.append('classifier_classes_unavailable')
    result['saved_solver_state'] = {name: optional_state(clf, name) for name in
                                   ('n_iter_', 'solver_model.num_iters', 'solver_model.objective')}
    # Missing optional solver fields are informative, never structural failures.
    result['iteration_limit_reached'] = None
    numeric_iterations = []
    for name in ('n_iter_', 'solver_model.num_iters'):
        state = result['saved_solver_state'][name]
        if state['available']:
            try:
                values = np.asarray(state['value'], dtype=float).reshape(-1)
                if values.size and np.isfinite(values).all():
                    numeric_iterations.extend(values.tolist())
            except (TypeError, ValueError):
                pass
    if numeric_iterations:
        result['iteration_limit_reached'] = any(value >= EXPECTED_PARAMS['max_iter'] for value in numeric_iterations)
    return result, key


def audit_collection(probes, split):
    if not isinstance(probes, list):
        return {'split': split, 'passed': False, 'defects': ['artifact_is_not_a_list'],
                'convergence_boundary': CONVERGENCE_BOUNDARY}
    rows, keys = [], []
    for index, probe in enumerate(probes):
        try:
            result, key = audit_probe(probe, index)
        except Exception as error:
            result = {'index': index, 'defects': ['record_inspection_failed'], 'error_type': type(error).__name__,
                      'convergence': 'unestablished'}
            key = None
        rows.append(result)
        if key is not None:
            keys.append(key)
    counts = Counter(keys)
    missing = sorted(EXPECTED_KEYS - set(keys))
    duplicate = sorted(key for key, count in counts.items() if count > 1)
    defects = []
    if len(probes) != 192:
        defects.append('expected_192_records')
    if missing:
        defects.append('missing_layer_role_combinations')
    if duplicate:
        defects.append('duplicate_layer_role_combinations')
    if any(row['defects'] for row in rows):
        defects.append('one_or_more_probe_records_failed')
    return {'split': split, 'passed': not defects, 'defects': defects,
            'n_records': len(probes), 'n_expected': 192, 'n_unique_expected_combinations': len(set(keys)),
            'missing_combinations': [key_record(key) for key in missing],
            'duplicate_combinations': [{**key_record(key), 'count': counts[key]} for key in duplicate],
            'n_probes_with_defects': sum(bool(row['defects']) for row in rows),
            'n_iteration_limit_reached': sum(row.get('iteration_limit_reached') is True for row in rows),
            'n_iteration_count_unavailable': sum(row.get('iteration_limit_reached') is None for row in rows),
            'convergence_boundary': CONVERGENCE_BOUNDARY, 'probes': rows}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--probe-run', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.probe_run or not args.out:
        parser.error('--probe-run and --out are required')
    sources = {'prompt': args.probe_run / 'gptoss-20b.pkl', 'base': args.probe_run / 'gptoss-20b-basesplit.pkl'}
    if args.out.resolve() in {path.resolve() for path in sources.values()}:
        parser.error('The JSON report must not overwrite a source pickle')
    reports = []
    for split, path in sources.items():
        try:
            before = path.stat()
            checksum = file_hash(path)
            with path.open('rb') as handle:
                trusted_probes = pickle.load(handle)
            after = path.stat()
            report = audit_collection(trusted_probes, split)
            del trusted_probes
            report['source'] = {'path': str(path.resolve()), 'sha256': checksum, 'bytes': after.st_size}
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                report['passed'] = False
                report['defects'].append('source_changed_during_read')
        except Exception as error:
            report = {'split': split, 'passed': False, 'defects': ['artifact_read_failed'],
                      'error_type': type(error).__name__, 'source_path': str(path.resolve()),
                      'convergence_boundary': CONVERGENCE_BOUNDARY}
        reports.append(report)
    result = {'audited_utc': datetime.now(timezone.utc).isoformat(),
              'operation': 'read saved native estimator state only; no fit/predict/forward',
              'passed': all(report['passed'] for report in reports),
              'expected_total_probes': 384, 'convergence': 'unestablished',
              'convergence_boundary': CONVERGENCE_BOUNDARY, 'splits': reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(json.dumps({'passed': result['passed'], 'out': str(args.out.resolve()),
                      'splits': [{'split': row['split'], 'passed': row['passed'],
                                  'n_probes_with_defects': row.get('n_probes_with_defects'),
                                  'defects': row['defects']} for row in reports],
                      'convergence': 'unestablished'}))
    return 0 if result['passed'] else 2


def self_test():
    """Inject independent structural defects without importing CUDA or sklearn."""
    import copy
    from types import SimpleNamespace
    import numpy as np

    class Native:
        def __init__(self, n_classes):
            rows = 1 if n_classes == 2 else n_classes
            self.coef_ = np.zeros((rows, 2880), dtype=np.float32)
            self.intercept_ = np.zeros(rows, dtype=np.float32)
            self.classes_ = np.arange(n_classes)
            self.n_iter_ = np.array([7])
            self.solver_model = SimpleNamespace(num_iters=7, objective=0.4)
        def get_params(self): return {**EXPECTED_PARAMS, 'solver': 'qn', 'class_weight': None}
        def fit(self, *_): raise AssertionError('Audit must never fit')
        def predict(self, *_): raise AssertionError('Audit must never predict')
        def predict_proba(self, *_): raise AssertionError('Audit must never predict')

    Native.__module__, Native.__name__ = 'cuml.linear_model.logistic_regression', 'LogisticRegression'
    Pipeline = type('Pipeline', (), {'__module__': 'sklearn.pipeline'})
    records = []
    for layer in range(24):
        for roles in ROLE_SPACES:
            pipeline = Pipeline()
            pipeline.named_steps = {'clf': Native(len(roles))}
            records.append({'probe': pipeline, 'layer_ix': layer, 'role_space': list(roles),
                            'roles_map': dict(zip(roles, range(len(roles)))), 'acc': 0.7, 'nll': 0.5})
    valid = audit_collection(records, 'prompt')
    assert valid['passed'] and valid['n_records'] == 192
    assert all(row['convergence'] == 'unestablished' for row in valid['probes'])
    assert valid['n_iteration_limit_reached'] == 0
    invalid = copy.deepcopy(records)
    invalid[0]['probe'].named_steps['clf'].coef_[0, 4] = np.nan
    invalid[1]['probe'].named_steps['clf'].classes_ = np.array([1, 0, 2])
    invalid[2]['probe'].named_steps['clf'].coef_ = np.zeros((3, 2879))
    invalid[3]['probe'].named_steps['extra_scaler'] = object()
    invalid[-1] = copy.deepcopy(invalid[-2])
    bad = audit_collection(invalid, 'base')
    assert not bad['passed']
    assert 'coef__contains_nonfinite_values' in bad['probes'][0]['defects']
    assert 'classifier_class_order_differs_from_roles_map' in bad['probes'][1]['defects']
    assert 'coefficient_shape_or_feature_count_mismatch' in bad['probes'][2]['defects']
    assert 'pipeline_has_scaler_or_unexpected_steps' in bad['probes'][3]['defects']
    assert bad['missing_combinations'] and bad['duplicate_combinations']
    binary = copy.deepcopy(records[0])
    binary['probe'].named_steps['clf'].coef_ = np.zeros((2, 2880))
    assert 'coefficient_shape_or_feature_count_mismatch' in audit_probe(binary, 0)[0]['defects']
    optional = copy.deepcopy(records[0])
    del optional['probe'].named_steps['clf'].n_iter_
    del optional['probe'].named_steps['clf'].solver_model
    row, _ = audit_probe(optional, 0)
    assert not row['defects'] and all(not field['available'] for field in row['saved_solver_state'].values())
    assert row['iteration_limit_reached'] is None
    json.dumps(bad, allow_nan=False)
    print('Passed: full coverage, finite weights, class order, feature count, scaler exclusion, binary shape, missing optional solver fields, and no convergence inference.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
