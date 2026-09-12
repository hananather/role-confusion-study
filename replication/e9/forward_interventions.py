"""I inject one masked vector into the authors' GPT-OSS manual forward.

I keep their frozen source untouched. ``adapt_forward(authors_forward)`` copies
the inspected function into a private namespace and inserts callbacks at its
one pre-MLP site and its two mutually exclusive post-block residual sites.

Example after ``persistent.cuda_readouts.model_loader`` returns its forward::

    forward = adapt_forward(authors_forward)
    intervention = Intervention("block_output", 11, unit_direction(clean), 0.3 * scale)
    result = forward(model, ids, attention, return_hidden_states=True,
                     intervention=intervention, content_mask=content_mask)

For my current UAT probe rows I use ``Intervention("pre_mlp", 12,
unit_direction(coef[role_index]), magnitude)``. UAT order is user, assistant,
tool. I supply the magnitude explicitly; this adapter does not choose it.
Masks are boolean NumPy arrays with shape (batch, sequence). ``token_mask``
defaults to ``content_mask``; if supplied, it must select only content tokens.
I validate both masks against the actual attention mask and prompt dimensions.

The original result keys remain available, with ``intervention_report`` added.
Pre-MLP captures at the injected layer include a pre-MLP intervention. A block
output intervention first affects the following layer's pre-MLP capture.
``None`` and zero magnitude preserve the original tensor arithmetic exactly.
I can run ``python e9/forward_interventions.py --self-test --source PATH`` without
Torch, a model, a GPU, or network access; PATH is the frozen gptoss.py source.
"""

import ast
import copy
from dataclasses import dataclass
import functools
import hashlib
import inspect
import math
from pathlib import Path
import textwrap

import numpy as np


def unit_direction(values):
    """I normalize one finite, nonzero vector without interpreting its role."""
    vector = np.array(values, dtype=np.float32, copy=True)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("My direction must be a finite nonempty vector")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not norm > 0:
        raise ValueError("My direction cannot have zero norm")
    vector /= norm
    return vector


@dataclass(frozen=True)
class Intervention:
    """I specify one unit vector, signed magnitude, layer, and injection site."""

    site: str
    layer: int
    direction: np.ndarray
    magnitude: float

    def __post_init__(self):
        if self.site not in {"pre_mlp", "block_output"}:
            raise ValueError("My site must be pre_mlp or block_output")
        if isinstance(self.layer, (bool, np.bool_)) or not isinstance(self.layer, (int, np.integer)) or self.layer < 0:
            raise ValueError("My layer must be a nonnegative integer")
        direction = np.array(self.direction, dtype=np.float32, copy=True)
        if direction.ndim != 1 or not direction.size or not np.isfinite(direction).all():
            raise ValueError("My direction must be a finite nonempty vector")
        norm = float(np.linalg.norm(direction.astype(np.float64)))
        if not np.isclose(norm, 1.0, atol=2e-6, rtol=0):
            raise ValueError("I require a unit direction; use unit_direction first")
        magnitude = float(self.magnitude)
        if not math.isfinite(magnitude):
            raise ValueError("My signed magnitude must be finite")
        direction.setflags(write=False)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "layer", int(self.layer))


_CALLBACK = "_codex_intervention_callback"


def _same(node, expression):
    return ast.dump(node, include_attributes=False) == ast.dump(ast.parse(expression).body[0], include_attributes=False)


def _instrument_source(source, namespace, filename):
    """I validate exact source locations before compiling my private copy."""
    source = textwrap.dedent(source)
    module = ast.parse(source, filename=filename)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "run_gptoss_return_topk":
        raise ValueError("I expected exactly the authors' run_gptoss_return_topk function")
    function = functions[0]
    if any(isinstance(n, ast.Name) and n.id == _CALLBACK for n in ast.walk(function)):
        raise ValueError("My reserved callback name already occurs in the source")
    expected_loop = ast.parse("for layer_ix, layer in enumerate(model.model.layers):\n    pass").body[0]
    loops = [n for n in ast.walk(function) if isinstance(n, ast.For)
             and ast.dump(n.target) == ast.dump(expected_loop.target)
             and ast.dump(n.iter) == ast.dump(expected_loop.iter)]
    if len(loops) != 1:
        raise ValueError("I require one exact decoder-layer loop")
    loop = loops[0]
    pre_pattern = "hidden_state = layer.post_attention_layernorm(hidden_state)"
    fused_pattern = "hidden_state = residual + routed_out"
    dense_pattern = "hidden_state = residual + final_hidden_states.view(B, N, D)"
    matches = {name: [n for n in ast.walk(function) if isinstance(n, ast.Assign) and _same(n, pattern)]
               for name, pattern in [("pre_mlp", pre_pattern), ("block_output_mxfp4", fused_pattern), ("block_output_dense", dense_pattern)]}
    if any(len(nodes) != 1 for nodes in matches.values()):
        raise ValueError(f"My injection sites are missing or duplicated: { {k: len(v) for k, v in matches.items()} }")
    branches = [n for n in loop.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "is_mxfp4"]
    if len(branches) != 1:
        raise ValueError("I require the authors' single MXFP4/dense branch")
    branch = branches[0]
    if (matches["pre_mlp"][0] not in loop.body
            or matches["block_output_mxfp4"][0] not in branch.body
            or matches["block_output_dense"][0] not in branch.orelse
            or loop.body.index(matches["pre_mlp"][0]) >= loop.body.index(branch)):
        raise ValueError("My source sites moved outside their expected computation order")
    capture_lists = [(loop.body, matches["pre_mlp"][0], "all_pre_mlp_hidden_states"),
                     (branch.body, matches["block_output_mxfp4"][0], "all_hidden_states"),
                     (branch.orelse, matches["block_output_dense"][0], "all_hidden_states")]
    for statements, assignment, capture_name in capture_lists:
        captures = [i for i, statement in enumerate(statements) if any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == capture_name
            and n.func.attr == "append" for n in ast.walk(statement))]
        if len(captures) != 1 or captures[0] <= statements.index(assignment):
            raise ValueError(f"My {capture_name} capture no longer follows its injection site")
    for statements, assignment, capture_name in capture_lists:
        site = "pre_mlp" if capture_name == "all_pre_mlp_hidden_states" else "block_output"
        callback = ast.parse(f"hidden_state = {_CALLBACK}(hidden_state, layer_ix, {site!r})").body[0]
        ast.copy_location(callback, assignment)
        statements.insert(statements.index(assignment) + 1, callback)
    function.decorator_list = []
    function.args.kwonlyargs.append(ast.arg(arg=_CALLBACK))
    function.args.kw_defaults.append(None)
    private_module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    private_globals = dict(namespace)
    exec(compile(private_module, filename, "exec"), private_globals)
    metadata = {"source_file": str(filename), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "patched_ast_sha256": hashlib.sha256(ast.dump(private_module).encode()).hexdigest(),
                "validated_site_matches": {k: len(v) for k, v in matches.items()},
                "source_site_lines_relative": {k: v[0].lineno for k, v in matches.items()}}
    return private_globals[function.name], metadata


def adapt_forward(authors_forward):
    """I adapt the exact runtime function returned by the existing model loader."""
    original = inspect.unwrap(authors_forward)
    if original.__closure__:
        raise ValueError("I cannot copy an unexpected closure from the authors' forward")
    source_lines, first_line = inspect.getsourcelines(original)
    adapted, metadata = _instrument_source("".join(source_lines), original.__globals__, inspect.getsourcefile(original))
    metadata["source_first_line"] = first_line
    metadata["source_site_lines"] = {key: first_line + line - 1 for key, line in metadata["source_site_lines_relative"].items()}
    return _wrap_forward(adapted, metadata, authors_forward)


def _wrap_forward(adapted, metadata, original):
    @functools.wraps(original)
    def forward(model, input_ids, attention_mask, return_hidden_states=False, *,
                intervention=None, content_mask=None, token_mask=None):
        import torch

        n_layers = len(model.model.layers)
        report = {"site": None, "layer": None, "magnitude": 0.0, "site_hits": 0, "n_layers": n_layers,
                  "applied_injections": 0, "selected_tokens": 0, "adapter": copy.deepcopy(metadata)}
        selected = delta = None
        if intervention is not None:
            if not isinstance(intervention, Intervention):
                raise TypeError("I require an Intervention specification")
            if intervention.layer >= n_layers:
                raise ValueError("My injection layer is outside this model")
            shape = tuple(input_ids.shape)
            if len(shape) != 2 or tuple(attention_mask.shape) != shape:
                raise ValueError("My input IDs and attention mask must share shape (batch, sequence)")
            content = np.asarray(content_mask)
            selected = content.copy() if token_mask is None else np.array(token_mask, copy=True)
            if content.dtype != np.bool_ or selected.dtype != np.bool_ or content.shape != shape or selected.shape != shape:
                raise ValueError("My content and token masks must be boolean arrays matching the input IDs")
            if np.any(selected & ~content) or not selected.any():
                raise ValueError("I must select at least one content token and no token outside content")
            attention = attention_mask.detach().cpu().numpy()
            if np.any(content & (attention == 0)):
                raise ValueError("My content mask includes padding")
            with np.errstate(over="ignore", invalid="ignore"):
                delta = (intervention.direction * intervention.magnitude).astype(np.float32)
            if not np.isfinite(delta).all():
                raise ValueError("My requested delta overflows float32")
            report.update(site=intervention.site, layer=intervention.layer, magnitude=intervention.magnitude,
                          selected_tokens=int(selected.sum()), tokens_per_sequence=selected.sum(1).tolist(),
                          direction_sha256=hashlib.sha256(intervention.direction.tobytes()).hexdigest(),
                          direction_l2_norm=float(np.linalg.norm(intervention.direction.astype(np.float64))),
                          delta_l2_norm_float32=float(np.linalg.norm(delta.astype(np.float64))))

        def callback(hidden, layer, site):
            if intervention is None or layer != intervention.layer or site != intervention.site:
                return hidden
            report["site_hits"] += 1
            if report["site_hits"] != 1:
                raise RuntimeError("My selected intervention site ran more than once")
            if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(selected.shape) or hidden.shape[-1] != len(delta):
                raise ValueError("My hidden-state shape does not match the mask and direction")
            report["hidden_dtype"] = str(hidden.dtype)
            if intervention.magnitude == 0:
                return hidden
            mask_tensor = torch.as_tensor(selected, device=hidden.device, dtype=torch.bool)
            delta_tensor = torch.as_tensor(delta, device=hidden.device, dtype=hidden.dtype)
            changed = hidden.clone()
            changed[mask_tensor] = changed[mask_tensor] + delta_tensor
            report["applied_injections"] += 1
            return changed

        with torch.no_grad():
            result = adapted(model, input_ids, attention_mask, return_hidden_states=return_hidden_states,
                             **{_CALLBACK: callback})
        if intervention is not None and report["site_hits"] != 1:
            raise RuntimeError("My selected intervention site did not run exactly once")
        if return_hidden_states:
            for key in ("all_pre_mlp_hidden_states", "all_hidden_states"):
                if len(result[key]) != n_layers:
                    raise RuntimeError(f"I did not capture all {n_layers} layers in {key}")
        result["intervention_report"] = report
        return result

    forward.source_metadata = copy.deepcopy(metadata)
    return forward


def self_test(author_source=None):
    """I test my source patch and propagation with deterministic NumPy tensors."""
    import contextlib
    import importlib.util
    import sys
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        shape = property(lambda self: self.data.shape)
        ndim = property(lambda self: self.data.ndim)
        dtype = property(lambda self: self.data.dtype)
        device = "cpu"

        def __add__(self, other):
            return Tensor(self.data + (other.data if isinstance(other, Tensor) else other))

        def __mul__(self, other):
            return Tensor(self.data * other)

        def __getitem__(self, key):
            return Tensor(self.data[key.data if isinstance(key, Tensor) else key])

        def __setitem__(self, key, value):
            self.data[key.data if isinstance(key, Tensor) else key] = value.data

        def clone(self):
            return Tensor(self.data.copy())

        def view(self, *shape):
            return Tensor(self.data.reshape(*shape))

        def detach(self):
            return self

        def cpu(self):
            return self.clone()

        def numpy(self):
            return self.data

    fake_torch = SimpleNamespace(bool=np.bool_, no_grad=contextlib.nullcontext,
                                 as_tensor=lambda x, device, dtype: Tensor(np.asarray(x, dtype=dtype)))
    source = '''
def run_gptoss_return_topk(model, input_ids, attention_mask, return_hidden_states=False):
    hidden_state = model.embed(input_ids)
    B, N, D = hidden_state.shape
    all_pre_mlp_hidden_states, all_hidden_states = [], []
    for layer_ix, layer in enumerate(model.model.layers):
        residual = hidden_state
        hidden_state = layer.attention(hidden_state)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, D).detach().cpu())
        is_mxfp4 = layer.is_mxfp4
        if is_mxfp4:
            routed_out = layer.mlp(hidden_state)
            hidden_state = residual + routed_out
            if return_hidden_states:
                all_hidden_states.append(hidden_state.view(-1, D).detach().cpu())
        else:
            final_hidden_states = layer.mlp(hidden_state).view(-1, D)
            hidden_state = residual + final_hidden_states.view(B, N, D)
            if return_hidden_states:
                all_hidden_states.append(hidden_state.view(-1, D).detach().cpu())
    return dict(logits=hidden_state, all_pre_mlp_hidden_states=all_pre_mlp_hidden_states,
                all_hidden_states=all_hidden_states)
'''
    original_globals = {}
    exec(source, original_globals)
    original = original_globals["run_gptoss_return_topk"]
    copied, metadata = _instrument_source(source, {}, "<my-deterministic-toy>")
    forward = _wrap_forward(copied, metadata, original)
    with tempfile.TemporaryDirectory(prefix="my-forward-test-") as directory:
        path = Path(directory) / "toy.py"
        path.write_text(source)
        module_spec = importlib.util.spec_from_file_location("my_toy_forward", path)
        source_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(source_module)
        source_original = source_module.run_gptoss_return_topk

        @functools.wraps(source_original)
        def decorated(*args, **kwargs):
            return source_original(*args, **kwargs)

        public_forward = adapt_forward(decorated)
        if path.read_text() != source:
            raise AssertionError("My public adapter changed its source file")
    ids = Tensor(np.arange(10).reshape(2, 5))
    attention = Tensor(np.ones((2, 5), dtype=np.int64))
    content = np.ones((2, 5), dtype=bool)
    content[:, 0] = False
    selected = np.zeros((2, 5), dtype=bool)
    selected[0, 1:3] = True
    selected[1, 4] = True
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(call, name):
        try:
            call()
        except (ValueError, TypeError):
            checks.append(name)
        else:
            raise AssertionError(name)

    def equal_results(a, b):
        return np.array_equal(a["logits"].data, b["logits"].data) and all(
            np.array_equal(x.data, y.data) for key in ("all_pre_mlp_hidden_states", "all_hidden_states")
            for x, y in zip(a[key], b[key]))

    with patch.dict(sys.modules, {"torch": fake_torch}):
        for fused in (True, False):
            layers = [SimpleNamespace(is_mxfp4=fused, attention=lambda x: x * 0.02,
                                      post_attention_layernorm=lambda x: x * 0.5, mlp=lambda x: x * 0.1)
                      for _ in range(24)]
            model = SimpleNamespace(model=SimpleNamespace(layers=layers),
                                    embed=lambda x: Tensor(np.repeat((x.data + 1.0)[..., None], 4, axis=2)))
            baseline = original(model, ids, attention, True)
            check(equal_results(baseline, forward(model, ids, attention, True)), f"none bitwise identity; fused={fused}")
            check(equal_results(baseline, public_forward(model, ids, attention, True)),
                  f"public adapter unwrap/inspect/compile identity; fused={fused}")
            for site, layer in (("block_output", 11), ("pre_mlp", 12)):
                zero = Intervention(site, layer, unit_direction([1, 2, 3, 4]), 0)
                result = forward(model, ids, attention, True, intervention=zero, content_mask=content)
                check(equal_results(baseline, result), f"zero bitwise identity at {site}; fused={fused}")
                check(result["intervention_report"]["site_hits"] == 1 and result["intervention_report"]["applied_injections"] == 0,
                      f"zero site accounting at {site}; fused={fused}")
                for magnitude in (-0.3, 0.3):
                    spec = Intervention(site, layer, unit_direction([1, 2, 3, 4]), magnitude)
                    result = forward(model, ids, attention, True, intervention=spec, content_mask=content, token_mask=selected)
                    public_result = public_forward(model, ids, attention, True, intervention=spec, content_mask=content, token_mask=selected)
                    check(equal_results(result, public_result), f"public adapter injected result {site}/{magnitude}; fused={fused}")
                    report = result["intervention_report"]
                    check(report["site_hits"] == report["applied_injections"] == 1 and report["selected_tokens"] == 3,
                          f"once and bounded mask {site}/{magnitude}; fused={fused}")
                    first_changed = layer + (site == "block_output")
                    pre = result["all_pre_mlp_hidden_states"]
                    before = baseline["all_pre_mlp_hidden_states"]
                    check(all(np.array_equal(pre[i].data, before[i].data) for i in range(first_changed)),
                          f"earlier layers unchanged {site}/{magnitude}; fused={fused}")
                    check(all(np.all(np.sign(pre[i].data[selected.ravel()] - before[i].data[selected.ravel()]) == np.sign(magnitude))
                              for i in range(first_changed, 24)), f"downstream propagation {site}/{magnitude}; fused={fused}")
                    check(all(np.array_equal(pre[i].data[~selected.ravel()], before[i].data[~selected.ravel()]) for i in range(24)),
                          f"no unselected token change in positionwise toy {site}/{magnitude}; fused={fused}")
                    capture_key = "all_hidden_states" if site == "block_output" else "all_pre_mlp_hidden_states"
                    expected = baseline[capture_key][layer].data.copy()
                    expected[selected.ravel()] += spec.direction * magnitude
                    check(np.array_equal(result[capture_key][layer].data, expected), f"exact selected-site addition {site}/{magnitude}; fused={fused}")
                    check(result["logits"].dtype == baseline["logits"].dtype, f"hidden dtype preserved {site}/{magnitude}; fused={fused}")
            spec = Intervention("pre_mlp", 12, unit_direction([1, 2, 3, 4]), 0.3)
            rejects(lambda: forward(model, ids, attention, True, intervention=spec, content_mask=content, token_mask=~content), "outside-content rejected")
            rejects(lambda: forward(model, ids, attention, True, intervention=spec, content_mask=np.ones((1, 5), bool)), "wrong mask shape rejected")
            rejects(lambda: forward(model, ids, attention, True, intervention=spec, content_mask=content.astype(int)), "nonboolean mask rejected")
            rejects(lambda: forward(model, ids, attention, True, intervention=spec, content_mask=content, token_mask=np.zeros_like(content)), "empty selection rejected")
            padded = Tensor(np.zeros((2, 5), dtype=np.int64))
            rejects(lambda: forward(model, ids, padded, True, intervention=spec, content_mask=content), "padding rejected")
            bad_layer = Intervention("pre_mlp", 24, spec.direction, 0.3)
            rejects(lambda: forward(model, ids, attention, True, intervention=bad_layer, content_mask=content), "out-of-range layer rejected")
            bad_width = Intervention("pre_mlp", 12, unit_direction([1, 2]), 0.3)
            rejects(lambda: forward(model, ids, attention, True, intervention=bad_width, content_mask=content), "wrong vector width rejected")
    rejects(lambda: unit_direction([0, 0]), "zero direction rejected")
    rejects(lambda: unit_direction([1, np.nan]), "nonfinite direction rejected")
    rejects(lambda: Intervention("pre_mlp", 12, np.array([2.0]), 1), "nonunit direction rejected")
    rejects(lambda: Intervention("pre_mlp", 12, np.array([1.0]), math.inf), "nonfinite magnitude rejected")
    rejects(lambda: _instrument_source(source.replace("hidden_state = residual + routed_out", "hidden_state = routed_out"), {}, "<bad>"), "missing residual branch rejected")
    rejects(lambda: _instrument_source(source.replace("hidden_state = layer.post_attention_layernorm(hidden_state)",
                                                     "hidden_state = layer.post_attention_layernorm(hidden_state)\n        hidden_state = layer.post_attention_layernorm(hidden_state)"), {}, "<bad>"), "duplicate pre-MLP site rejected")
    if author_source is not None:
        path = Path(author_source)
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gptoss_return_topk")
        runtime_source = ast.get_source_segment(path.read_text(), node)
        _, actual_metadata = _instrument_source(runtime_source, {}, str(path))
        check(actual_metadata["validated_site_matches"] == {"pre_mlp": 1, "block_output_mxfp4": 1, "block_output_dense": 1},
              "actual frozen source has exactly the three intended sites")
    return {"passed": len(checks), "checks": checks, "model_or_gpu_used": False}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("I expose only --self-test here; I import adapt_forward from my runner")
    print(json.dumps(self_test(args.source), indent=2))
