"""I illustrate my saved probe's response to offsets, without running the LLM."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe-offset-illustration'
OUT.mkdir(exist_ok=True)
SOURCE = HERE.parent / 'outputs/20260911T030050Z'
ACTS = SOURCE / 'appendix-e-L12/gardening-layer12-float16.npy'
PROBES = SOURCE / 'probes-full/probes.npz'
TOKENS = SOURCE / 'appendix-e-L12/all-tokens-and-labels.csv'
DISPLAY = SOURCE / 'appendix-e-L12/figure-7-displayed-rows.csv'
roles = ['system', 'user', 'cot', 'assistant', 'tool']
h_all = np.load(ACTS, allow_pickle=False).astype(np.float64)
with np.load(PROBES, allow_pickle=False) as archive:
    w = archive['sucat_L12__coef'].astype(np.float64)
    b = archive['sucat_L12__intercept'].astype(np.float64)
tokens = pd.read_csv(TOKENS, keep_default_na=False)
display = pd.read_csv(DISPLAY, keep_default_na=False)
display = display[display.prompt_key == 'everything_in_user_tags'].sort_values('display_token_ix').copy()
assert len(display) == 512 and display.sample_ix.is_unique
assert h_all.shape == (2791, 2880) and w.shape == (5, 2880)
assert np.isfinite(h_all).all() and np.isfinite(w).all() and np.isfinite(b).all()
ref_indices = tokens[tokens.prompt_key == 'everything_in_user_tags'].sample_ix.to_numpy()
reference_norm = float(np.median(np.linalg.norm(h_all[ref_indices], axis=1)))
h = h_all[display.sample_ix.to_numpy()]

def project(states):
    # I use explicit contraction to avoid the local BLAS runtime's spurious warnings.
    logits = np.einsum('nd,kd->nk', states, w, optimize=False) + b
    shifted = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=1, keepdims=True)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-12)
    return logits, p

base_logits, baseline = project(h)
all_rows = []
summaries = []
checks = []
for fraction in [.01, .05]:
    magnitude = fraction * reference_norm
    for row_ix, edited_role in enumerate(['user', 'tool']):
        role_ix = roles.index(edited_role)
        unit = w[role_ix] / np.linalg.norm(w[role_ix])
        for col_ix, sign in enumerate([-1, 0, 1]):
            offset = sign * magnitude * unit
            logits, p = project(h + offset)
            expected = np.einsum('kd,d->k', w, offset, optimize=False)
            err = float(np.max(np.abs((logits - base_logits) - expected)))
            assert err < 1e-10
            checks.append({'fraction': fraction, 'role': edited_role, 'sign': sign, 'max_logit_identity_error': err})
            frame = display[['sample_ix', 'display_token_ix', 'base_message_type', 'seg_ix']].copy()
            frame['offset_fraction'] = fraction
            frame['edited_role'] = edited_role
            frame['sign'] = sign
            for k, role in enumerate(roles):
                frame[f'p_{role}'] = p[:, k]
            all_rows.append(frame)
            summaries.append({'offset_fraction': fraction, 'edited_role': edited_role, 'sign': sign,
                              'offset_norm': abs(sign)*magnitude,
                              **{f'mean_p_{role}': float(p[:, k].mean()) for k, role in enumerate(roles)}})
pd.concat(all_rows, ignore_index=True).to_csv(OUT/'plotted-probabilities.csv', index=False)
pd.DataFrame(summaries).to_csv(OUT/'summary.csv', index=False)
(OUT/'provenance.json').write_text(json.dumps({
    'evidence_type': 'offline derived probe-response illustration; not a model steering run',
    'sources': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in [ACTS, PROBES, TOKENS, DISPLAY]],
    'role_order': roles, 'probe': 'prompt-split sucat_L12', 'activation_site': 'post_attention_layernorm / pre-MLP',
    'reference_population': 'every forwarded token in everything_in_user_tags', 'reference_token_count': len(ref_indices),
    'reference_median_activation_norm': reference_norm, 'displayed_tokens': 512,
    'arbitrary_illustration_fractions': [.01, .05], 'formula': 'h_prime = h + sign * fraction * median_norm * w_role / norm(w_role)',
    'precision': 'saved float16 activations and float32 probe weights promoted to float64 for local arithmetic',
    'new_model_forward': False, 'downstream_continuation': False, 'behavior_measured': False,
    'role_weight_caveat': 'I use the raw saved classifier rows; individual rows depend on the chosen equivalent softmax representation. The User-minus-Tool row difference is invariant to a common row shift.',
    'subtraction_caveat': 'I apply a constant opposite offset, not removal of the activation projection along a direction.',
    'validation': checks
}, indent=2)+'\n')
print(OUT)
print(pd.DataFrame(summaries)[['offset_fraction','edited_role','sign','mean_p_user','mean_p_tool']].to_string(index=False))

from paper_style_figures import offsets
offsets()
