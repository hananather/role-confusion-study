# TOY LAB (MATS 12.0 application experiment): dummy secrets, loopback receiver, isolated sandbox; runtime-only activation edits on a local cached gpt-oss-20b, weights never modified. See ../../TOY-LAB-NOTICE.md. Never copy this notice into model-facing prompts, fixtures, or trajectories.
"""Span-masked activation steering for gpt-oss-20b in MLX, plus a probe readout.

MaskedBlockHook wraps one TransformerBlock (zero-based index L). It adds `vec` to the block
output only at absolute prompt positions where `mask` is 1. Positions are tracked by counting
tokens per call (reset per generation) and cross-checked against the per-layer KV cache offset,
which mlx_lm advances by the chunk size on every call. Generated tokens sit past the prompt
length and are never edited. vec=None is an identity path.

ProbeRecorder wraps post_attention_layernorm of a later layer (the authors' probe site) and
accumulates mean probe probabilities over named spans during prefill: the manipulation check.
"""
from __future__ import annotations
import numpy as np


class MaskedBlockHook:
    def __init__(self, block):
        self.block = block
        self.vec = None
        self.mask = None
        self.prompt_len = 0
        self.pos = 0
        self.calls = 0
        self.edited = 0.0
        self.offset_mismatch = 0
        self.steer_generated = False

    def set(self, vec, mask, prompt_len: int):
        import mlx.core as mx
        self.vec = None if vec is None else mx.array(np.asarray(vec, np.float32))
        self.mask = None if mask is None else mx.array(np.asarray(mask, np.float32))
        self.prompt_len = int(prompt_len)
        self.pos, self.calls, self.edited, self.offset_mismatch = 0, 0, 0.0, 0

    def __call__(self, x, mask, cache=None):
        import mlx.core as mx
        start = self.pos
        if cache is not None and hasattr(cache, "offset"):
            off = cache.offset
            if isinstance(off, int) and off != start:
                self.offset_mismatch += 1
        y = self.block(x, mask, cache)
        n = int(y.shape[1])
        self.pos = start + n
        self.calls += 1
        if self.vec is None or self.mask is None:
            return y
        if start >= self.prompt_len:
            if not self.steer_generated:
                return y
            self.edited += n
            return y + self.vec[None, None, :].astype(y.dtype)
        m = self.mask[start:start + n]
        if int(m.shape[0]) < n:
            m = mx.concatenate([m, mx.zeros((n - int(m.shape[0]),), dtype=m.dtype)])
        s = m.sum()
        mx.eval(s)
        if float(s.item()) == 0.0:
            return y
        self.edited += float(s.item())
        return y + (m[None, :, None] * self.vec[None, None, :]).astype(y.dtype)


class ProbeRecorder:
    """Wraps an RMSNorm module; accumulates softmax probe probabilities over spans during prefill."""

    def __init__(self, wrapped, coef: np.ndarray, intercept: np.ndarray, roles: list[str]):
        import mlx.core as mx
        self.wrapped = wrapped
        self.W = mx.array(coef.astype(np.float32))
        self.b = mx.array(intercept.astype(np.float32))
        self.roles = roles
        self.spans = {}
        self.prompt_len = 0
        self.pos = 0
        self.sums = {}
        self.counts = {}

    def set(self, spans: dict, prompt_len: int):
        import mlx.core as mx
        self.spans = {k: mx.array(np.asarray(v, np.float32)) for k, v in spans.items()}
        self.prompt_len = int(prompt_len)
        self.pos = 0
        self.sums = {k: np.zeros(len(self.roles), np.float64) for k in spans}
        self.counts = {k: 0.0 for k in spans}

    def __call__(self, x):
        import mlx.core as mx
        y = self.wrapped(x)
        start = self.pos
        n = int(y.shape[1])
        self.pos = start + n
        if not self.spans or start >= self.prompt_len:
            return y
        logits = y[0].astype(mx.float32) @ self.W.T + self.b
        probs = mx.softmax(logits, axis=-1)
        for k, full in self.spans.items():
            m = full[start:start + n]
            if int(m.shape[0]) < n:
                m = mx.concatenate([m, mx.zeros((n - int(m.shape[0]),), dtype=m.dtype)])
            s = (probs * m[:, None]).sum(0)
            c = m.sum()
            mx.eval(s, c)
            self.sums[k] += np.array(s, np.float64)
            self.counts[k] += float(c.item())
        return y

    def means(self) -> dict:
        out = {}
        for k in self.sums:
            c = self.counts[k]
            out[k] = {"n_tokens": int(c), **({f"p_{r}": float(v) for r, v in zip(self.roles, self.sums[k] / c)} if c else {})}
        return out


def install(model, steer_layer: int, probe_layer: int, coef, intercept, roles):
    inner = model.model
    hook = MaskedBlockHook(inner.layers[steer_layer])
    inner.layers[steer_layer] = hook
    rec = ProbeRecorder(inner.layers[probe_layer].post_attention_layernorm, coef, intercept, roles)
    inner.layers[probe_layer].post_attention_layernorm = rec
    return hook, rec
