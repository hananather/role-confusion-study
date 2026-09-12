"""Source-local addition at transformer block 11 output.

I edit only controller-identified external prompt tokens. Trusted wrappers and
later generated tokens are not added into. A zero vector is an identity path.

A block-output edit cannot rewrite that block's already written keys and values.
It can change later layers and later tokens. Layer 11 is zero-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field


STEER_LAYER = 11


@dataclass
class SteeringSpec:
    layer: int = STEER_LAYER
    dose: float = 0.0
    vector: list[float] | None = None
    external_indices: list[int] = field(default_factory=list)
    prompt_length: int = 0


class SourceLocalHook:
    """Wrap one TransformerBlock. Tests may pass a numpy-like block."""

    def __init__(self, block, *, layer: int = STEER_LAYER):
        self.block = block
        self.layer = layer
        self.offset = 0
        self.prompt_length = 0
        self.external = set()
        self.vec = None
        self.dose = 0.0
        self.edits = 0
        self.chunk_sizes: list[int] = []
        self.generated_edits = 0

    def reset(self, spec: SteeringSpec) -> None:
        if spec.layer != self.layer:
            raise ValueError(f"hook is installed at {self.layer}, spec asks for {spec.layer}")
        self.offset = 0
        self.prompt_length = spec.prompt_length
        self.external = set(spec.external_indices)
        self.dose = spec.dose
        self.edits = 0
        self.generated_edits = 0
        self.chunk_sizes = []
        if spec.vector is None or spec.dose == 0:
            self.vec = None
        else:
            self.vec = spec.vector

    def __call__(self, x, mask=None, cache=None):
        y = self.block(x, mask, cache)
        n = int(y.shape[1])
        self.chunk_sizes.append(n)
        start = self.offset
        self.offset = start + n
        if self.vec is None:
            return y
        local = [
            i for i in range(n)
            if (start + i) in self.external and (start + i) < self.prompt_length
        ]
        generated = [i for i in range(n) if (start + i) >= self.prompt_length]
        if generated:
            self.generated_edits += 0  # generated tokens are observed, never edited
        if not local:
            return y
        self.edits += len(local)
        return _add_at(y, local, self.vec)

    def one_addition_per_selected_token(self) -> bool:
        return self.edits == len(self.external) or self.vec is None


def _add_at(y, local_indices: list[int], vec):
    """Add vec to selected time steps. Supports numpy and MLX arrays."""
    try:
        import mlx.core as mx
        if isinstance(y, mx.array):
            scale = mx.array(vec).astype(y.dtype)
            pieces = []
            for i in range(int(y.shape[1])):
                slice_i = y[:, i : i + 1, :]
                if i in local_indices:
                    slice_i = slice_i + scale
                pieces.append(slice_i)
            return mx.concatenate(pieces, axis=1)
    except Exception:
        pass
    import numpy as np
    out = np.array(y, copy=True)
    addend = np.asarray(vec, dtype=out.dtype)
    for i in local_indices:
        out[:, i, :] += addend
    return out


def install_source_hook(model, *, layer: int = STEER_LAYER) -> SourceLocalHook:
    inner = model.model if hasattr(model, "model") else model
    block = inner.layers[layer]
    if isinstance(block, SourceLocalHook):
        return block
    hook = SourceLocalHook(block, layer=layer)
    inner.layers[layer] = hook
    return hook
