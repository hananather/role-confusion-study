"""Affine softmax role scores. Random-probe baseline uses a frozen seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROLE_CHAR = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_probe(npz_path: Path, space: str, layer: int):
    archive = np.load(npz_path)
    key = f"{space}_L{layer:02d}"
    coef = archive[f"{key}__coef"].astype(np.float32)
    intercept = archive[f"{key}__intercept"].astype(np.float32)
    roles = [ROLE_CHAR[char] for char in space]
    return coef, intercept, roles


def random_probe(coef: np.ndarray, intercept: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=coef.shape).astype(np.float32), rng.normal(
        size=intercept.shape
    ).astype(np.float32)


def token_probs(hidden: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return softmax(hidden.astype(np.float32) @ coef.T + intercept)


def span_means(probs: np.ndarray, roles: list[str], indices: list[int]) -> dict:
    if not indices:
        return {f"p_{role}": None for role in roles} | {
            "n_tokens": 0,
            "log_user_tool": None,
            "log_cot_tool": None,
        }
    mean = probs[indices].mean(axis=0)
    by_role = {role: float(mean[i]) for i, role in enumerate(roles)}
    user = by_role.get("user")
    cot = by_role.get("cot")
    tool = by_role.get("tool")
    log_user_tool = None if user is None or tool is None else float(np.log(user) - np.log(tool))
    log_cot_tool = None if cot is None or tool is None else float(np.log(cot) - np.log(tool))
    return {
        **{f"p_{role}": by_role[role] for role in roles},
        "n_tokens": len(indices),
        "log_user_tool": log_user_tool,
        "log_cot_tool": log_cot_tool,
    }
