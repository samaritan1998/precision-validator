"""Streaming FP64 statistics over the stored FP32 vectors."""
import math
import numpy as np
from .common import require


class Metrics:
    def __init__(self):
        self.elements = 0
        self.diff2 = self.reference2 = self.actual2 = self.dot = self.max_abs = 0.0

    def add(self, actual, reference):
        x = np.asarray(actual, dtype=np.float64).reshape(-1)
        y = np.asarray(reference, dtype=np.float64).reshape(-1)
        require(x.shape == y.shape and x.size > 0, "Metric shape mismatch or empty tensor")
        require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite tensor value")
        d = x - y
        self.elements += x.size
        self.diff2 += float(np.dot(d, d))
        self.reference2 += float(np.dot(y, y))
        self.actual2 += float(np.dot(x, x))
        self.dot += float(np.dot(x, y))
        self.max_abs = max(self.max_abs, float(np.max(np.abs(d))))
        return self

    def merge(self, other):
        for key in ("elements", "diff2", "reference2", "actual2", "dot"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.max_abs = max(self.max_abs, other.max_abs)
        return self

    def result(self):
        require(self.elements > 0, "No tensor elements compared")
        # Zero-reference nonzero candidates explicitly fail any relative gate;
        # JSON stays strict (no Infinity or NaN).
        relative = math.sqrt(self.diff2 / self.reference2) if self.reference2 else (0.0 if not self.diff2 else None)
        cosine = self.dot / math.sqrt(self.reference2 * self.actual2) if self.reference2 and self.actual2 else (1.0 if not self.diff2 else 0.0)
        return {"elements": self.elements, "diff2": self.diff2, "reference2": self.reference2,
                "actual2": self.actual2, "dot": self.dot, "max_abs": self.max_abs,
                "absolute_l2": math.sqrt(self.diff2), "rmse": math.sqrt(self.diff2 / self.elements),
                "relative_l2": relative, "cosine": max(-1.0, min(1.0, cosine)), "finite": True}


def evaluate(repeat, target, policy):
    noise, delta = repeat["relative_l2"], target["relative_l2"]
    limit = min(policy["target_relative_ceiling"], max(policy["noise_multiplier"] * noise,
                noise + policy["noise_slack"])) if noise is not None else None
    passed = noise is not None and delta is not None and noise <= policy["repeat_relative_ceiling"] and delta <= limit
    reasons = []
    if noise is None or noise > policy["repeat_relative_ceiling"]:
        reasons.append("Reference repeat exceeds the fixed repeat ceiling")
    if delta is None or limit is None or delta > limit:
        reasons.append("Candidate exceeds the noise-aware or fixed target ceiling")
    for metric, key in (("max_abs", "max_abs_ceiling"), ("rmse", "rmse_ceiling")):
        ceiling = policy.get(key)
        if ceiling is not None and max(repeat[metric], target[metric]) > ceiling:
            passed = False
            reasons.append(f"{metric} exceeds fixed ceiling {ceiling}")
    return {"passed": bool(passed), "relative_limit": limit, "policy": policy, "reasons": reasons}
