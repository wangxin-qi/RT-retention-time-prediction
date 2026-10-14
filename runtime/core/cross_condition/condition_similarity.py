"""Condition-weighted chromatographic similarity.

This replaces the old "raw cosine over the route signature" similarity used to decide
expert routing. Each chromatographic condition is given a learned weight equal to how
many minutes of RT discrepancy a unit difference in that condition causes (fitted from
same-molecule cross-method pairs; see ``core/train/train_condition_similarity_weights.py``).

Distance between two methods is therefore an *expected RT disagreement in minutes*:

    distance(a, b) = sum_k  w_k * |feature_a[k] - feature_b[k]|
    similarity      = exp(-distance / tau)   in (0, 1]

similarity == 1 means identical conditions; it decays as the conditions that actually
matter for RT diverge. The router turns this similarity into a soft alpha (0 .. alpha_max)
instead of the old "looks similar -> treat as an explicit method code -> alpha = 1.0".
"""

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from core.cross_condition.condition_features import (
    metadata_to_condition_features,
    phase_flags_from_metadata,
)


# Canonical, interpretable feature order shared by the trainer and the runtime.
FEATURE_NAMES: List[str] = [
    "column_length",
    "column_id",
    "particle_size",
    "flowrate",
    "temperature",
    "t0",
    "ph_a",
    "ph_b",
    "init_b",
    "final_b",
    "gradient_span",
    "gradient_segments",
    "gradient_time_proxy",
    "gradient_slope_proxy",
    "gradient_missing",
    "organic_a",
    "organic_b",
    "solvent_is_acn",
    "solvent_is_meoh",
    "solvent_is_mixed",
    "solvent_is_unknown",
    "phase_c18",
    "phase_hss_t3",
    "phase_hilic",
    "phase_phenyl",
    "phase_unknown",
]


def method_feature_dict(meta: Dict[str, Any]) -> Dict[str, float]:
    """Named condition features for one method's metadata dict."""
    features = dict(metadata_to_condition_features(meta))
    features.update(phase_flags_from_metadata(meta))
    return features


def method_feature_vector(meta: Dict[str, Any], feature_names: Optional[List[str]] = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES
    features = method_feature_dict(meta)
    return np.asarray([float(features.get(name, 0.0)) for name in names], dtype=np.float64)


class ConditionSimilarity:
    """Weighted-distance similarity loaded from a learned weights JSON."""

    def __init__(
        self,
        feature_names: List[str],
        weights: np.ndarray,
        tau: float,
        feature_scale: Optional[np.ndarray] = None,
        intercept: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.feature_names = list(feature_names)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if self.weights.shape[0] != len(self.feature_names):
            raise ValueError("weights length does not match feature_names length")
        if feature_scale is None:
            self.feature_scale = np.ones_like(self.weights)
        else:
            self.feature_scale = np.asarray(feature_scale, dtype=np.float64).reshape(-1)
            self.feature_scale = np.where(self.feature_scale < 1e-9, 1.0, self.feature_scale)
        if self.feature_scale.shape[0] != len(self.feature_names):
            raise ValueError("feature_scale length does not match feature_names length")
        self.tau = float(tau) if float(tau) > 1e-6 else 1.0
        self.intercept = float(intercept)
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path: str) -> "ConditionSimilarity":
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        feature_scale = payload.get("feature_scale")
        return cls(
            feature_names=list(payload["feature_names"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            tau=float(payload.get("tau", 1.0)),
            feature_scale=None if feature_scale is None else np.asarray(feature_scale, dtype=np.float64),
            intercept=float(payload.get("intercept", 0.0)),
            meta={k: v for k, v in payload.items() if k not in {"feature_names", "weights", "feature_scale"}},
        )

    def vector(self, meta: Dict[str, Any]) -> np.ndarray:
        return method_feature_vector(meta, self.feature_names)

    def distance_vec(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        # Distance in expected |delta RT| minutes: weights act on standardised differences.
        a = np.asarray(vec_a, dtype=np.float64).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float64).reshape(-1)
        return float(np.sum(self.weights * np.abs(a - b) / self.feature_scale))

    def distance(self, meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> float:
        return self.distance_vec(self.vector(meta_a), self.vector(meta_b))

    def similarity_from_distance(self, distance: float) -> float:
        return float(math.exp(-float(distance) / self.tau))

    def similarity_vec(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return self.similarity_from_distance(self.distance_vec(vec_a, vec_b))

    def similarity(self, meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> float:
        return self.similarity_from_distance(self.distance(meta_a, meta_b))


def load_condition_similarity(path: Optional[str]) -> Optional[ConditionSimilarity]:
    """Best-effort load. Returns None if the artifact is missing/unreadable, so callers
    transparently fall back to legacy behaviour when the weights have not been trained yet."""
    if not path or not os.path.exists(path):
        return None
    try:
        return ConditionSimilarity.load(path)
    except Exception:
        return None
