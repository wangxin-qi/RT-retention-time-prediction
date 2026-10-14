from typing import Dict, Iterable, Tuple

import numpy as np


def condition_distance(source: Dict[str, float], target: Dict[str, float]) -> float:
    keys = sorted(set(source) | set(target))
    if not keys:
        return 0.0
    a = np.asarray([source.get(k, 0.0) for k in keys], dtype=np.float64)
    b = np.asarray([target.get(k, 0.0) for k in keys], dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    return float(np.mean(np.abs(a - b) / scale))


def route_uncertainty_score(row: Dict[str, object]) -> float:
    score = float(row.get("pred_rt_v48_local_similarity_top1_score") or 0.0)
    margin = float(row.get("pred_rt_v48_local_similarity_score_margin") or 0.0)
    route_l1 = float(row.get("pred_rt_v48_local_similarity_top1_route_l1") or 0.0)
    public_fallback = str(row.get("pred_rt_v48_public_safe_fallback", "")).lower() == "true"
    uncertainty = 0.0
    uncertainty += max(0.0, 0.98 - score)
    uncertainty += max(0.0, 0.05 - margin)
    uncertainty += min(route_l1 / 10.0, 1.0)
    uncertainty += 0.25 if public_fallback else 0.0
    return float(uncertainty)


def expert_disagreement(predictions: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(predictions), dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.std(arr)), float(np.max(arr) - np.min(arr))
