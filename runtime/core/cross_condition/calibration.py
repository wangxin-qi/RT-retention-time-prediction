import json
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class LinearCalibrator:
    slope: float
    intercept: float

    def predict(self, values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(values), dtype=np.float64)
        return self.slope * arr + self.intercept

    def to_dict(self) -> Dict[str, float]:
        return {"type": "linear", "slope": float(self.slope), "intercept": float(self.intercept)}


@dataclass
class IsotonicCalibrator:
    x: List[float]
    y: List[float]

    def predict(self, values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(values), dtype=np.float64)
        return np.interp(arr, np.asarray(self.x), np.asarray(self.y), left=self.y[0], right=self.y[-1])

    def to_dict(self) -> Dict[str, object]:
        return {"type": "isotonic", "x": [float(v) for v in self.x], "y": [float(v) for v in self.y]}


def fit_linear(predicted: Iterable[float], observed: Iterable[float]) -> LinearCalibrator:
    x = np.asarray(list(predicted), dtype=np.float64)
    y = np.asarray(list(observed), dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 1e-12:
        return LinearCalibrator(slope=1.0, intercept=float(np.mean(y - x)) if len(x) else 0.0)
    slope, intercept = np.polyfit(x, y, deg=1)
    return LinearCalibrator(float(slope), float(intercept))


def fit_isotonic(predicted: Iterable[float], observed: Iterable[float]) -> IsotonicCalibrator:
    x = np.asarray(list(predicted), dtype=np.float64)
    y = np.asarray(list(observed), dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    blocks = [[float(xi), float(xi), float(yi), 1.0] for xi, yi in zip(x, y)]
    idx = 0
    while idx < len(blocks) - 1:
        if blocks[idx][2] <= blocks[idx + 1][2]:
            idx += 1
            continue
        total_w = blocks[idx][3] + blocks[idx + 1][3]
        avg_y = (blocks[idx][2] * blocks[idx][3] + blocks[idx + 1][2] * blocks[idx + 1][3]) / total_w
        blocks[idx : idx + 2] = [[blocks[idx][0], blocks[idx + 1][1], avg_y, total_w]]
        idx = max(idx - 1, 0)
    iso_x = []
    iso_y = []
    for lo, hi, val, _ in blocks:
        iso_x.extend([lo, hi])
        iso_y.extend([val, val])
    return IsotonicCalibrator(x=iso_x, y=iso_y)


def save_calibrator(calibrator, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibrator.to_dict(), handle, indent=2)

