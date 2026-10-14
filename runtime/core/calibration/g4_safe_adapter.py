"""Leakage-safe G4 anchor adapter used for 10/50-anchor deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    strength: float = 0.0
    complexity_penalty: float = 0.0


CANDIDATES = (
    Candidate("identity", "identity"),
    Candidate("mean_offset_s2", "mean_offset", 2.0, 0.005),
    Candidate("mean_offset_s5", "mean_offset", 5.0, 0.005),
    Candidate("mean_offset_s10", "mean_offset", 10.0, 0.005),
    Candidate("median_offset_s2", "median_offset", 2.0, 0.008),
    Candidate("median_offset_s5", "median_offset", 5.0, 0.008),
    Candidate("median_offset_s10", "median_offset", 10.0, 0.008),
    Candidate("ridge_affine_a1", "ridge_affine", 1.0, 0.015),
    Candidate("ridge_affine_a10", "ridge_affine", 10.0, 0.012),
    Candidate("ridge_affine_a100", "ridge_affine", 100.0, 0.010),
)


def _fit(candidate: Candidate, prediction: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=float)
    observed = np.asarray(observed, dtype=float)
    residual = observed - prediction
    n = max(len(prediction), 1)
    if candidate.kind == "identity":
        return {"kind": "identity"}
    shrinkage = n / (n + candidate.strength)
    if candidate.kind == "mean_offset":
        return {"kind": candidate.kind, "offset": float(shrinkage * np.mean(residual))}
    if candidate.kind == "median_offset":
        return {"kind": candidate.kind, "offset": float(shrinkage * np.median(residual))}
    if candidate.kind == "ridge_affine":
        center = float(np.mean(prediction))
        scale = max(float(np.std(prediction)), 1e-8)
        z = (prediction - center) / scale
        design = np.column_stack([np.ones(n), z])
        penalty = np.diag([candidate.strength * 0.25, candidate.strength])
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ residual
        )
        return {
            "kind": candidate.kind,
            "center": center,
            "scale": scale,
            "offset": float(coefficients[0]),
            "slope_delta": float(coefficients[1]),
        }
    raise ValueError(f"Unknown G4 candidate: {candidate.kind}")


def _apply(parameters: dict[str, Any], prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    kind = str(parameters["kind"])
    if kind == "identity":
        return prediction.copy()
    if kind in {"mean_offset", "median_offset"}:
        return prediction + float(parameters["offset"])
    if kind == "ridge_affine":
        z = (prediction - float(parameters["center"])) / float(parameters["scale"])
        return (
            prediction
            + float(parameters["offset"])
            + float(parameters["slope_delta"]) * z
        )
    raise ValueError(f"Unknown fitted G4 candidate: {kind}")


def _cross_validated_scores(
    prediction: np.ndarray, observed: np.ndarray, folds: int = 5
) -> pd.DataFrame:
    fold_count = min(max(2, int(folds)), len(prediction))
    fold_id = np.arange(len(prediction)) % fold_count
    rows = []
    for candidate in CANDIDATES:
        errors = []
        for fold in range(fold_count):
            train = fold_id != fold
            valid = fold_id == fold
            parameters = _fit(candidate, prediction[train], observed[train])
            estimated = _apply(parameters, prediction[valid])
            errors.extend((estimated - observed[valid]).tolist())
        error = np.asarray(errors, dtype=float)
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        rows.append(
            {
                "candidate": candidate.name,
                "cv_mae_min": mae,
                "cv_rmse_min": rmse,
                "selection_score": mae + 0.15 * rmse + candidate.complexity_penalty,
            }
        )
    return pd.DataFrame(rows).sort_values(["selection_score", "candidate"])


@dataclass
class G4SafeAnchorAdapter:
    base_pred_col: str
    target_col: str
    anchor_rows: int
    candidate_name: str
    parameters: dict[str, Any]
    identity_cv_mae: float
    identity_cv_rmse: float
    selected_cv_mae: float
    selected_cv_rmse: float
    selection_reason: str
    version: str = "g4_safe_anchor_cv_v1"

    @property
    def changes_prediction(self) -> bool:
        return str(self.parameters.get("kind")) != "identity"

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.base_pred_col not in frame.columns:
            raise ValueError(f"Base prediction column not found: {self.base_pred_col}")
        out = frame.copy()
        base = pd.to_numeric(out[self.base_pred_col], errors="coerce").to_numpy(float)
        adapted = _apply(self.parameters, base)
        out["pred_rt_before_calibration_adapter"] = base
        out["pred_rt_calibration_adapter"] = adapted
        out["pred_rt_with_calibration_adapter"] = adapted
        out["pred_rt_calibration_adapter_weight"] = 1.0 if self.changes_prediction else 0.0
        out["pred_rt_calibration_adapter_weight_reason"] = self.selection_reason
        out["pred_rt_calibration_adapter_train_rows"] = int(self.anchor_rows)
        out["pred_rt_calibration_adapter_model"] = self.version
        out["pred_rt_calibration_adapter_candidate"] = self.candidate_name
        out["pred_rt_calibration_adapter_evaluated"] = True
        out["pred_rt_calibration_adapter_used"] = self.changes_prediction
        return out

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "base_pred_col": self.base_pred_col,
            "target_col": self.target_col,
            "anchor_rows": int(self.anchor_rows),
            "candidate_name": self.candidate_name,
            "parameters": dict(self.parameters),
            "identity_cv_mae": float(self.identity_cv_mae),
            "identity_cv_rmse": float(self.identity_cv_rmse),
            "selected_cv_mae": float(self.selected_cv_mae),
            "selected_cv_rmse": float(self.selected_cv_rmse),
            "selection_reason": self.selection_reason,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "G4SafeAnchorAdapter":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})

    def save(self, path: str) -> None:
        joblib.dump(self.to_payload(), path)


def fit_g4_safe_adapter(
    frame: pd.DataFrame, *, target_col: str, base_pred_col: str
) -> G4SafeAnchorAdapter:
    if target_col not in frame.columns or base_pred_col not in frame.columns:
        raise ValueError("G4 fitting requires observed and base-prediction columns.")
    numeric = frame[[target_col, base_pred_col]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[np.isfinite(numeric.to_numpy(float)).all(axis=1)]
    if len(numeric) < 10:
        raise ValueError("G4 safe adaptation requires at least 10 valid anchors.")
    observed = numeric[target_col].to_numpy(float)
    prediction = numeric[base_pred_col].to_numpy(float)
    scores = _cross_validated_scores(prediction, observed)
    identity = scores.loc[scores["candidate"].eq("identity")].iloc[0]
    best = scores.iloc[0]
    required_gain = max(0.03, 0.05 * float(identity["cv_mae_min"]))
    passes = (
        str(best["candidate"]) != "identity"
        and float(identity["cv_mae_min"]) - float(best["cv_mae_min"]) >= required_gain
        and float(best["cv_rmse_min"]) <= float(identity["cv_rmse_min"]) + 0.01
    )
    selected = best if passes else identity
    candidate = next(x for x in CANDIDATES if x.name == str(selected["candidate"]))
    return G4SafeAnchorAdapter(
        base_pred_col=base_pred_col,
        target_col=target_col,
        anchor_rows=int(len(numeric)),
        candidate_name=candidate.name,
        parameters=_fit(candidate, prediction, observed),
        identity_cv_mae=float(identity["cv_mae_min"]),
        identity_cv_rmse=float(identity["cv_rmse_min"]),
        selected_cv_mae=float(selected["cv_mae_min"]),
        selected_cv_rmse=float(selected["cv_rmse_min"]),
        selection_reason="anchor_cv_gain_passed" if passes else "safe_identity_fallback",
    )


def apply_g4_adapter_to_frame(
    frame: pd.DataFrame,
    adapter_path: str,
    *,
    update_final: bool = True,
    final_col: str | None = None,
) -> pd.DataFrame:
    payload = joblib.load(adapter_path)
    adapter = G4SafeAnchorAdapter.from_payload(payload)
    out = adapter.apply(frame)
    if update_final:
        column = final_col or adapter.base_pred_col
        out[column] = out["pred_rt_with_calibration_adapter"]
    return out
