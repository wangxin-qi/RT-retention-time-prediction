import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


DEFAULT_BASE_PRED_COLS = [
    "pred_rt_v48_mean",
    "pred_rt_v48",
    "pred_rt_public_global",
    "pred_rt_v48_public_global",
    "pred_rt_public_global_order_score",
    "pred_rt_v48_public_global_order_score",
    "pred_rt_cross_condition",
    "pred_rt_v48_cross_condition",
]

DEFAULT_FEATURE_CANDIDATES = [
    "pred_rt_v48_mean",
    "pred_rt_v48",
    "pred_rt_public_global",
    "pred_rt_v48_public_global",
    "pred_rt_public_global_error_est",
    "pred_rt_v48_public_global_error_est",
    "pred_rt_cross_condition",
    "pred_rt_v48_cross_condition",
    "pred_rt_cross_condition_error_est",
    "pred_rt_v48_cross_condition_error_est",
    "pred_rt_v48_local_similarity_top1_score",
    "pred_rt_v48_local_similarity_score_margin",
    "pred_rt_v48_gate_strength",
    "pred_rt_v48_alpha",
    "pred_rt_v48_alpha_base",
    "pred_rt_v48_family_confidence",
    "pred_rt_v48_family_margin",
    "pred_rt_v48_top_expert_weight",
    "pred_rt_v48_expert_margin",
]


def _to_float_array(values: Iterable[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {"rows": 0, "mae": float("nan"), "rmse": float("nan"), "median_ae": float("nan")}
    err = np.abs(y_pred - y_true)
    return {
        "rows": int(len(y_true)),
        "mae": float(np.mean(err)),
        "rmse": float(math.sqrt(np.mean((y_pred - y_true) ** 2))),
        "median_ae": float(np.median(err)),
    }


def choose_base_prediction_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Base prediction column not found: {preferred}")
        return preferred
    for col in DEFAULT_BASE_PRED_COLS:
        if col in df.columns:
            return col
    raise ValueError("No base prediction column found. Pass --base-pred-col explicitly.")


@dataclass
class SoftWeightPolicy:
    reference_max_rows: int = 99
    low_max_rows: int = 299
    medium_max_rows: int = 999
    low_weight: float = 0.20
    medium_weight: float = 0.50
    high_weight: float = 0.80
    min_relative_mae_gain: float = 0.02
    require_validation_gain: bool = True

    def weight_for(self, train_rows: int, validation: Optional[Dict[str, object]] = None) -> Tuple[float, str]:
        rows = int(train_rows)
        if rows <= self.reference_max_rows:
            return 0.0, "reference_only_lt_100"

        if self.require_validation_gain and validation:
            base = validation.get("base_metrics", {}) if isinstance(validation.get("base_metrics"), dict) else {}
            adapter = validation.get("adapter_metrics", {}) if isinstance(validation.get("adapter_metrics"), dict) else {}
            base_mae = float(base.get("mae", float("nan")))
            adapter_mae = float(adapter.get("mae", float("nan")))
            if np.isfinite(base_mae) and np.isfinite(adapter_mae) and base_mae > 0:
                rel_gain = (base_mae - adapter_mae) / base_mae
                if rel_gain < float(self.min_relative_mae_gain):
                    return 0.0, "validation_gain_below_threshold"

        if rows <= self.low_max_rows:
            return float(self.low_weight), f"low_weight_{self.reference_max_rows + 1}_{self.low_max_rows}"
        if rows <= self.medium_max_rows:
            return float(self.medium_weight), f"medium_weight_{self.low_max_rows + 1}_{self.medium_max_rows}"
        return float(self.high_weight), f"high_weight_gt_{self.medium_max_rows}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "reference_max_rows": int(self.reference_max_rows),
            "low_max_rows": int(self.low_max_rows),
            "medium_max_rows": int(self.medium_max_rows),
            "low_weight": float(self.low_weight),
            "medium_weight": float(self.medium_weight),
            "high_weight": float(self.high_weight),
            "min_relative_mae_gain": float(self.min_relative_mae_gain),
            "require_validation_gain": bool(self.require_validation_gain),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "SoftWeightPolicy":
        return cls(**{key: value[key] for key in cls().__dict__.keys() if key in value})


@dataclass
class LightweightCalibrationAdapter:
    feature_columns: List[str]
    coefficients: List[float]
    intercept: float
    base_pred_col: str
    target_col: str
    train_rows: int
    blend_weight: float
    weight_reason: str
    feature_medians: Dict[str, float]
    l2: float = 1.0
    version: str = "lightweight_ridge_residual_v1"
    validation: Optional[Dict[str, object]] = None
    policy: Optional[Dict[str, object]] = None

    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
            else:
                values = pd.Series(np.nan, index=df.index)
            fill = float(self.feature_medians.get(col, 0.0))
            out[col] = values.fillna(fill).astype(float)
        return out

    def predict_adapter(self, df: pd.DataFrame) -> np.ndarray:
        x = self._feature_frame(df).to_numpy(dtype=np.float64)
        coef = np.asarray(self.coefficients, dtype=np.float64)
        return x @ coef + float(self.intercept)

    def apply(self, df: pd.DataFrame, *, mode: str = "auto") -> pd.DataFrame:
        if self.base_pred_col not in df.columns:
            raise ValueError(f"Base prediction column not found in input: {self.base_pred_col}")
        out = df.copy()
        base = pd.to_numeric(out[self.base_pred_col], errors="coerce").to_numpy(dtype=np.float64)
        adapter_pred = self.predict_adapter(out)
        weight = float(self.blend_weight)
        if mode == "report":
            weight = 0.0
        elif mode == "force":
            weight = max(weight, 1.0)
        elif mode not in {"auto", "blend"}:
            raise ValueError(f"Unknown adapter mode: {mode}")

        blended = (1.0 - weight) * base + weight * adapter_pred
        out["pred_rt_calibration_adapter"] = adapter_pred
        out["pred_rt_calibration_adapter_weight"] = weight
        out["pred_rt_calibration_adapter_weight_reason"] = self.weight_reason
        out["pred_rt_calibration_adapter_train_rows"] = int(self.train_rows)
        out["pred_rt_calibration_adapter_model"] = self.version
        out["pred_rt_before_calibration_adapter"] = base
        out["pred_rt_with_calibration_adapter"] = blended
        out["pred_rt_calibration_adapter_used"] = bool(weight > 0.0)
        return out

    def to_payload(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "feature_columns": list(self.feature_columns),
            "coefficients": [float(v) for v in self.coefficients],
            "intercept": float(self.intercept),
            "base_pred_col": self.base_pred_col,
            "target_col": self.target_col,
            "train_rows": int(self.train_rows),
            "blend_weight": float(self.blend_weight),
            "weight_reason": self.weight_reason,
            "feature_medians": {str(k): float(v) for k, v in self.feature_medians.items()},
            "l2": float(self.l2),
            "validation": self.validation or {},
            "policy": self.policy or {},
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "LightweightCalibrationAdapter":
        return cls(
            feature_columns=list(payload["feature_columns"]),
            coefficients=[float(v) for v in payload["coefficients"]],
            intercept=float(payload["intercept"]),
            base_pred_col=str(payload["base_pred_col"]),
            target_col=str(payload.get("target_col", "rt_minutes")),
            train_rows=int(payload["train_rows"]),
            blend_weight=float(payload.get("blend_weight", 0.0)),
            weight_reason=str(payload.get("weight_reason", "")),
            feature_medians={str(k): float(v) for k, v in dict(payload.get("feature_medians", {})).items()},
            l2=float(payload.get("l2", 1.0)),
            version=str(payload.get("version", "lightweight_ridge_residual_v1")),
            validation=dict(payload.get("validation", {})),
            policy=dict(payload.get("policy", {})),
        )

    def save(self, path: str) -> None:
        joblib.dump(self.to_payload(), path)

    @classmethod
    def load(cls, path: str) -> "LightweightCalibrationAdapter":
        payload = joblib.load(path)
        return cls.from_payload(payload)


def infer_feature_columns(df: pd.DataFrame, base_pred_col: str, target_col: str) -> List[str]:
    cols = []
    for col in DEFAULT_FEATURE_CANDIDATES:
        if col in df.columns and col not in {target_col}:
            cols.append(col)
    if base_pred_col not in cols:
        cols.insert(0, base_pred_col)
    # Keep only numeric columns with at least two finite values.
    usable = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        if np.isfinite(values.to_numpy(dtype=np.float64)).sum() >= 2:
            usable.append(col)
    if base_pred_col not in usable:
        usable.insert(0, base_pred_col)
    return usable


def _build_matrix(df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    medians = {}
    parts = []
    for col in feature_columns:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        arr = values.to_numpy(dtype=np.float64)
        finite = np.isfinite(arr)
        median = float(np.median(arr[finite])) if finite.any() else 0.0
        medians[col] = median
        arr = np.where(finite, arr, median)
        parts.append(arr)
    return np.column_stack(parts).astype(np.float64), medians


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> Tuple[np.ndarray, float]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std <= 1e-12, 1.0, x_std)
    y_mean = float(y.mean())
    x_scaled = (x - x_mean) / x_std
    reg = float(l2) * np.eye(x_scaled.shape[1], dtype=np.float64)
    coef_scaled = np.linalg.solve(x_scaled.T @ x_scaled + reg, x_scaled.T @ (y - y_mean))
    coef = coef_scaled / x_std
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def fit_adapter_from_frame(
    df: pd.DataFrame,
    *,
    target_col: str,
    base_pred_col: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
    l2: float = 1.0,
    holdout_fraction: float = 0.25,
    random_seed: int = 17,
    policy: Optional[SoftWeightPolicy] = None,
) -> LightweightCalibrationAdapter:
    base_col = choose_base_prediction_column(df, base_pred_col)
    if target_col not in df.columns:
        raise ValueError(f"Target RT column not found: {target_col}")
    feature_cols = list(feature_columns) if feature_columns else infer_feature_columns(df, base_col, target_col)
    y_all = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    base_all = pd.to_numeric(df[base_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y_all) & np.isfinite(base_all)
    train_df = df.loc[valid].reset_index(drop=True)
    if len(train_df) < 2:
        raise ValueError("At least two rows with target and base prediction are required.")

    rng = np.random.default_rng(int(random_seed))
    indices = np.arange(len(train_df))
    rng.shuffle(indices)
    holdout_n = int(round(len(train_df) * float(holdout_fraction)))
    holdout_n = min(max(holdout_n, 0), max(len(train_df) - 2, 0))
    holdout_idx = indices[:holdout_n]
    fit_idx = indices[holdout_n:]
    fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True) if holdout_n else pd.DataFrame()

    x_fit, medians = _build_matrix(fit_df, feature_cols)
    y_fit = pd.to_numeric(fit_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    coef, intercept = _fit_ridge(x_fit, y_fit, l2=l2)

    provisional = LightweightCalibrationAdapter(
        feature_columns=feature_cols,
        coefficients=[float(v) for v in coef],
        intercept=float(intercept),
        base_pred_col=base_col,
        target_col=target_col,
        train_rows=int(len(train_df)),
        blend_weight=0.0,
        weight_reason="unassigned",
        feature_medians=medians,
        l2=float(l2),
    )

    validation = {}
    if len(holdout_df):
        y_hold = pd.to_numeric(holdout_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
        base_hold = pd.to_numeric(holdout_df[base_col], errors="coerce").to_numpy(dtype=np.float64)
        adapter_hold = provisional.predict_adapter(holdout_df)
        validation = {
            "holdout_rows": int(len(holdout_df)),
            "base_metrics": _metrics(y_hold, base_hold),
            "adapter_metrics": _metrics(y_hold, adapter_hold),
        }
    else:
        y_fit_eval = pd.to_numeric(fit_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
        base_fit_eval = pd.to_numeric(fit_df[base_col], errors="coerce").to_numpy(dtype=np.float64)
        adapter_fit_eval = provisional.predict_adapter(fit_df)
        validation = {
            "holdout_rows": 0,
            "base_metrics": _metrics(y_fit_eval, base_fit_eval),
            "adapter_metrics": _metrics(y_fit_eval, adapter_fit_eval),
            "note": "training_metrics_only_no_holdout",
        }

    policy_obj = policy or SoftWeightPolicy()
    weight, reason = policy_obj.weight_for(len(train_df), validation=validation)
    # Validation decides whether/how strongly the adapter should be used. Once
    # that decision is fixed, refit on every supplied anchor so deployment does
    # not discard the validation subset.
    if len(holdout_df):
        x_all, medians_all = _build_matrix(train_df, feature_cols)
        y_refit = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
        coef_all, intercept_all = _fit_ridge(x_all, y_refit, l2=l2)
        provisional.coefficients = [float(v) for v in coef_all]
        provisional.intercept = float(intercept_all)
        provisional.feature_medians = medians_all

    provisional.blend_weight = float(weight)
    provisional.weight_reason = reason
    provisional.validation = validation
    provisional.policy = policy_obj.to_dict()
    return provisional


def apply_adapter_to_frame(
    df: pd.DataFrame,
    adapter_path: str,
    *,
    mode: str = "auto",
    update_final: bool = True,
    final_col: Optional[str] = None,
) -> pd.DataFrame:
    payload = joblib.load(adapter_path)
    if str(payload.get("version", "")).startswith("g4_safe_anchor_cv"):
        from core.calibration.g4_safe_adapter import apply_g4_adapter_to_frame

        return apply_g4_adapter_to_frame(
            df,
            adapter_path,
            update_final=update_final,
            final_col=final_col,
        )
    adapter = LightweightCalibrationAdapter.load(adapter_path)
    out = adapter.apply(df, mode=mode)
    if update_final:
        target_col = final_col or adapter.base_pred_col
        if target_col in out.columns:
            out[target_col] = out["pred_rt_with_calibration_adapter"]
    return out


def write_summary(adapter: LightweightCalibrationAdapter, path: str) -> None:
    Path(path).write_text(json.dumps(adapter.to_payload(), indent=2), encoding="utf-8")
