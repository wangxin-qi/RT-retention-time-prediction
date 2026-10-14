from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _fit_affine(prediction: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if len(prediction) < 2 or float(np.std(prediction)) <= 1e-10:
        return 1.0, float(np.mean(observed - prediction))
    slope, intercept = np.polyfit(prediction, observed, deg=1)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return 1.0, float(np.mean(observed - prediction))
    return float(slope), float(intercept)


def _calibrated_matrix(
    train_prediction: np.ndarray,
    train_observed: np.ndarray,
    apply_prediction: np.ndarray,
) -> tuple[np.ndarray, list[float], list[float]]:
    columns = []
    slopes = []
    intercepts = []
    for column in range(train_prediction.shape[1]):
        slope, intercept = _fit_affine(train_prediction[:, column], train_observed)
        columns.append(intercept + slope * apply_prediction[:, column])
        slopes.append(slope)
        intercepts.append(intercept)
    return np.column_stack(columns), slopes, intercepts


@dataclass
class StackedAnchorAdapter:
    prediction_columns: list[str]
    affine_slopes: list[float]
    affine_intercepts: list[float]
    ridge_coefficients: list[float]
    ridge_intercept: float
    ridge_alpha: float
    inner_cv_mae: float
    anchor_rows: int
    version: str = "stacked_anchor_affine_ridge_v1"

    def predict_array(self, prediction: np.ndarray) -> np.ndarray:
        values = np.asarray(prediction, dtype=np.float64)
        calibrated = np.column_stack(
            [
                intercept + slope * values[:, index]
                for index, (slope, intercept) in enumerate(
                    zip(self.affine_slopes, self.affine_intercepts)
                )
            ]
        )
        return calibrated @ np.asarray(self.ridge_coefficients) + float(self.ridge_intercept)

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.prediction_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing stacked prediction columns: {missing}")
        values = frame[self.prediction_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Stacked prediction columns contain missing or non-finite values.")
        return self.predict_array(values)


def fit_stacked_anchor_adapter(
    anchor_frame: pd.DataFrame,
    *,
    target_col: str,
    prediction_columns: Sequence[str],
    alpha_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    inner_folds: int = 5,
    random_seed: int = 17,
) -> StackedAnchorAdapter:
    columns = list(prediction_columns)
    required = [target_col, *columns]
    missing = [column for column in required if column not in anchor_frame.columns]
    if missing:
        raise ValueError(f"Missing anchor columns: {missing}")
    numeric = anchor_frame[required].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[np.isfinite(numeric.to_numpy(float)).all(axis=1)].reset_index(drop=True)
    if len(numeric) < max(6, len(columns) + 2):
        raise ValueError("Too few complete anchors for stacked calibration.")

    observed = numeric[target_col].to_numpy(float)
    prediction = numeric[columns].to_numpy(float)
    fold_count = min(max(2, int(inner_folds)), len(numeric))
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=int(random_seed))

    alpha_scores: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    for train_idx, valid_idx in splitter.split(prediction):
        train_calibrated, _, _ = _calibrated_matrix(
            prediction[train_idx],
            observed[train_idx],
            prediction[train_idx],
        )
        valid_calibrated, _, _ = _calibrated_matrix(
            prediction[train_idx],
            observed[train_idx],
            prediction[valid_idx],
        )
        for alpha in alpha_scores:
            model = Ridge(alpha=alpha).fit(train_calibrated, observed[train_idx])
            fold_pred = model.predict(valid_calibrated)
            alpha_scores[alpha].extend(np.abs(fold_pred - observed[valid_idx]).tolist())

    selected_alpha = min(alpha_scores, key=lambda alpha: float(np.mean(alpha_scores[alpha])))
    inner_cv_mae = float(np.mean(alpha_scores[selected_alpha]))
    calibrated, slopes, intercepts = _calibrated_matrix(prediction, observed, prediction)
    ridge = Ridge(alpha=selected_alpha).fit(calibrated, observed)
    return StackedAnchorAdapter(
        prediction_columns=columns,
        affine_slopes=slopes,
        affine_intercepts=intercepts,
        ridge_coefficients=[float(value) for value in ridge.coef_],
        ridge_intercept=float(ridge.intercept_),
        ridge_alpha=float(selected_alpha),
        inner_cv_mae=inner_cv_mae,
        anchor_rows=int(len(numeric)),
    )
