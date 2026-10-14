from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - import-time fallback for non-RDKit envs
    Chem = None
    AllChem = None


DEFAULT_INTERACTION_META_IDXS = (
    3,   # flowrate
    4,   # t0
    5,   # temperature
    6,   # pH A
    7,   # pH B
    8,   # additive A
    9,   # additive B
    12,  # B acn
    13,  # B meoh
    14,  # nh4form
    15,  # init B
    16,  # final B
    17,  # segments
    18,  # total time
    28,  # max slope
    31,  # phase c18
    32,  # phase hss t3
    33,  # phase hilic
    34,  # phase phenyl
    35,  # solvent acn
    36,  # solvent meoh
    37,  # solvent mixed
    42,  # gradient fast
    43,  # gradient medium
    44,  # gradient slow
)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def normalize_method_code(code: Any) -> str:
    text = "" if code is None else str(code).strip()
    if text.isdigit():
        return text.zfill(4)
    return text


def smiles_to_fp(smiles: Any, n_bits: int = 1024, radius: int = 3) -> Optional[np.ndarray]:
    text = "" if smiles is None else str(smiles).strip()
    if not text or Chem is None or AllChem is None:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    arr = np.zeros((int(n_bits),), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fit_feature_stats(meta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    meta = np.asarray(meta, dtype=np.float32)
    mean = np.nanmean(meta, axis=0).astype(np.float32)
    std = (np.nanstd(meta, axis=0) + 1e-6).astype(np.float32)
    return mean, std


def make_features(
    fps: np.ndarray,
    meta: np.ndarray,
    *,
    meta_mean: np.ndarray,
    meta_std: np.ndarray,
    interaction_meta_idxs: Sequence[int],
) -> sparse.csr_matrix:
    fps = np.asarray(fps, dtype=np.float32)
    meta = np.asarray(meta, dtype=np.float32)
    if fps.ndim == 1:
        fps = fps.reshape(1, -1)
    if meta.ndim == 1:
        meta = meta.reshape(1, -1)
    fp_csr = sparse.csr_matrix(fps)
    meta_norm = np.clip((meta - meta_mean) / meta_std, -6.0, 6.0).astype(np.float32)
    blocks = [fp_csr, sparse.csr_matrix(meta_norm)]
    for idx in interaction_meta_idxs:
        if 0 <= int(idx) < meta_norm.shape[1]:
            blocks.append(fp_csr.multiply(meta_norm[:, int(idx)].reshape(-1, 1)))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def fit_family_residual_model(
    fps: np.ndarray,
    meta: np.ndarray,
    rt: np.ndarray,
    global_pred: np.ndarray,
    *,
    alpha: float = 25.0,
    interaction_meta_idxs: Sequence[int] = DEFAULT_INTERACTION_META_IDXS,
) -> Dict[str, Any]:
    meta_mean, meta_std = fit_feature_stats(meta)
    x_train = make_features(
        fps,
        meta,
        meta_mean=meta_mean,
        meta_std=meta_std,
        interaction_meta_idxs=interaction_meta_idxs,
    )
    delta = (np.asarray(rt, dtype=np.float32) - np.asarray(global_pred, dtype=np.float32)).astype(np.float32)
    delta_mean = float(np.nanmean(delta))
    delta_std = float(np.nanstd(delta) + 1e-6)
    y = ((delta - delta_mean) / delta_std).astype(np.float32)
    model = Ridge(alpha=float(alpha), solver="lsqr", fit_intercept=True)
    model.fit(x_train, y)
    return {
        "model": model,
        "meta_mean": meta_mean,
        "meta_std": meta_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "interaction_meta_idxs": tuple(int(i) for i in interaction_meta_idxs),
    }


def predict_family_delta(fitted: Dict[str, Any], fps: np.ndarray, meta: np.ndarray) -> np.ndarray:
    x = make_features(
        fps,
        meta,
        meta_mean=np.asarray(fitted["meta_mean"], dtype=np.float32),
        meta_std=np.asarray(fitted["meta_std"], dtype=np.float32),
        interaction_meta_idxs=tuple(int(i) for i in fitted["interaction_meta_idxs"]),
    )
    pred_z = fitted["model"].predict(x).astype(np.float32)
    return pred_z * float(fitted["delta_std"]) + float(fitted["delta_mean"])


def method_meta_mean(meta: np.ndarray) -> np.ndarray:
    meta = np.asarray(meta, dtype=np.float32)
    if meta.ndim != 2 or meta.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.nanmean(meta, axis=0).astype(np.float32)


def fit_meta_distance_stats(
    method_meta_means: Sequence[np.ndarray],
    interaction_meta_idxs: Sequence[int] = DEFAULT_INTERACTION_META_IDXS,
) -> Dict[str, Any]:
    if not method_meta_means:
        return {"valid_idxs": [], "mean": [], "std": [], "method_meta_means": []}
    max_dim = max(len(v) for v in method_meta_means)
    padded = []
    for vec in method_meta_means:
        arr = np.asarray(vec, dtype=np.float32)
        if len(arr) < max_dim:
            arr = np.pad(arr, (0, max_dim - len(arr)), constant_values=np.nan)
        padded.append(arr)
    meta_stack = np.vstack(padded).astype(np.float32)
    valid_idxs = [int(idx) for idx in interaction_meta_idxs if 0 <= int(idx) < meta_stack.shape[1]]
    if not valid_idxs:
        valid_idxs = list(range(meta_stack.shape[1]))
    sub = meta_stack[:, valid_idxs]
    mean = np.nanmean(sub, axis=0).astype(np.float32)
    std = (np.nanstd(sub, axis=0) + 1e-6).astype(np.float32)
    return {
        "valid_idxs": valid_idxs,
        "mean": mean,
        "std": std,
        "method_meta_means": meta_stack,
    }


def query_neighbor_distance(meta_vec: np.ndarray, distance_stats: Dict[str, Any]) -> Tuple[float, float]:
    method_meta = np.asarray(distance_stats.get("method_meta_means", []), dtype=np.float32)
    valid_idxs = [int(i) for i in distance_stats.get("valid_idxs", [])]
    if method_meta.size == 0 or not valid_idxs:
        return float("inf"), float("inf")
    meta_vec = np.asarray(meta_vec, dtype=np.float32)
    if len(meta_vec) < method_meta.shape[1]:
        meta_vec = np.pad(meta_vec, (0, method_meta.shape[1] - len(meta_vec)), constant_values=np.nan)
    mean = np.asarray(distance_stats.get("mean"), dtype=np.float32)
    std = np.asarray(distance_stats.get("std"), dtype=np.float32)
    query = np.nan_to_num((meta_vec[valid_idxs] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    train = np.nan_to_num((method_meta[:, valid_idxs] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    query = np.clip(query, -6.0, 6.0)
    train = np.clip(train, -6.0, 6.0)
    denom = max(float(len(valid_idxs)) ** 0.5, 1.0)
    distances = np.linalg.norm(train - query.reshape(1, -1), axis=1) / denom
    return float(np.min(distances)), float(np.median(distances))


def nearest_distance_threshold(distance_stats: Dict[str, Any], *, fallback: float = 2.5) -> float:
    method_meta = np.asarray(distance_stats.get("method_meta_means", []), dtype=np.float32)
    valid_idxs = [int(i) for i in distance_stats.get("valid_idxs", [])]
    if len(method_meta) < 2 or not valid_idxs:
        return float(fallback)
    mean = np.asarray(distance_stats.get("mean"), dtype=np.float32)
    std = np.asarray(distance_stats.get("std"), dtype=np.float32)
    z = np.nan_to_num((method_meta[:, valid_idxs] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, -6.0, 6.0)
    denom = max(float(len(valid_idxs)) ** 0.5, 1.0)
    nearest = []
    for i in range(len(z)):
        dist = np.linalg.norm(z - z[i].reshape(1, -1), axis=1) / denom
        dist[i] = np.inf
        nearest.append(float(np.min(dist)))
    q90 = float(np.quantile(nearest, 0.90)) if nearest else float(fallback)
    return max(float(fallback), q90 + 0.25)


class FamilyConditionExpertPredictor:
    def __init__(self, model_path: str, *, global_bundle: Any = None, dataset_mod: Any = None):
        self.model_path = model_path
        self.payload = joblib.load(model_path)
        self.family_models: Dict[str, Dict[str, Any]] = dict(self.payload.get("family_models", {}))
        self.default_max_neighbor_distance = float(self.payload.get("default_max_neighbor_distance", 2.5))
        self.global_bundle = global_bundle
        if dataset_mod is None:
            from core.data import dataset_v11_pro_meta_threehead_v47_calibrated_gate as dataset_mod  # noqa: WPS433

        self.dataset_mod = dataset_mod

    def _metadata_to_meta(self, metadata_dict: Dict[str, Any]) -> Tuple[str, np.ndarray]:
        family_id = str(self.dataset_mod.auto_family_id_from_metadata(metadata_dict))
        meta_vec = np.asarray(self.dataset_mod.extract_method_meta_vector(metadata_dict), dtype=np.float32)
        return family_id, meta_vec

    def _global_predict(self, fp: np.ndarray, meta_vec: np.ndarray, fallback: Any = None) -> float:
        if self.global_bundle is not None:
            return float(self.global_bundle.predict(fp.astype(np.float32), meta_vec.astype(np.float32)))
        try:
            value = float(fallback)
            return value if np.isfinite(value) else float("nan")
        except Exception:
            return float("nan")

    def predict_frame(
        self,
        df: pd.DataFrame,
        *,
        metadata_dict: Dict[str, Any],
        smiles_col: str = "smiles",
        max_neighbor_distance: Optional[float] = None,
        base_rt_col: Optional[str] = None,
    ) -> pd.DataFrame:
        family_id, meta_vec = self._metadata_to_meta(metadata_dict)
        family = self.family_models.get(family_id)
        threshold_arg = self.default_max_neighbor_distance if max_neighbor_distance is None else float(max_neighbor_distance)

        out = pd.DataFrame(index=df.index)
        out["pred_rt_family_condition"] = np.nan
        out["pred_rt_family_condition_global"] = np.nan
        out["pred_rt_family_condition_delta"] = np.nan
        out["pred_rt_family_condition_error_est"] = np.nan
        out["pred_rt_family_condition_accepted"] = False
        out["pred_rt_family_condition_family_id"] = family_id
        out["pred_rt_family_condition_status"] = "no_family_model"
        out["pred_rt_family_condition_blend_weight"] = 0.0
        out["pred_rt_family_condition_neighbor_distance"] = np.nan
        out["pred_rt_family_condition_neighbor_threshold"] = threshold_arg
        out["pred_rt_family_condition_train_methods"] = 0
        out["pred_rt_family_condition_train_rows"] = 0

        if not family:
            return out

        neighbor_distance, neighbor_median = query_neighbor_distance(meta_vec, family.get("distance_stats", {}))
        threshold = float(family.get("max_neighbor_distance", threshold_arg))
        if max_neighbor_distance is not None:
            threshold = float(max_neighbor_distance)
        accepted_family = bool(family.get("accepted", False))
        distance_ok = bool(np.isfinite(neighbor_distance) and neighbor_distance <= threshold)
        blend_weight = float(family.get("blend_weight", 0.0) or 0.0)
        status = "accepted" if accepted_family and distance_ok and blend_weight > 0 else "guard_rejected"
        if not accepted_family:
            status = "family_internal_rejected"
        elif not distance_ok:
            status = "neighbor_distance_rejected"
        elif blend_weight <= 0:
            status = "zero_blend_weight"

        out["pred_rt_family_condition_status"] = status
        out["pred_rt_family_condition_blend_weight"] = blend_weight
        out["pred_rt_family_condition_neighbor_distance"] = neighbor_distance
        out["pred_rt_family_condition_neighbor_median_distance"] = neighbor_median
        out["pred_rt_family_condition_neighbor_threshold"] = threshold
        out["pred_rt_family_condition_train_methods"] = int(family.get("train_methods", 0) or 0)
        out["pred_rt_family_condition_train_rows"] = int(family.get("train_rows", 0) or 0)
        out["pred_rt_family_condition_error_est"] = safe_float(family.get("internal_direct_mae"), np.nan)

        meta_batch = np.repeat(meta_vec.reshape(1, -1), len(df), axis=0).astype(np.float32)
        fps: List[np.ndarray] = []
        valid_mask: List[bool] = []
        for value in df[smiles_col]:
            fp = smiles_to_fp(value)
            if fp is None:
                fps.append(np.zeros((1024,), dtype=np.float32))
                valid_mask.append(False)
            else:
                fps.append(fp)
                valid_mask.append(True)
        if not fps:
            return out
        fp_batch = np.stack(fps, axis=0).astype(np.float32)
        delta = predict_family_delta(family["fitted"], fp_batch, meta_batch)

        global_pred = np.full(len(df), np.nan, dtype=np.float64)
        for i in range(len(df)):
            fallback = df.iloc[i].get(base_rt_col) if base_rt_col and base_rt_col in df.columns else None
            global_pred[i] = self._global_predict(fp_batch[i], meta_vec, fallback=fallback)
        pred = global_pred + blend_weight * delta.astype(np.float64)
        row_ok = (
            np.asarray(valid_mask, dtype=bool)
            & np.isfinite(global_pred)
            & np.isfinite(pred)
            & bool(accepted_family)
            & bool(distance_ok)
            & (blend_weight > 0)
        )
        out["pred_rt_family_condition"] = pred
        out["pred_rt_family_condition_global"] = global_pred
        out["pred_rt_family_condition_delta"] = delta
        out["pred_rt_family_condition_accepted"] = row_ok
        return out

    def predict_one(
        self,
        *,
        smiles: str,
        metadata_dict: Dict[str, Any],
        max_neighbor_distance: Optional[float] = None,
        base_rt: Optional[float] = None,
    ) -> Dict[str, Any]:
        df = pd.DataFrame([{"smiles": smiles, "base_rt": base_rt}])
        pred = self.predict_frame(
            df,
            metadata_dict=metadata_dict,
            smiles_col="smiles",
            max_neighbor_distance=max_neighbor_distance,
            base_rt_col="base_rt",
        ).iloc[0]
        return pred.to_dict()
