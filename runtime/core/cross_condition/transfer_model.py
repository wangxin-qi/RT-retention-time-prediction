import os
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from core.cross_condition.condition_features import condition_delta_features, load_condition_features


MOLECULAR_DESCRIPTOR_NAMES = [
    "mol_valid",
    "mol_mw",
    "mol_logp",
    "mol_tpsa",
    "mol_hbd",
    "mol_hba",
    "mol_rotatable",
    "mol_ring_count",
    "mol_aromatic_ring_count",
    "mol_heavy_atoms",
    "mol_fraction_csp3",
    "mol_formal_charge",
]


def molecular_descriptors_from_smiles(smiles: object) -> Dict[str, float]:
    values = {name: 0.0 for name in MOLECULAR_DESCRIPTOR_NAMES}
    text = "" if smiles is None else str(smiles).strip()
    if not text:
        return values
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return values
        values.update(
            {
                "mol_valid": 1.0,
                "mol_mw": float(Descriptors.MolWt(mol)),
                "mol_logp": float(Crippen.MolLogP(mol)),
                "mol_tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "mol_hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
                "mol_hba": float(rdMolDescriptors.CalcNumHBA(mol)),
                "mol_rotatable": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
                "mol_ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
                "mol_aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                "mol_heavy_atoms": float(mol.GetNumHeavyAtoms()),
                "mol_fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                "mol_formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
            }
        )
    except Exception:
        return values
    return values


def _predict_model(model_payload: Dict[str, object], x: np.ndarray) -> np.ndarray:
    model = model_payload["model"]
    model_type = model.get("model_type")
    if model_type == "ridge_condition_transfer":
        mean = np.asarray(model["feature_mean"], dtype=np.float64)
        std = np.asarray(model["feature_std"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        xz = (x - mean) / (std + 1e-8)
        design = np.concatenate([np.ones((len(xz), 1)), xz], axis=1)
        return design @ weights
    if model_type == "hist_gradient_boosting":
        return model["sklearn_model"].predict(x)
    raise ValueError(f"Unsupported model type: {model_type}")


def _predict_ridge_model(model: Dict[str, object], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    xz = (x - mean) / (std + 1e-8)
    design = np.concatenate([np.ones((len(xz), 1)), xz], axis=1)
    return design @ weights


def build_feature_frame(
    input_df: pd.DataFrame,
    *,
    source_rt_col: str,
    source_metadata: Any,
    target_metadata: Any,
    feature_columns: Iterable[str],
    smiles_col: Optional[str] = "smiles",
) -> pd.DataFrame:
    source_features = load_condition_features(source_metadata, prefix="source_")
    target_features = load_condition_features(target_metadata, prefix="target_")
    source_plain = {key.replace("source_", ""): value for key, value in source_features.items()}
    target_plain = {key.replace("target_", ""): value for key, value in target_features.items()}
    delta_features = condition_delta_features(source_plain, target_plain)

    out = pd.DataFrame(index=input_df.index)
    out["source_rt_minutes"] = pd.to_numeric(input_df[source_rt_col], errors="coerce")
    for key, value in {**source_features, **target_features, **delta_features}.items():
        out[key] = value
    if smiles_col and smiles_col in input_df.columns:
        desc = pd.DataFrame([molecular_descriptors_from_smiles(v) for v in input_df[smiles_col]], index=input_df.index)
        out = pd.concat([out, desc], axis=1)
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    return out[list(feature_columns)].fillna(0.0).astype(float)


def build_error_feature_frame(
    x_df: pd.DataFrame,
    *,
    pred_best: np.ndarray,
    pred_ridge: np.ndarray,
    pred_hgb: np.ndarray,
    error_feature_columns: Iterable[str],
) -> pd.DataFrame:
    out = x_df.copy()
    out["pred_best"] = pred_best
    out["pred_ridge"] = pred_ridge
    out["pred_hgb"] = pred_hgb
    out["abs_ridge_hgb_diff"] = np.abs(pred_ridge - pred_hgb)
    out["abs_best_source_delta"] = np.abs(pred_best - out.get("source_rt_minutes", 0.0))
    for col in error_feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    return out[list(error_feature_columns)].fillna(0.0).astype(float)


class CrossConditionTransferPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.payload = joblib.load(model_path)
        self.feature_columns = list(self.payload["feature_columns"])
        self.error_model = self.payload.get("error_model")
        self.error_feature_columns = list(self.payload.get("error_feature_columns", []))
        self.default_max_predicted_error = float(self.payload.get("default_max_predicted_error", 2.0))
        self.model_format = "legacy"
        self.delta_model = None
        self.target_model = None
        self.delta_weight = float(self.payload.get("delta_weight", 0.0))
        if "delta_model_path" in self.payload and "target_model_path" in self.payload:
            import xgboost as xgb

            self.model_format = "scaffold_xgb_blend_v2"
            delta_path = self._resolve_model_path(str(self.payload["delta_model_path"]))
            target_path = self._resolve_model_path(str(self.payload["target_model_path"]))
            self.delta_model = xgb.XGBRegressor()
            self.delta_model.load_model(delta_path)
            self.target_model = xgb.XGBRegressor()
            self.target_model.load_model(target_path)

    def _resolve_model_path(self, configured_path: str) -> str:
        if configured_path and os.path.exists(configured_path):
            return configured_path
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(self.model_path)),
            os.path.basename(configured_path),
        )
        if os.path.exists(sibling):
            return sibling
        raise FileNotFoundError(
            f"Cross-condition XGBoost model not found: {configured_path}; "
            f"also checked bundled sibling: {sibling}"
        )

    def predict_frame(
        self,
        input_df: pd.DataFrame,
        *,
        source_rt_col: str,
        source_metadata: Any,
        target_metadata: Any,
        smiles_col: Optional[str] = "smiles",
        max_predicted_error: Optional[float] = None,
    ) -> pd.DataFrame:
        x_df = build_feature_frame(
            input_df,
            source_rt_col=source_rt_col,
            source_metadata=source_metadata,
            target_metadata=target_metadata,
            feature_columns=self.feature_columns,
            smiles_col=smiles_col,
        )
        x = x_df.to_numpy()
        if self.model_format == "scaffold_xgb_blend_v2":
            source_rt = pd.to_numeric(
                input_df[source_rt_col], errors="coerce"
            ).to_numpy(dtype=np.float64)
            pred_delta = source_rt + self.delta_model.predict(x)
            pred_target = self.target_model.predict(x)
            pred_best = (
                self.delta_weight * pred_delta
                + (1.0 - self.delta_weight) * pred_target
            )
            # The two independently trained heads provide an inference-only
            # disagreement guard.  It is intentionally reported as a safety
            # proxy, not as a calibrated estimate of the true error.
            pred_error = np.abs(pred_delta - pred_target)
            pred_ridge = pred_delta
            pred_hgb = pred_target
        else:
            pred_best = _predict_model(self.payload, x)
            ridge_model = self.payload.get("ridge_model")
            hgb_model = self.payload.get("hgb_model")
            pred_ridge = _predict_ridge_model(ridge_model, x) if ridge_model else pred_best
            pred_hgb = hgb_model.predict(x) if hgb_model is not None else pred_best

            pred_error = np.full(len(input_df), np.nan, dtype=np.float64)
            if self.error_model is not None and self.error_feature_columns:
                err_x = build_error_feature_frame(
                    x_df,
                    pred_best=pred_best,
                    pred_ridge=pred_ridge,
                    pred_hgb=pred_hgb,
                    error_feature_columns=self.error_feature_columns,
                ).to_numpy()
                pred_error = np.expm1(self.error_model.predict(err_x))
                pred_error = np.maximum(pred_error, 0.0)

        threshold = self.default_max_predicted_error if max_predicted_error is None else float(max_predicted_error)
        accepted = (
            np.isfinite(pred_best)
            & (pred_best >= 0.0)
            & np.isfinite(pred_error)
            & (pred_error <= threshold)
        )
        out = pd.DataFrame(index=input_df.index)
        out["pred_rt_cross_condition"] = pred_best
        out["pred_rt_cross_condition_error_est"] = pred_error
        out["pred_rt_cross_condition_disagreement"] = np.abs(pred_ridge - pred_hgb)
        out["pred_rt_cross_condition_accepted"] = accepted
        out["pred_rt_cross_condition_threshold"] = threshold
        out["pred_rt_cross_condition_model"] = (
            "scaffold_xgb_delta_target_blend_v2"
            if self.model_format == "scaffold_xgb_blend_v2"
            else str(self.payload.get("best_model_name", ""))
        )
        return out

    def predict_one(
        self,
        *,
        smiles: str,
        source_rt: float,
        source_metadata: Any,
        target_metadata: Any,
        max_predicted_error: Optional[float] = None,
    ) -> Dict[str, object]:
        df = pd.DataFrame([{"smiles": smiles, "source_rt_minutes": source_rt}])
        pred = self.predict_frame(
            df,
            source_rt_col="source_rt_minutes",
            source_metadata=source_metadata,
            target_metadata=target_metadata,
            smiles_col="smiles",
            max_predicted_error=max_predicted_error,
        ).iloc[0]
        return pred.to_dict()
