from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from core.cross_condition.condition_features import load_condition_features, metadata_to_condition_features
from core.cross_condition.transfer_model import molecular_descriptors_from_smiles

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


def morgan_bits_from_smiles(smiles: object, n_bits: int = 256, radius: int = 2) -> Dict[str, float]:
    if int(n_bits) <= 0:
        return {}
    values = {f"morgan_{idx}": 0.0 for idx in range(int(n_bits))}
    text = "" if smiles is None else str(smiles).strip()
    if not text:
        return values
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return values
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
        on_bits = list(fp.GetOnBits())
        for bit in on_bits:
            values[f"morgan_{int(bit)}"] = 1.0
    except Exception:
        return values
    return values


def feature_dict_for_smiles(
    smiles: object,
    condition_features: Dict[str, float],
    *,
    n_bits: int = 256,
    radius: int = 2,
    public_like: float = 1.0,
) -> Dict[str, float]:
    row = {}
    row.update(molecular_descriptors_from_smiles(smiles))
    row.update(morgan_bits_from_smiles(smiles, n_bits=n_bits, radius=radius))
    row.update({f"cond_{key}": float(value) for key, value in condition_features.items()})
    row["domain_public_like"] = float(public_like)
    return row


def build_feature_frame(
    df: pd.DataFrame,
    *,
    metadata_path: Optional[str] = None,
    metadata_dict: Optional[Dict[str, object]] = None,
    feature_columns: Optional[Iterable[str]] = None,
    smiles_col: str = "smiles",
    n_bits: int = 256,
    radius: int = 2,
    public_like: float = 1.0,
) -> pd.DataFrame:
    if metadata_dict is not None:
        condition_features = metadata_to_condition_features(metadata_dict)
    elif metadata_path is not None:
        condition_features = load_condition_features(metadata_path)
    else:
        raise ValueError("Either metadata_path or metadata_dict is required.")
    rows = [
        feature_dict_for_smiles(
            value,
            condition_features,
            n_bits=n_bits,
            radius=radius,
            public_like=public_like,
        )
        for value in df[smiles_col]
    ]
    out = pd.DataFrame(rows, index=df.index)
    if feature_columns is not None:
        for col in feature_columns:
            if col not in out.columns:
                out[col] = 0.0
        out = out[list(feature_columns)]
    return out.fillna(0.0).astype(float)


class PublicAwareGlobalPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.payload = joblib.load(model_path)
        self.feature_columns = list(self.payload["feature_columns"])
        self.n_bits = int(self.payload.get("n_bits", 256))
        self.radius = int(self.payload.get("radius", 2))
        self.model = self.payload["model"]
        self.error_model = self.payload.get("error_model")
        self.error_feature_columns = list(self.payload.get("error_feature_columns", []))
        self.default_max_predicted_error = float(self.payload.get("default_max_predicted_error", 2.0))

    def predict_frame(
        self,
        df: pd.DataFrame,
        *,
        metadata_path: Optional[str] = None,
        metadata_dict: Optional[Dict[str, object]] = None,
        smiles_col: str = "smiles",
        max_predicted_error: Optional[float] = None,
        public_like: float = 1.0,
    ) -> pd.DataFrame:
        x_df = build_feature_frame(
            df,
            metadata_path=metadata_path,
            metadata_dict=metadata_dict,
            feature_columns=self.feature_columns,
            smiles_col=smiles_col,
            n_bits=self.n_bits,
            radius=self.radius,
            public_like=public_like,
        )
        pred = self.model.predict(x_df.to_numpy())
        err_est = np.full(len(df), np.nan, dtype=np.float64)
        if self.error_model is not None and self.error_feature_columns:
            err_df = x_df.copy()
            err_df["pred_public_global"] = pred
            for col in self.error_feature_columns:
                if col not in err_df.columns:
                    err_df[col] = 0.0
            err_est = np.maximum(np.expm1(self.error_model.predict(err_df[self.error_feature_columns].to_numpy())), 0.0)
        threshold = self.default_max_predicted_error if max_predicted_error is None else float(max_predicted_error)
        accepted = np.isfinite(pred) & np.isfinite(err_est) & (err_est <= threshold)
        out = pd.DataFrame(index=df.index)
        out["pred_rt_public_global"] = pred
        out["pred_rt_public_global_error_est"] = err_est
        out["pred_rt_public_global_accepted"] = accepted
        out["pred_rt_public_global_threshold"] = threshold
        return out
