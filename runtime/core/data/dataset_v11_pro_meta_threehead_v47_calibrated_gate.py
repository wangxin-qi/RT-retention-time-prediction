import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

BASE_META_DIM = 18
ROUTE_SAMPLE_FRACS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def load_yaml_or_json(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8-sig") as handle:
        if ext in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(handle) or {}
        return json.load(handle)


def resolve_metadata_path(path_value: Optional[str], method_code: Optional[str], search_root: str) -> str:
    if path_value and os.path.exists(path_value):
        return os.path.abspath(path_value)
    if method_code:
        candidate = os.path.join(search_root, str(method_code), f"{method_code}_metadata.yaml")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Metadata path not found: path_value={path_value}, method_code={method_code}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default
        return float(text)
    except Exception:
        return default


def _pick_additive(solvent: Dict[str, Any]) -> float:
    vals = []
    for key in ["formic", "acetic", "value"]:
        value = solvent.get(key)
        if isinstance(value, dict):
            vals.append(_safe_float(value.get("value"), 0.0))
        elif value is not None:
            vals.append(_safe_float(value, 0.0))
    return float(max(vals) if vals else 0.0)


def _pick_nh4form(solvent: Dict[str, Any]) -> float:
    value = solvent.get("nh4form")
    if isinstance(value, dict):
        return _safe_float(value.get("value"), 0.0)
    return _safe_float(value, 0.0)


def _norm_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    return " ".join(text.split())


def _extract_column_name(meta_dict: Dict[str, Any], column: Dict[str, Any]) -> str:
    return _norm_text(
        meta_dict.get("source_row_column_name")
        or meta_dict.get("column_name")
        or column.get("name")
        or column.get("type")
    )


def _extract_fixed_phase_name(meta_dict: Dict[str, Any], column: Dict[str, Any]) -> str:
    return _norm_text(
        column.get("fixed_phase")
        or column.get("stationary_phase")
        or column.get("phase")
        or column.get("chemistry")
        or meta_dict.get("fixed_phase")
    )


def _phase_flags(column_name: str, fixed_phase_name: str) -> Dict[str, float]:
    text = f"{column_name} {fixed_phase_name}".strip()
    flags = {
        "c18": 1.0 if ("c18" in text or "beh" in text or "xbridge" in text) else 0.0,
        "hss_t3": 1.0 if ("hss t3" in text or "t3" in text) else 0.0,
        "hilic": 1.0 if ("hilic" in text or "amide" in text or "zic" in text) else 0.0,
        "phenyl_hexyl": 1.0 if ("phenyl" in text or "hexyl" in text or "biphenyl" in text) else 0.0,
        "unknown": 0.0,
    }
    if sum(flags.values()) <= 0:
        flags["unknown"] = 1.0
    return flags


def _dominant_solvent_flags(solvent: Dict[str, Any]) -> Dict[str, float]:
    water = _safe_float(solvent.get("h2o"), 0.0)
    acn = _safe_float(solvent.get("acn", solvent.get("acetonitrile")), 0.0)
    meoh = _safe_float(solvent.get("meoh"), 0.0)
    dominant = "mixed"
    if acn > meoh and acn > 0:
        dominant = "acn"
    elif meoh > acn and meoh > 0:
        dominant = "meoh"
    elif water > 0 and acn <= 0 and meoh <= 0:
        dominant = "water"
    return {
        "water": 1.0 if dominant == "water" else 0.0,
        "acn": 1.0 if dominant == "acn" else 0.0,
        "meoh": 1.0 if dominant == "meoh" else 0.0,
        "mixed": 1.0 if dominant == "mixed" else 0.0,
    }


def _additive_flags(solvent_a: Dict[str, Any], solvent_b: Dict[str, Any]) -> Dict[str, float]:
    text = _norm_text(json.dumps({"A": solvent_a, "B": solvent_b}, ensure_ascii=False))
    return {
        "formic": 1.0 if "formic" in text else 0.0,
        "acetic": 1.0 if "acetic" in text else 0.0,
        "nh4form": 1.0 if max(_pick_nh4form(solvent_a), _pick_nh4form(solvent_b)) > 0 else 0.0,
        "other_additive": 1.0 if (_pick_additive(solvent_a) > 0 or _pick_additive(solvent_b) > 0) else 0.0,
    }


def _gradient_segments_from_table(method_dir: str, method_code: str) -> float:
    path = os.path.join(method_dir, f"{method_code}_gradient.tsv")
    if not os.path.exists(path):
        return 0.0
    try:
        df = pd.read_csv(path, sep="\t")
        return float(max(0, len(df) - 1))
    except Exception:
        return 0.0


def _canonical_key(key: Any) -> str:
    text = "" if key is None else str(key).strip().lower()
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
    return "".join(cleaned)


_TIME_KEY_ALIASES = {"time", "t", "minute", "min", "tmin"}
_B_KEY_ALIASES = {"b", "bphase", "b_phase", "b%", "bpercent", "phaseb", "solventb"}


def _pick_by_canonical(row: Dict[str, Any], aliases: set) -> Any:
    for key, value in row.items():
        if _canonical_key(key) in aliases:
            return value
    return None


def _normalize_gradient_table_rows(rows: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    normalized: List[Tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        time_value = _safe_float(_pick_by_canonical(row, _TIME_KEY_ALIASES), default=np.nan)
        b_value = _safe_float(_pick_by_canonical(row, _B_KEY_ALIASES), default=np.nan)
        if np.isnan(time_value) or np.isnan(b_value):
            continue
        normalized.append((float(time_value), float(b_value)))
    normalized.sort(key=lambda item: item[0])
    return normalized


def _load_gradient_table(method_dir: Optional[str], method_code: Optional[str], gradient_meta: Dict[str, Any]) -> List[Tuple[float, float]]:
    table_rows = gradient_meta.get("table")
    if isinstance(table_rows, list):
        normalized = _normalize_gradient_table_rows(table_rows)
        if normalized:
            return normalized

    if method_dir and method_code:
        path = os.path.join(method_dir, f"{method_code}_gradient.tsv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep="\t")
                rows = [dict(row) for _, row in df.iterrows()]
                normalized = _normalize_gradient_table_rows(rows)
                if normalized:
                    return normalized
            except Exception:
                pass

    total_time = _safe_float(
        gradient_meta.get("total_time", gradient_meta.get("gradient_total_time", gradient_meta.get("time"))),
        0.0,
    )
    if total_time > 0:
        init_b = _safe_float(gradient_meta.get("init_B", gradient_meta.get("start_B")), 0.0)
        final_b = _safe_float(gradient_meta.get("final_B", gradient_meta.get("end_B")), init_b)
        return [(0.0, init_b), (float(total_time), final_b)]
    return []


def _interp_b_at_time(rows: Sequence[Tuple[float, float]], t_query: float) -> float:
    if not rows:
        return 0.0
    if t_query <= rows[0][0]:
        return float(rows[0][1])
    if t_query >= rows[-1][0]:
        return float(rows[-1][1])
    for idx in range(1, len(rows)):
        t0, b0 = rows[idx - 1]
        t1, b1 = rows[idx]
        if t0 <= t_query <= t1:
            if abs(t1 - t0) <= 1e-8:
                return float(b1)
            frac = (t_query - t0) / (t1 - t0)
            return float(b0 + frac * (b1 - b0))
    return float(rows[-1][1])


def _gradient_feature_summary(rows: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    if not rows:
        return {
            "sampled_b": [0.0] * len(ROUTE_SAMPLE_FRACS),
            "total_time": 0.0,
            "max_b": 0.0,
            "min_b": 0.0,
            "max_slope": 0.0,
            "has_reequilibration": 0.0,
            "plateau_count": 0.0,
            "style_fast": 0.0,
            "style_medium": 1.0,
            "style_slow": 0.0,
        }

    total_time = float(rows[-1][0])
    if total_time > 1e-8:
        sample_times = [float(total_time * frac) for frac in ROUTE_SAMPLE_FRACS]
    else:
        sample_times = [0.0] * len(ROUTE_SAMPLE_FRACS)
    sampled_b = [_interp_b_at_time(rows, t) for t in sample_times]

    max_b = float(max((b for _, b in rows), default=0.0))
    min_b = float(min((b for _, b in rows), default=0.0))
    max_slope = 0.0
    has_reequilibration = 0.0
    plateau_count = 0.0
    for idx in range(1, len(rows)):
        t0, b0 = rows[idx - 1]
        t1, b1 = rows[idx]
        dt = max(t1 - t0, 1e-8)
        slope = (b1 - b0) / dt
        max_slope = max(max_slope, abs(float(slope)))
        if abs(b1 - b0) < 1e-6:
            plateau_count += 1.0
        if b1 + 1e-6 < b0:
            has_reequilibration = 1.0

    if total_time <= 8.0:
        style_fast, style_medium, style_slow = 1.0, 0.0, 0.0
    elif total_time >= 20.0:
        style_fast, style_medium, style_slow = 0.0, 0.0, 1.0
    else:
        style_fast, style_medium, style_slow = 0.0, 1.0, 0.0

    return {
        "sampled_b": sampled_b,
        "total_time": total_time,
        "max_b": max_b,
        "min_b": min_b,
        "max_slope": max_slope,
        "has_reequilibration": has_reequilibration,
        "plateau_count": plateau_count,
        "style_fast": style_fast,
        "style_medium": style_medium,
        "style_slow": style_slow,
    }


def extract_route_signature(meta_dict: Dict[str, Any], method_dir: Optional[str] = None, method_code: Optional[str] = None) -> np.ndarray:
    base_meta = extract_method_meta_vector(meta_dict, method_dir=method_dir, method_code=method_code)
    gradient = meta_dict.get("gradient", {}) or {}
    rows = _load_gradient_table(method_dir, method_code, gradient)
    grad = _gradient_feature_summary(rows)
    extras = np.asarray(
        grad["sampled_b"]
        + [
            grad["total_time"],
            grad["max_b"],
            grad["min_b"],
            grad["max_slope"],
            grad["has_reequilibration"],
            grad["plateau_count"],
            grad["style_fast"],
            grad["style_medium"],
            grad["style_slow"],
        ],
        dtype=np.float32,
    )
    return np.concatenate([base_meta.astype(np.float32), extras], axis=0).astype(np.float32)


def extract_method_meta_vector(meta_dict: Dict[str, Any], method_dir: Optional[str] = None, method_code: Optional[str] = None) -> np.ndarray:
    column = meta_dict.get("column", {}) or {}
    eluent = meta_dict.get("eluent", {}) or {}
    solvent_a = eluent.get("A", {}) or {}
    solvent_b = eluent.get("B", {}) or {}
    gradient = meta_dict.get("gradient", {}) or {}

    segments = _safe_float(gradient.get("segments"), 0.0)
    if segments <= 0.0 and method_dir and method_code:
        segments = _gradient_segments_from_table(method_dir, method_code)

    base_meta = np.asarray(
        [
            _safe_float(column.get("length"), 0.0),
            _safe_float(column.get("id"), 0.0),
            _safe_float(column.get("particle.size"), 0.0),
            _safe_float(column.get("flowrate"), 0.0),
            _safe_float(column.get("t0"), 0.0),
            _safe_float(column.get("temperature"), 0.0),
            _safe_float(solvent_a.get("pH"), 0.0),
            _safe_float(solvent_b.get("pH"), 0.0),
            _pick_additive(solvent_a),
            _pick_additive(solvent_b),
            1.0 if _safe_float(solvent_a.get("h2o"), 0.0) > 0 else 0.0,
            1.0 if _safe_float(solvent_a.get("meoh"), 0.0) > 0 else 0.0,
            1.0 if _safe_float(solvent_b.get("acn", solvent_b.get("acetonitrile")), 0.0) > 0 else 0.0,
            1.0 if _safe_float(solvent_b.get("meoh"), 0.0) > 0 else 0.0,
            1.0 if max(_pick_nh4form(solvent_a), _pick_nh4form(solvent_b)) > 0 else 0.0,
            _safe_float(gradient.get("init_B", gradient.get("start_B")), 0.0),
            _safe_float(gradient.get("final_B", gradient.get("end_B")), 0.0),
            segments,
        ],
        dtype=np.float32,
    )
    rows = _load_gradient_table(method_dir, method_code, gradient)
    grad = _gradient_feature_summary(rows)
    column_name = _extract_column_name(meta_dict, column)
    fixed_phase_name = _extract_fixed_phase_name(meta_dict, column)
    phase = _phase_flags(column_name, fixed_phase_name)
    solvent_b_flags = _dominant_solvent_flags(solvent_b)
    additive = _additive_flags(solvent_a, solvent_b)
    sampled_b = list(grad["sampled_b"])
    if len(sampled_b) < len(ROUTE_SAMPLE_FRACS):
        sampled_b = sampled_b + [0.0] * (len(ROUTE_SAMPLE_FRACS) - len(sampled_b))
    sampled_b_full = sampled_b[: len(ROUTE_SAMPLE_FRACS)]
    extra_meta = np.asarray(
        [
            grad["total_time"],
            *sampled_b_full,
            grad["max_slope"],
            grad["has_reequilibration"],
            grad["plateau_count"],
            phase["c18"],
            phase["hss_t3"],
            phase["hilic"],
            phase["phenyl_hexyl"],
            solvent_b_flags["acn"],
            solvent_b_flags["meoh"],
            solvent_b_flags["mixed"],
            additive["formic"],
            additive["acetic"],
            additive["nh4form"],
            additive["other_additive"],
            grad["style_fast"],
            grad["style_medium"],
            grad["style_slow"],
        ],
        dtype=np.float32,
    )
    return np.concatenate([base_meta, extra_meta], axis=0).astype(np.float32)


def auto_family_id_from_meta(meta_vec: Sequence[float]) -> str:
    vec = np.asarray(meta_vec, dtype=np.float32)
    if len(vec) >= BASE_META_DIM + 27:
        phase_offset = BASE_META_DIM + 13
        c18 = vec[phase_offset + 0] > 0.5
        hss_t3 = vec[phase_offset + 1] > 0.5
        hilic = vec[phase_offset + 2] > 0.5
        phenyl = vec[phase_offset + 3] > 0.5
        solvent_offset = phase_offset + 4
        b_acn = vec[solvent_offset + 0] > 0.5
        b_meoh = vec[solvent_offset + 1] > 0.5
        mixed = vec[solvent_offset + 2] > 0.5
        style_offset = solvent_offset + 7
        fast = vec[style_offset + 0] > 0.5
        medium = vec[style_offset + 1] > 0.5
        slow = vec[style_offset + 2] > 0.5
    elif len(vec) >= BASE_META_DIM + 23:
        phase_offset = BASE_META_DIM + 9
        c18 = vec[phase_offset + 0] > 0.5
        hss_t3 = vec[phase_offset + 1] > 0.5
        hilic = vec[phase_offset + 2] > 0.5
        phenyl = vec[phase_offset + 3] > 0.5
        solvent_offset = phase_offset + 4
        b_acn = vec[solvent_offset + 0] > 0.5
        b_meoh = vec[solvent_offset + 1] > 0.5
        mixed = vec[solvent_offset + 2] > 0.5
        style_offset = solvent_offset + 7
        fast = vec[style_offset + 0] > 0.5
        medium = vec[style_offset + 1] > 0.5
        slow = vec[style_offset + 2] > 0.5
    else:
        c18 = True
        hss_t3 = False
        hilic = False
        phenyl = False
        b_acn = vec[12] > 0.5 if len(vec) > 12 else False
        b_meoh = vec[13] > 0.5 if len(vec) > 13 else False
        mixed = not (b_acn or b_meoh)
        total_time = 0.0
        if len(vec) > 17:
            total_time = abs(float(vec[17] - vec[16])) * max(float(vec[17]), 1.0)
        fast = total_time <= 8.0
        slow = total_time >= 20.0
        medium = not (fast or slow)

    phase_name = "unknown"
    if c18:
        phase_name = "c18"
    if hss_t3:
        phase_name = "hss_t3"
    elif hilic:
        phase_name = "hilic"
    elif phenyl:
        phase_name = "phenyl"

    solvent_name = "mixed"
    if b_acn:
        solvent_name = "acn"
    elif b_meoh:
        solvent_name = "meoh"
    elif mixed:
        solvent_name = "mixed"

    gradient_name = "medium"
    if fast:
        gradient_name = "fast"
    elif slow:
        gradient_name = "slow"

    return f"fam_{phase_name}_{solvent_name}_{gradient_name}"


def auto_family_id_from_metadata(meta_dict: Dict[str, Any], method_dir: Optional[str] = None, method_code: Optional[str] = None) -> str:
    return auto_family_id_from_meta(extract_method_meta_vector(meta_dict, method_dir=method_dir, method_code=method_code))


def smiles_to_fp(smiles: str, n_bits: int = 1024, radius: int = 3) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _bits_on_to_vec(bits_on: str, n_bits: int = 1024) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    for token in str(bits_on).split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if 0 <= idx < n_bits:
            arr[idx] = 1.0
    return arr


@dataclass
class RTDataset:
    fps: np.ndarray
    meta: np.ndarray
    rts: np.ndarray
    smiles: np.ndarray
    ids: np.ndarray
    rt_mean: float
    rt_std: float
    rt_z: np.ndarray

    def __len__(self) -> int:
        return int(len(self.rts))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "fp": self.fps[index],
            "meta": self.meta[index],
            "rt": self.rts[index],
            "rt_z": self.rt_z[index],
            "smiles": self.smiles[index],
            "id": self.ids[index],
        }


@dataclass
class DeltaDataset:
    fps: np.ndarray
    meta: np.ndarray
    rts: np.ndarray
    smiles: np.ndarray
    ids: np.ndarray
    base_pred: np.ndarray
    delta: np.ndarray
    delta_mean: float
    delta_std: float
    delta_z: np.ndarray

    def __len__(self) -> int:
        return int(len(self.rts))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "fp": self.fps[index],
            "meta": self.meta[index],
            "rt": self.rts[index],
            "base_pred": self.base_pred[index],
            "delta": self.delta[index],
            "delta_z": self.delta_z[index],
            "smiles": self.smiles[index],
            "id": self.ids[index],
        }


def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    out: Dict[str, List[Any]] = {}
    for row in batch:
        for key, value in row.items():
            out.setdefault(key, []).append(value)
    collated: Dict[str, np.ndarray] = {}
    for key, values in out.items():
        first = values[0]
        if isinstance(first, np.ndarray):
            collated[key] = np.stack(values, axis=0)
        elif isinstance(first, (float, int, np.floating, np.integer)):
            collated[key] = np.asarray(values, dtype=np.float32)
        else:
            collated[key] = np.asarray(values)
    return collated


def split_indices_by_group(groups: Sequence[str], val_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique = unique.copy()
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * float(val_ratio)))) if len(unique) > 1 else 0
    val_groups = set(unique[:n_val].tolist())
    train_idx = np.asarray([i for i, g in enumerate(groups.tolist()) if g not in val_groups], dtype=np.int64)
    val_idx = np.asarray([i for i, g in enumerate(groups.tolist()) if g in val_groups], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        perm = np.arange(len(groups), dtype=np.int64)
        rng.shuffle(perm)
        n_val = max(1, int(round(len(perm) * float(val_ratio)))) if len(perm) > 1 else 0
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def load_method_family_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    data = load_yaml_or_json(path)
    return {str(k): str(v) for k, v in data.items()}


def list_method_dirs(search_root: str) -> List[str]:
    if not os.path.isdir(search_root):
        return []
    return [
        os.path.join(search_root, name)
        for name in sorted(os.listdir(search_root))
        if os.path.isdir(os.path.join(search_root, name)) and name.isdigit()
    ]


def _read_tsv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(path, sep="\t", encoding=enc, encoding_errors="ignore", engine="python")
        except Exception:
            continue
    raise RuntimeError(f"Failed to read TSV: {path}")


def _valid_smiles(smiles: Any) -> bool:
    try:
        return Chem.MolFromSmiles(str(smiles)) is not None
    except Exception:
        return False


def _collapse_duplicate_rt_records(df: pd.DataFrame) -> pd.DataFrame:
    """QC RT records within one method before building tensors.

    The old loader kept the first row for a duplicated ``id``.  For RT modelling the
    meaningful replicate key is the standardized structure under the same method, so
    duplicate standardized SMILES are collapsed to a median RT and annotated with the
    replicate count/spread.  This avoids order-dependent training targets while keeping
    the existing tensor schema unchanged.
    """
    if df.empty:
        return df
    df = df.copy()
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    finite_rt = np.isfinite(df["rt"].to_numpy(dtype=np.float64, copy=False))
    df = df.loc[finite_rt & (df["rt"] >= 0)].copy()
    if df.empty:
        return df
    df = df[df["smiles.std"].map(_valid_smiles)].copy()
    if df.empty:
        return df

    rows = []
    for _, group in df.groupby("smiles.std", sort=False):
        base = group.iloc[0].copy()
        rt_values = group["rt"].to_numpy(dtype=np.float64)
        base["rt"] = float(np.median(rt_values))
        base["rt_replicate_count"] = int(len(group))
        base["rt_replicate_spread"] = float(np.max(rt_values) - np.min(rt_values)) if len(rt_values) else 0.0
        rows.append(base)
    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_rt_and_fp(method_dir: str, method_code: str) -> pd.DataFrame:
    pieces = []
    for suffix in ["canonical_success", "isomeric_success"]:
        rt_path = os.path.join(method_dir, f"{method_code}_rtdata_{suffix}.tsv")
        fp_path = os.path.join(method_dir, f"{method_code}_fingerprints_ecfp6_{suffix}.tsv")
        if not (os.path.exists(rt_path) and os.path.exists(fp_path)):
            continue
        rt_df = _read_tsv(rt_path)
        fp_df = _read_tsv(fp_path)
        if "id" not in rt_df.columns or "id" not in fp_df.columns or "rt" not in rt_df.columns or "smiles.std" not in rt_df.columns:
            continue
        merged = rt_df.merge(fp_df[["id", "bits.on"]], on="id", how="inner")
        pieces.append(merged)
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True)
    df = df.dropna(subset=["id", "rt", "smiles.std", "bits.on"]).copy()
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    return _collapse_duplicate_rt_records(df)


def _dataset_from_frame(df: pd.DataFrame, meta_vec: np.ndarray) -> RTDataset:
    fps = np.stack([_bits_on_to_vec(x) for x in df["bits.on"].tolist()], axis=0).astype(np.float32)
    metas = np.repeat(meta_vec[None, :], len(df), axis=0).astype(np.float32)
    rts = df["rt"].to_numpy(dtype=np.float32)
    rt_mean = float(np.mean(rts))
    rt_std = float(np.std(rts) + 1e-8)
    rt_z = ((rts - rt_mean) / rt_std).astype(np.float32)
    smiles = df["smiles.std"].astype(str).to_numpy()
    ids = df["id"].astype(str).to_numpy()
    return RTDataset(
        fps=fps,
        meta=metas,
        rts=rts,
        smiles=smiles,
        ids=ids,
        rt_mean=rt_mean,
        rt_std=rt_std,
        rt_z=rt_z,
    )


def load_method_dataset(method_dir: str, min_rows: int = 300, family_map: Optional[Dict[str, str]] = None):
    family_map = family_map or {}
    method_code = os.path.basename(os.path.abspath(method_dir))
    metadata_path = os.path.join(method_dir, f"{method_code}_metadata.yaml")
    if not os.path.exists(metadata_path):
        return None, {"method_code": method_code, "reason": "missing_metadata"}

    try:
        df = _load_rt_and_fp(method_dir, method_code)
    except Exception as exc:
        return None, {"method_code": method_code, "reason": f"load_failed:{exc}"}
    if len(df) < int(min_rows):
        return None, {"method_code": method_code, "reason": "too_few_rows", "rows": int(len(df))}

    meta_dict = load_yaml_or_json(metadata_path)
    meta_vec = extract_method_meta_vector(meta_dict, method_dir=method_dir, method_code=method_code)
    ds = _dataset_from_frame(df, meta_vec)
    family_id = family_map.get(method_code, auto_family_id_from_metadata(meta_dict, method_dir=method_dir, method_code=method_code))
    info = {
        "method_code": method_code,
        "family_id": family_id,
        "rows": int(len(ds)),
        "metadata_path": metadata_path,
        "prototype": meta_vec.astype(np.float32).tolist(),
        "route_signature": extract_route_signature(meta_dict, method_dir=method_dir, method_code=method_code).astype(np.float32).tolist(),
    }
    return ds, info


def build_combined_dataset(selected_pairs: Sequence[Tuple[RTDataset, Dict[str, Any]]]) -> RTDataset:
    fps = np.concatenate([ds.fps for ds, _ in selected_pairs], axis=0).astype(np.float32)
    meta = np.concatenate([ds.meta for ds, _ in selected_pairs], axis=0).astype(np.float32)
    rts = np.concatenate([ds.rts for ds, _ in selected_pairs], axis=0).astype(np.float32)
    smiles = np.concatenate([ds.smiles for ds, _ in selected_pairs], axis=0)
    ids = np.concatenate([ds.ids for ds, _ in selected_pairs], axis=0)
    rt_mean = float(np.mean(rts))
    rt_std = float(np.std(rts) + 1e-8)
    rt_z = ((rts - rt_mean) / rt_std).astype(np.float32)
    return RTDataset(fps=fps, meta=meta, rts=rts, smiles=smiles, ids=ids, rt_mean=rt_mean, rt_std=rt_std, rt_z=rt_z)


def build_delta_dataset(method_ds: RTDataset, base_preds: Sequence[float]) -> DeltaDataset:
    base_pred = np.asarray(base_preds, dtype=np.float32).reshape(-1)
    delta = (method_ds.rts - base_pred).astype(np.float32)
    delta_mean = float(np.mean(delta))
    delta_std = float(np.std(delta) + 1e-8)
    delta_z = ((delta - delta_mean) / delta_std).astype(np.float32)
    return DeltaDataset(
        fps=method_ds.fps,
        meta=method_ds.meta,
        rts=method_ds.rts,
        smiles=method_ds.smiles,
        ids=method_ds.ids,
        base_pred=base_pred,
        delta=delta,
        delta_mean=delta_mean,
        delta_std=delta_std,
        delta_z=delta_z,
    )
