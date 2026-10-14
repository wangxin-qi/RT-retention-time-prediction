import json
import math
import os
from typing import Any, Dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default
        return float(text)
    except Exception:
        return default


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        if os.path.splitext(path)[1].lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    return data or {}


def _organic_fraction(block: Dict[str, Any]) -> float:
    acn = max(_safe_float(block.get("acn")), _safe_float(block.get("acetonitrile")))
    meoh = max(_safe_float(block.get("meoh")), _safe_float(block.get("methanol")))
    return acn + meoh


def _solvent_family(eluent: Dict[str, Any]) -> str:
    b = eluent.get("B", {}) or {}
    acn = max(_safe_float(b.get("acn")), _safe_float(b.get("acetonitrile")))
    meoh = max(_safe_float(b.get("meoh")), _safe_float(b.get("methanol")))
    if acn > 0 and meoh <= 0:
        return "acn"
    if meoh > 0 and acn <= 0:
        return "meoh"
    if acn > 0 or meoh > 0:
        return "mixed"
    return "unknown"


def metadata_to_condition_features(meta: Dict[str, Any]) -> Dict[str, float]:
    column = meta.get("column", {}) or {}
    eluent = meta.get("eluent", {}) or {}
    gradient = meta.get("gradient", {}) or {}
    a = eluent.get("A", {}) or {}
    b = eluent.get("B", {}) or {}

    length = _safe_float(column.get("length"))
    column_id = _safe_float(column.get("id"))
    particle = _safe_float(column.get("particle.size") or column.get("particle_size"))
    flowrate = _safe_float(column.get("flowrate"))
    temperature = _safe_float(column.get("temperature"))
    t0 = _safe_float(column.get("t0"))
    init_b = _safe_float(gradient.get("init_B"))
    final_b = _safe_float(gradient.get("final_B"))
    segments = max(_safe_float(gradient.get("segments"), 0.0), 0.0)
    total_time = _safe_float(
        gradient.get("total_time")
        or gradient.get("gradient_total_time")
        or gradient.get("time")
    )
    gradient_missing = 0.0 if (total_time > 0 or segments > 0) else 1.0

    diameter_area = math.pi * (column_id / 2.0) ** 2 if column_id > 0 else 0.0
    column_volume_proxy = length * diameter_area
    dead_time_proxy = column_volume_proxy / flowrate if flowrate > 0 else 0.0
    gradient_span = final_b - init_b
    gradient_time_proxy = total_time if total_time > 0 else 0.0
    gradient_slope_proxy = gradient_span / gradient_time_proxy if gradient_time_proxy > 0 else 0.0
    solvent_family = _solvent_family(eluent)

    return {
        "column_length": length,
        "column_id": column_id,
        "particle_size": particle,
        "flowrate": flowrate,
        "temperature": temperature,
        "t0": t0,
        "ph_a": _safe_float(a.get("pH")),
        "ph_b": _safe_float(b.get("pH")),
        "init_b": init_b,
        "final_b": final_b,
        "gradient_span": gradient_span,
        "gradient_segments": segments,
        "gradient_time_proxy": gradient_time_proxy,
        "gradient_slope_proxy": gradient_slope_proxy,
        "gradient_missing": gradient_missing,
        "column_volume_proxy": column_volume_proxy,
        "dead_time_proxy": dead_time_proxy,
        "linear_velocity_proxy": flowrate / diameter_area if diameter_area > 0 else 0.0,
        "organic_a": _organic_fraction(a),
        "organic_b": _organic_fraction(b),
        "solvent_is_acn": 1.0 if solvent_family == "acn" else 0.0,
        "solvent_is_meoh": 1.0 if solvent_family == "meoh" else 0.0,
        "solvent_is_mixed": 1.0 if solvent_family == "mixed" else 0.0,
        "solvent_is_unknown": 1.0 if solvent_family == "unknown" else 0.0,
    }


def load_condition_features(path_or_metadata: Any, prefix: str = "") -> Dict[str, float]:
    if isinstance(path_or_metadata, dict):
        metadata = path_or_metadata
    else:
        metadata = _load_yaml_or_json(str(path_or_metadata))
    features = metadata_to_condition_features(metadata)
    if not prefix:
        return features
    return {f"{prefix}{key}": value for key, value in features.items()}


def condition_delta_features(source: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(source) | set(target))
    return {f"delta_{key}": float(target.get(key, 0.0) - source.get(key, 0.0)) for key in keys}


def _norm_phase_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    return " ".join(text.split())


def phase_flags_from_metadata(meta: Dict[str, Any]) -> Dict[str, float]:
    """Stationary-phase one-hot flags parsed from metadata column name / chemistry.

    Mirrors the column-name heuristics used in the v47 dataset's ``_phase_flags`` so
    that the cross-condition similarity sees the same phase signal, but stays free of
    any rdkit dependency (this module is intentionally lightweight).
    """
    column = meta.get("column", {}) or {}
    column_name = _norm_phase_text(
        column.get("name")
        or column.get("type")
        or meta.get("column_name")
        or meta.get("source_row_column_name")
    )
    fixed_phase = _norm_phase_text(
        column.get("fixed_phase")
        or column.get("stationary_phase")
        or column.get("phase")
        or column.get("chemistry")
        or meta.get("fixed_phase")
    )
    text = f"{column_name} {fixed_phase}".strip()
    flags = {
        "phase_c18": 1.0 if ("c18" in text or "beh" in text or "xbridge" in text) else 0.0,
        "phase_hss_t3": 1.0 if ("hss t3" in text or "t3" in text) else 0.0,
        "phase_hilic": 1.0 if ("hilic" in text or "amide" in text or "zic" in text) else 0.0,
        "phase_phenyl": 1.0 if ("phenyl" in text or "hexyl" in text or "biphenyl" in text) else 0.0,
        "phase_unknown": 0.0,
    }
    if sum(flags.values()) <= 0:
        flags["phase_unknown"] = 1.0
    return flags
