import argparse
import importlib.util
import json
import math
import os
import re
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
CORE_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(CORE_ROOT)
WEIGHTS_ROOT = os.path.join(CORE_ROOT, "models", "weights")
ARCHIVE_ROOT = os.path.join(ROOT, "artifacts", "archive")
for _path in (HERE, CORE_ROOT, ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _first_existing_path(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0] if paths else ""

LOCAL_V47_RUN_DIR = os.path.join(ARCHIVE_ROOT, "runs", "v11_pro_meta_threehead_v47_calibrated_gate")
LOCAL_MCMRT_ADAPTER_DIR = os.path.join(ARCHIVE_ROOT, "runs", "v47_mcmrt_adapter")
LOCAL_MCMRT_DIR = os.path.join(ROOT, "data", "MCMRT")
LOCAL_GNN_MODEL_DIR = os.path.join(ROOT, "external", "third_party", "retention_time_gnn_min", "model")
DEFAULT_V47_TRAIN_SCRIPT = os.path.join(CORE_ROOT, "train", "train_v11_pro_meta_threehead_v47_calibrated_gate.py")
DEFAULT_V47_MANIFEST = _first_existing_path(
    os.path.join(WEIGHTS_ROOT, "v47_main", "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "overnight_meta_embed", "candidate_full_embed8", "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "overnight_meta_embed", "ablation_no_embed_full", "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "v11_pro_meta_threehead_v47_calibrated_gate_smoke_meta_embed", "router_manifest_v47.json"),
    os.path.join(LOCAL_V47_RUN_DIR, "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "v11_pro_meta_threehead_v47_calibrated_gate_pre_meta_embed_copy", "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "v11_pro_meta_threehead_v47_calibrated_gate", "router_manifest_v47.json"),
)
DEFAULT_PUBLIC_SAFE_FALLBACK_MANIFEST = _first_existing_path(
    os.path.join(WEIGHTS_ROOT, "v47_public_safe_fallback", "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "overnight_meta_embed", "ablation_no_embed_full", "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "v11_pro_meta_threehead_v47_calibrated_gate_pre_meta_embed_copy", "router_manifest_v47.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "v11_pro_meta_threehead_v47_calibrated_gate", "router_manifest_v47.json"),
)
DEFAULT_CROSS_CONDITION_MODEL = _first_existing_path(
    os.path.join(WEIGHTS_ROOT, "cross_condition_transfer_v2", "expert_inference_bundle.joblib"),
    os.path.join(WEIGHTS_ROOT, "cross_condition_transfer", "local_cross_condition_transfer_model.joblib"),
    os.path.join(
        ARCHIVE_ROOT,
        "cross_condition",
        "local_transfer_v2_desc_gate",
        "local_cross_condition_transfer_model.joblib",
    ),
)
DEFAULT_PUBLIC_AWARE_GLOBAL_MODEL = _first_existing_path(
    os.path.join(WEIGHTS_ROOT, "public_aware_global", "public_aware_global_model.joblib"),
    os.path.join(
        ARCHIVE_ROOT,
        "public_global",
        "candidate_v2_units_fixed",
        "public_aware_global_model.joblib",
    ),
)
DEFAULT_FAMILY_CONDITION_EXPERT_MODEL = _first_existing_path(
    os.path.join(WEIGHTS_ROOT, "family_condition_expert", "family_condition_expert_model.joblib"),
    os.path.join(
        ARCHIVE_ROOT,
        "family_condition",
        "candidate_v1",
        "family_condition_expert_model.joblib",
    ),
)
DEFAULT_CONDITION_WEIGHTS = _first_existing_path(
    os.path.join(WEIGHTS_ROOT, "condition_similarity", "condition_weights.json"),
    os.path.join(ARCHIVE_ROOT, "condition_similarity", "condition_weights.json"),
)
DEFAULT_MCMRT_ADAPTER_TRAIN_SCRIPT = os.path.join(CORE_ROOT, "train", "train_v47_mcmrt_adapter.py")
DEFAULT_MCMRT_ADAPTER_MANIFEST = _first_existing_path(
    os.path.join(WEIGHTS_ROOT, "mcmrt_adapter", "router_manifest_v47_mcmrt_adapter.json"),
    os.path.join(LOCAL_MCMRT_ADAPTER_DIR, "router_manifest_v47_mcmrt_adapter.json"),
    os.path.join(ARCHIVE_ROOT, "runs", "v47_mcmrt_adapter", "router_manifest_v47_mcmrt_adapter.json"),
)
DEFAULT_V47_DATASET_SCRIPT = os.path.join(CORE_ROOT, "data", "dataset_v11_pro_meta_threehead_v47_calibrated_gate.py")
DEFAULT_V47_MODEL_SCRIPT = os.path.join(CORE_ROOT, "models", "model_v11_pro_meta_threehead_v47_calibrated_gate.py")
DEFAULT_MCMRT_SPLIT_SCRIPT = os.path.join(CORE_ROOT, "data", "dataset_v11_pro_meta_threehead_v50_mcmrt_transfer.py")
DEFAULT_MCMRT_COMPARE_SCRIPT = os.path.join(ROOT, "analysis", "compare", "compare_v47_vs_gnn_mcmrt.py")
META_COLUMNS = [
    "chromatographic_column_length",
    "chromatographic_column_id",
    "chromatographic_particle_size",
    "chromatographic_flowrate",
    "chromatographic_t0",
    "chromatographic_temperature",
    "chromatographic_eluent_A_h2o",
    "chromatographic_eluent_A_acetonitrile",
    "chromatographic_eluent_A_meoh",
    "chromatographic_eluent_A_acetic_acid",
    "chromatographic_eluent_A_formic_acid",
    "chromatographic_eluent_A_additive_value",
    "chromatographic_eluent_A_pH",
    "chromatographic_eluent_B_h2o",
    "chromatographic_eluent_B_acetonitrile",
    "chromatographic_eluent_B_meoh",
    "chromatographic_eluent_B_acetic_acid",
    "chromatographic_eluent_B_formic_acid",
    "chromatographic_eluent_B_additive_value",
    "chromatographic_eluent_B_pH",
    "gradient_init_B",
    "gradient_final_B",
    "gradient_segments",
]

LOCAL_TEMPLATE_AUTO_SCORE_THRESHOLD = 0.95
PUBLIC_BENCHMARK_METHOD_HINTS = {
    # Keep this mapping conservative. Public benchmark names are not always safe
    # substitutes for explicit method codes because public files and processed
    # expert datasets can differ in feature-generation path.
}
PUBLIC_EXPERT_MIN_CODE = 9000
PUBLIC_EXPERT_ROUTE_MIN_ROWS_DEFAULT = 1000
PUBLIC_EXPERT_AUTO_SCORE_THRESHOLD = 0.93


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _nanmax_or_nan(values: Any) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if len(finite) else float("nan")


def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def _normalize_method_code_hint(x: Any) -> str:
    s = _norm_str(x)
    if not s:
        return ""
    if re.fullmatch(r"\d{1,4}", s):
        return s.zfill(4)
    return s


def _is_public_expert_code(x: Any) -> bool:
    code = _normalize_method_code_hint(x)
    return bool(code and code.isdigit() and int(code) >= PUBLIC_EXPERT_MIN_CODE)


@lru_cache(maxsize=256)
def _public_expert_training_rows(method_code: str) -> int:
    code = _normalize_method_code_hint(method_code)
    if not _is_public_expert_code(code):
        return 0
    for manifest_path in (
        DEFAULT_V47_MANIFEST,
        DEFAULT_PUBLIC_SAFE_FALLBACK_MANIFEST,
    ):
        if not manifest_path or not os.path.exists(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8-sig") as handle:
                manifest = json.load(handle)
            info = (manifest.get("experts") or {}).get(code) or {}
            rows = int(info.get("rows", 0))
            if rows > 0:
                return rows
        except Exception:
            pass
    method_dir = os.path.join(ROOT, "data", "processed_data", code)
    if not os.path.isdir(method_dir):
        return 0
    candidates = [
        os.path.join(method_dir, f"{code}_rtdata_canonical_success.tsv"),
        os.path.join(method_dir, f"{code}_rtdata_isomeric_success.tsv"),
    ]
    candidates.extend(
        os.path.join(method_dir, name)
        for name in os.listdir(method_dir)
        if name.endswith("_rtdata_canonical_success.tsv") or name.endswith("_rtdata_isomeric_success.tsv")
    )
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return 0
    return 0


def _is_protected_public_expert_route(detail: Dict[str, Any], *, min_rows: int = PUBLIC_EXPERT_ROUTE_MIN_ROWS_DEFAULT) -> bool:
    match_type = _norm_str(detail.get("v48_local_match_type") or detail.get("match_type"))
    if match_type not in {"explicit_method_code", "exact_method", "public_expert_similarity"}:
        return False
    code = (
        _normalize_method_code_hint(detail.get("v48_template_method_code"))
        or _normalize_method_code_hint(detail.get("v48_method_hint_used"))
        or _normalize_method_code_hint(detail.get("v48_local_router_top1"))
        or _normalize_method_code_hint(detail.get("v48_selected_expert"))
        or _normalize_method_code_hint(detail.get("method_code"))
    )
    return bool(_is_public_expert_code(code) and _public_expert_training_rows(code) >= int(min_rows))


def _is_protected_known_method_route(detail: Dict[str, Any]) -> bool:
    match_type = str(detail.get("v48_local_match_type") or "").strip()
    match_source = str(detail.get("v48_local_match_source") or "").strip()
    exact = match_type in {"explicit_method_code", "exact_method"} or match_source in {
        "explicit_method_code",
        "exact_method",
        "meta_signature_exact",
        "route_signature_exact",
    }
    code = (
        _normalize_method_code_hint(detail.get("v48_template_method_code"))
        or _normalize_method_code_hint(detail.get("v48_method_hint_used"))
        or _normalize_method_code_hint(detail.get("v48_local_router_top1"))
        or _normalize_method_code_hint(detail.get("v48_selected_expert"))
    )
    return bool(exact and code and not _is_public_expert_code(code))


def _should_promote_public_expert_similarity(
    local_match: Dict[str, Any],
    *,
    metadata_path: Optional[str],
    min_rows: int = PUBLIC_EXPERT_ROUTE_MIN_ROWS_DEFAULT,
    score_threshold: float = PUBLIC_EXPERT_AUTO_SCORE_THRESHOLD,
) -> bool:
    if not _path_looks_like_public_reference_data(metadata_path):
        return False
    method_code = _normalize_method_code_hint(local_match.get("similarity_top1_method"))
    score = _safe_float(local_match.get("similarity_top1_score"), default=float("nan"))
    if not np.isfinite(score) or score < float(score_threshold):
        return False
    return bool(_is_public_expert_code(method_code) and _public_expert_training_rows(method_code) >= int(min_rows))


def _add_parent_to_syspath(path_to_file: str) -> None:
    parent = os.path.dirname(os.path.abspath(path_to_file))
    if parent and parent not in sys.path:
        sys.path.insert(0, parent)


def _import_module_from_path(module_name: str, path: str):
    resolved_path = os.path.abspath(path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Module file not found: {resolved_path}")
    _add_parent_to_syspath(resolved_path)
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from path: {resolved_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _guess_default_input() -> Optional[str]:
    candidates = [
        os.path.join(ROOT, "integrated_compound_data_v47_input.tsv"),
        os.path.join(ROOT, "integrated_compound_data_v50_input.tsv"),
        os.path.join(ROOT, "integrated_compound_data_with_conditions.tsv"),
        os.path.join(ROOT, "integrated_output.tsv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_table(input_path: str, sep: Optional[str], sheet_name: Optional[str]) -> pd.DataFrame:
    lower = input_path.lower()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(input_path, sheet_name=sheet_name or 0)
    actual_sep = "\t" if (sep is None and lower.endswith(".tsv")) else ("," if sep is None else ("\t" if sep == r"\t" else sep))
    return pd.read_csv(input_path, sep=actual_sep)


def _load_yaml_or_json_fallback(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8-sig") as f:
        if ext in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(f)
        return json.load(f)


def _has_any_meta_columns(df: pd.DataFrame) -> bool:
    return any(col in df.columns for col in META_COLUMNS)


def _split_smiles(smiles_raw: str) -> List[str]:
    normalized = smiles_raw.replace(",", ";")
    return [s.strip() for s in normalized.split(";") if s.strip()] or [smiles_raw]


def _path_looks_like_mcmrt(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = os.path.abspath(path).lower().replace("/", "\\")
    name = os.path.basename(normalized)
    if "\\mcmrt\\" in normalized:
        return True
    if re.fullmatch(r"dataset_\d{1,2}\.(xlsx|xls|yaml|yml|json)", name):
        return True
    return False


def _path_looks_like_local_processed_data(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = os.path.abspath(path).lower().replace("/", "\\")
    processed_root = os.path.abspath(os.path.join(ROOT, "data", "processed_data")).lower().replace("/", "\\")
    return normalized == processed_root or normalized.startswith(processed_root + "\\")


def _path_looks_like_public_reference_data(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = os.path.abspath(path).lower().replace("/", "\\")
    public_roots = [
        os.path.abspath(os.path.join(ROOT, "data", "public_benchmarks")).lower().replace("/", "\\"),
        os.path.abspath(os.path.join(ROOT, "data", "public_rt_sources")).lower().replace("/", "\\"),
    ]
    return any(normalized == root or normalized.startswith(root + "\\") for root in public_roots)


def _meta_looks_like_local_processed_method(meta_dict: Dict[str, Any]) -> bool:
    code = (
        _normalize_method_code_hint(meta_dict.get("method_code_hint"))
        or _normalize_method_code_hint(meta_dict.get("template_method_code"))
        or _normalize_method_code_hint(meta_dict.get("id"))
    )
    if not code or not code.isdigit() or int(code) >= PUBLIC_EXPERT_MIN_CODE:
        return False
    return os.path.isdir(os.path.join(ROOT, "data", "processed_data", code))


def _should_admit_public_global_branch(
    *,
    public_safe_fallback: bool,
    meta_dict: Dict[str, Any],
    metadata_path: Optional[str],
    input_path: Optional[str],
    protected_public_expert_route: bool = False,
) -> bool:
    if protected_public_expert_route:
        return False
    if _path_looks_like_local_processed_data(metadata_path) or _path_looks_like_local_processed_data(input_path):
        return False
    if _meta_looks_like_local_processed_method(meta_dict):
        return False
    if not public_safe_fallback and not (
        _path_looks_like_public_reference_data(metadata_path)
        or _path_looks_like_public_reference_data(input_path)
    ):
        return False
    return True


def _should_admit_cross_condition_branch(
    *,
    public_safe_fallback: bool,
    meta_dict: Dict[str, Any],
    metadata_path: Optional[str],
    input_path: Optional[str],
    protected_public_expert_route: bool = False,
    protected_known_method_route: bool = False,
) -> bool:
    del public_safe_fallback, meta_dict, metadata_path, input_path
    return not (protected_public_expert_route or protected_known_method_route)


def _should_admit_family_condition_branch(
    *,
    public_safe_fallback: bool,
    meta_dict: Dict[str, Any],
    metadata_path: Optional[str],
    input_path: Optional[str],
    protected_public_expert_route: bool = False,
) -> bool:
    return _should_admit_public_global_branch(
        public_safe_fallback=public_safe_fallback,
        meta_dict=meta_dict,
        metadata_path=metadata_path,
        input_path=input_path,
        protected_public_expert_route=protected_public_expert_route,
    )


def build_meta_dict(
    *,
    column: Dict[str, Any],
    solvent_a: Dict[str, Any],
    solvent_b: Dict[str, Any],
    gradient: Dict[str, Any],
    method_code_hint: Any = None,
    row_id: Any = None,
    source_row_column_name: Any = None,
    source_gradient_profile: Any = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_hint = _normalize_method_code_hint(method_code_hint)
    meta_dict = {
        "id": normalized_hint or _normalize_method_code_hint(row_id) or _norm_str(row_id),
        "method_code_hint": normalized_hint,
        "column": column,
        "eluent": {"A": solvent_a, "B": solvent_b},
        "gradient": gradient,
        "source_row_column_name": _norm_str(source_row_column_name),
        "source_gradient_profile": _norm_str(source_gradient_profile),
    }
    if extra_fields:
        meta_dict.update(extra_fields)
    return meta_dict


def build_meta_dict_from_row(row: pd.Series) -> Dict[str, Any]:
    solvent_a = {}
    solvent_b = {}
    column = {
        "length": _safe_float(row.get("chromatographic_column_length"), 0.0),
        "id": _safe_float(row.get("chromatographic_column_id"), 0.0),
        "particle.size": _safe_float(row.get("chromatographic_particle_size"), 0.0),
        "flowrate": _safe_float(row.get("chromatographic_flowrate"), 0.0),
        "t0": _safe_float(row.get("chromatographic_t0"), 0.0),
        "temperature": _safe_float(row.get("chromatographic_temperature"), 0.0),
    }

    a_h2o = _safe_float(row.get("chromatographic_eluent_A_h2o"), 0.0)
    a_acn = _safe_float(row.get("chromatographic_eluent_A_acetonitrile"), 0.0)
    a_meoh = _safe_float(row.get("chromatographic_eluent_A_meoh"), 0.0)
    if a_h2o > 0:
        solvent_a["h2o"] = a_h2o
    if a_acn > 0:
        solvent_a["acetonitrile"] = a_acn
    if a_meoh > 0:
        solvent_a["meoh"] = a_meoh

    a_value = _safe_float(
        row.get("chromatographic_eluent_A_acetic_acid"),
        _safe_float(
            row.get("chromatographic_eluent_A_formic_acid"),
            _safe_float(row.get("chromatographic_eluent_A_additive_value"), 0.0),
        ),
    )
    if a_value > 0:
        solvent_a["value"] = a_value
        solvent_a["formic"] = {"value": a_value}
    a_ph = _safe_float(row.get("chromatographic_eluent_A_pH"), 0.0)
    if a_ph > 0:
        solvent_a["pH"] = a_ph

    b_h2o = _safe_float(row.get("chromatographic_eluent_B_h2o"), 0.0)
    b_acn = _safe_float(row.get("chromatographic_eluent_B_acetonitrile"), 0.0)
    b_meoh = _safe_float(row.get("chromatographic_eluent_B_meoh"), 0.0)
    if b_h2o > 0:
        solvent_b["h2o"] = b_h2o
    if b_acn > 0:
        solvent_b["acetonitrile"] = b_acn
    if b_meoh > 0:
        solvent_b["meoh"] = b_meoh

    b_value = _safe_float(
        row.get("chromatographic_eluent_B_acetic_acid"),
        _safe_float(
            row.get("chromatographic_eluent_B_formic_acid"),
            _safe_float(row.get("chromatographic_eluent_B_additive_value"), 0.0),
        ),
    )
    if b_value > 0:
        solvent_b["value"] = b_value
        solvent_b["formic"] = {"value": b_value}
    b_ph = _safe_float(row.get("chromatographic_eluent_B_pH"), 0.0)
    if b_ph > 0:
        solvent_b["pH"] = b_ph

    gradient = {
        "init_B": _safe_float(row.get("gradient_init_B"), 0.0),
        "final_B": _safe_float(row.get("gradient_final_B"), 0.0),
        "segments": _safe_float(row.get("gradient_segments"), 0.0),
    }

    method_code_hint = (
        _normalize_method_code_hint(row.get("method_code"))
        or _normalize_method_code_hint(row.get("chromatographic_method_code"))
        or _normalize_method_code_hint(row.get("lc_method_code"))
        or _normalize_method_code_hint(row.get("metadata_id"))
    )
    return build_meta_dict(
        column=column,
        solvent_a=solvent_a,
        solvent_b=solvent_b,
        gradient=gradient,
        method_code_hint=method_code_hint,
        row_id=row.get("id"),
        source_row_column_name=row.get("chromatographic_column_name"),
        source_gradient_profile=row.get("gradient_profile"),
    )


def load_metadata_any(path: str, train_mod, mcmrt_compare_mod) -> Dict[str, Any]:
    lower = path.lower()
    load_meta = getattr(train_mod, "load_yaml_or_json", None)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        extract_meta = getattr(mcmrt_compare_mod, "extract_metadata_from_setup_sheet", None)
        if extract_meta is None:
            raise RuntimeError("MCMRT metadata extractor is unavailable.")
        return extract_meta(path)
    if load_meta is not None:
        return load_meta(path)
    return _load_yaml_or_json_fallback(path)


def _metadata_with_gradient_summary(meta_dict: Dict[str, Any], method_dir: Optional[str], method_code: Optional[str]) -> Dict[str, Any]:
    """Return metadata enriched with gradient TSV summary for condition-similarity only."""
    if not method_dir or not method_code:
        return meta_dict
    path = os.path.join(method_dir, f"{method_code}_gradient.tsv")
    if not os.path.exists(path):
        return meta_dict
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return meta_dict

    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        item: Dict[str, float] = {}
        for key, value in row.items():
            key_text = re.sub(r"[^a-z0-9]+", "", str(key).strip().lower())
            if key_text in {"t", "time", "tmin", "timemin", "min", "minute"}:
                item["time"] = _safe_float(value, default=float("nan"))
            elif key_text in {"b", "bphase", "bpercent", "bpercent", "phaseb", "solventb"}:
                item["B"] = _safe_float(value, default=float("nan"))
        if "time" in item and "B" in item and np.isfinite(item["time"]) and np.isfinite(item["B"]):
            rows.append(item)
    rows.sort(key=lambda item: item["time"])
    if not rows:
        return meta_dict

    out = dict(meta_dict)
    gradient = dict(out.get("gradient", {}) or {})
    gradient["table"] = rows
    gradient.setdefault("segments", max(1, len(rows) - 1))
    gradient.setdefault("init_B", rows[0]["B"])
    gradient.setdefault("final_B", rows[-1]["B"])
    gradient.setdefault("time", rows[-1]["time"])
    gradient.setdefault("total_time", rows[-1]["time"])
    out["gradient"] = gradient
    return out


class McmrtMetadataMatcher:
    def __init__(self, dataset_mod, mcmrt_split_mod):
        self.dataset_mod = dataset_mod
        self.prototype_rows: List[Dict[str, Any]] = []
        method_ids = list(getattr(mcmrt_split_mod, "METHOD_IDS", []))
        load_method_split = getattr(mcmrt_split_mod, "load_method_split", None)
        if load_method_split is None:
            raise RuntimeError("MCMRT split module does not expose load_method_split().")

        for method_name in method_ids:
            split = load_method_split(method_name)
            meta = split["metadata"]
            vec = np.asarray(dataset_mod.extract_method_meta_vector(meta), dtype=np.float32)
            family_id = str(dataset_mod.auto_family_id_from_metadata(meta))
            self.prototype_rows.append(
                {
                    "method_name": str(method_name),
                    "family_id": family_id,
                    "meta_vec": vec,
                }
            )

    def classify(
        self,
        meta_dict: Dict[str, Any],
        mode: str,
        metadata_path: Optional[str] = None,
        input_path: Optional[str] = None,
        atol: float = 1e-6,
    ) -> Dict[str, Any]:
        meta_vec = np.asarray(self.dataset_mod.extract_method_meta_vector(meta_dict), dtype=np.float32)
        query_family_id = str(self.dataset_mod.auto_family_id_from_metadata(meta_dict))
        best_row = None
        best_l1 = None
        best_l2 = None
        exact_match = False

        for row in self.prototype_rows:
            proto = row["meta_vec"]
            l1 = float(np.mean(np.abs(meta_vec - proto)))
            l2 = float(np.sqrt(np.mean((meta_vec - proto) ** 2)))
            if best_l1 is None or l1 < best_l1:
                best_l1 = l1
                best_l2 = l2
                best_row = row
            if np.allclose(meta_vec, proto, atol=atol, rtol=0.0):
                exact_match = True
                best_row = row
                best_l1 = l1
                best_l2 = l2
                break

        path_hint = _path_looks_like_mcmrt(metadata_path) or _path_looks_like_mcmrt(input_path)
        family_match = any(query_family_id == row["family_id"] for row in self.prototype_rows)

        if mode == "mcmrt":
            is_mcmrt = True
            reason = "forced_mcmrt"
        elif mode == "v47":
            is_mcmrt = False
            reason = "forced_v47"
        elif path_hint:
            is_mcmrt = True
            reason = "mcmrt_path_hint"
        elif exact_match:
            is_mcmrt = True
            reason = "mcmrt_metadata_exact_match"
        else:
            is_mcmrt = False
            reason = "fallback_to_v47"

        return {
            "is_mcmrt": bool(is_mcmrt),
            "reason": reason,
            "query_family_id": query_family_id,
            "family_match": bool(family_match),
            "best_method_name": None if best_row is None else str(best_row["method_name"]),
            "best_family_id": None if best_row is None else str(best_row["family_id"]),
            "best_l1": None if best_l1 is None else float(best_l1),
            "best_l2": None if best_l2 is None else float(best_l2),
            "meta_vec": meta_vec,
        }


class LocalMethodTemplateMatcher:
    def __init__(self, dataset_mod, router_manifest: Dict[str, Any], condition_sim=None):
        self.dataset_mod = dataset_mod
        self.condition_sim = condition_sim
        self.rows: List[Dict[str, Any]] = []
        experts = router_manifest.get("experts", {}) if isinstance(router_manifest, dict) else {}
        for code, info in experts.items():
            metadata_path = info.get("metadata_path")
            if not metadata_path or not os.path.exists(metadata_path):
                continue
            try:
                meta_dict = dataset_mod.load_yaml_or_json(metadata_path)
                method_dir = os.path.dirname(metadata_path)
                inferred_method_code = os.path.basename(method_dir)
                method_code = str(code)
                if inferred_method_code.isdigit():
                    method_code = inferred_method_code
                condition_meta = _metadata_with_gradient_summary(meta_dict, method_dir, method_code)
                template_profile = self._extract_template_profile(condition_meta)
                meta_vec = np.asarray(
                    dataset_mod.extract_method_meta_vector(meta_dict, method_dir=method_dir, method_code=method_code),
                    dtype=np.float32,
                )
                route_signature = np.asarray(
                    dataset_mod.extract_route_signature(meta_dict, method_dir=method_dir, method_code=method_code),
                    dtype=np.float32,
                )
                condition_vec = self.condition_sim.vector(condition_meta) if self.condition_sim is not None else None
                self.rows.append(
                    {
                        "method_code": str(code),
                        "family_id": str(info.get("family_id", dataset_mod.auto_family_id_from_metadata(meta_dict, method_dir=method_dir, method_code=method_code))),
                        "metadata_path": metadata_path,
                        "meta_vec": meta_vec,
                        "route_signature": route_signature,
                        "condition_vec": condition_vec,
                        "meta_key": tuple(np.round(meta_vec, 6).tolist()),
                        "route_key": tuple(np.round(route_signature, 6).tolist()),
                        "template_name": f"{method_code} | {template_profile['column_name_norm'] or 'local_method'}",
                        **template_profile,
                    }
                )
            except Exception:
                continue

        self.by_route_key: Dict[tuple, List[Dict[str, Any]]] = {}
        self.by_meta_key: Dict[tuple, List[Dict[str, Any]]] = {}
        for row in self.rows:
            self.by_route_key.setdefault(row["route_key"], []).append(row)
            self.by_meta_key.setdefault(row["meta_key"], []).append(row)
        self.by_method_code = {row["method_code"]: row for row in self.rows}

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = _norm_str(value).lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _extract_template_profile(self, meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        column = meta_dict.get("column", {}) if isinstance(meta_dict, dict) else {}
        gradient = meta_dict.get("gradient", {}) if isinstance(meta_dict, dict) else {}
        gradient_rows = gradient.get("table", []) if isinstance(gradient, dict) else []
        total_time = 0.0
        if isinstance(gradient_rows, list) and gradient_rows:
            total_time = max(_safe_float(row.get("time")) for row in gradient_rows if isinstance(row, dict))
        if total_time <= 0:
            total_time = _safe_float(gradient.get("time"))
        column_name = (
            meta_dict.get("source_row_column_name")
            or meta_dict.get("column_name")
            or column.get("name")
        )
        return {
            "column_name_norm": self._normalize_text(column_name),
            "column_length": _safe_float(column.get("length")),
            "column_id": _safe_float(column.get("id")),
            "particle_size": _safe_float(column.get("particle.size")),
            "flowrate": _safe_float(column.get("flowrate")),
            "temperature": _safe_float(column.get("temperature")),
            "gradient_total_time": total_time,
            "gradient_init_B": _safe_float(gradient.get("init_B")),
            "gradient_final_B": _safe_float(gradient.get("final_B")),
            "gradient_segments": _safe_float(gradient.get("segments")),
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_similarity(self, query_meta_vec: np.ndarray, query_route_signature: np.ndarray, query_profile: Dict[str, Any], row: Dict[str, Any], query_condition_vec=None) -> Dict[str, Any]:
        meta_l1 = float(np.mean(np.abs(query_meta_vec - row["meta_vec"])))
        route_l1 = float(np.mean(np.abs(query_route_signature - row["route_signature"])))
        cosine_score = self._cosine_similarity(query_route_signature, row["route_signature"])
        weighted_distance = None
        weighted_similarity = None
        if self.condition_sim is not None and query_condition_vec is not None and row.get("condition_vec") is not None:
            weighted_distance = self.condition_sim.distance_vec(query_condition_vec, row["condition_vec"])
            weighted_similarity = self.condition_sim.similarity_from_distance(weighted_distance)
            score = weighted_similarity
        else:
            score = cosine_score
        flowrate_diff = abs(float(query_profile["flowrate"]) - float(row["flowrate"]))
        temperature_diff = abs(float(query_profile["temperature"]) - float(row["temperature"]))
        column_length_diff = abs(float(query_profile["column_length"]) - float(row["column_length"]))
        gradient_time_diff = abs(float(query_profile["gradient_total_time"]) - float(row["gradient_total_time"]))
        particle_size_diff = abs(float(query_profile["particle_size"]) - float(row["particle_size"]))
        init_b_diff = abs(float(query_profile["gradient_init_B"]) - float(row["gradient_init_B"]))
        final_b_diff = abs(float(query_profile["gradient_final_B"]) - float(row["gradient_final_B"]))
        column_name_match = bool(
            query_profile["column_name_norm"]
            and row["column_name_norm"]
            and query_profile["column_name_norm"] == row["column_name_norm"]
        )
        return {
            "method_code": row["method_code"],
            "family_id": row["family_id"],
            "score": float(score),
            "cosine_score": float(cosine_score),
            "weighted_distance": None if weighted_distance is None else float(weighted_distance),
            "weighted_similarity": None if weighted_similarity is None else float(weighted_similarity),
            "meta_l1": meta_l1,
            "route_l1": route_l1,
            "column_name_match": column_name_match,
            "flowrate_diff": float(flowrate_diff),
            "temperature_diff": float(temperature_diff),
            "column_length_diff": float(column_length_diff),
            "gradient_time_diff": float(gradient_time_diff),
            "particle_size_diff": float(particle_size_diff),
            "gradient_init_b_diff": float(init_b_diff),
            "gradient_final_b_diff": float(final_b_diff),
        }

    def classify(self, meta_dict: Dict[str, Any], metadata_path: Optional[str] = None) -> Dict[str, Any]:
        method_dir = None
        method_code = None
        if metadata_path and os.path.exists(metadata_path):
            parent = os.path.dirname(metadata_path)
            if os.path.basename(parent).isdigit():
                method_dir = parent
                method_code = os.path.basename(parent)

        meta_vec = np.asarray(
            self.dataset_mod.extract_method_meta_vector(meta_dict, method_dir=method_dir, method_code=method_code),
            dtype=np.float32,
        )
        route_signature = np.asarray(
            self.dataset_mod.extract_route_signature(meta_dict, method_dir=method_dir, method_code=method_code),
            dtype=np.float32,
        )
        query_profile = self._extract_template_profile(meta_dict)
        query_condition_vec = self.condition_sim.vector(meta_dict) if self.condition_sim is not None else None
        meta_key = tuple(np.round(meta_vec, 6).tolist())
        route_key = tuple(np.round(route_signature, 6).tolist())
        similarity_rows = [
            self._build_similarity(meta_vec, route_signature, query_profile, row, query_condition_vec=query_condition_vec)
            for row in self.rows
        ]
        similarity_rows.sort(key=lambda item: item["score"], reverse=True)
        top1 = similarity_rows[0] if similarity_rows else None
        top2 = similarity_rows[1] if len(similarity_rows) > 1 else None
        score_margin = float(top1["score"] - top2["score"]) if top1 and top2 else None
        top_candidates = [
            {
                "method_code": row["method_code"],
                "family_id": row["family_id"],
                "score": float(row["score"]),
            }
            for row in similarity_rows[:8]
        ]
        similarity_summary = {
            "similarity_top1_template_name": "" if top1 is None else self.by_method_code.get(top1["method_code"], {}).get("template_name", top1["method_code"]),
            "similarity_top1_method": top1["method_code"] if top1 else "",
            "similarity_top1_family": top1["family_id"] if top1 else "",
            "similarity_top1_score": None if top1 is None else float(top1["score"]),
            "similarity_top1_meta_l1": None if top1 is None else float(top1["meta_l1"]),
            "similarity_top1_route_l1": None if top1 is None else float(top1["route_l1"]),
            "similarity_top1_column_name_match": bool(top1["column_name_match"]) if top1 else False,
            "similarity_top1_flowrate_diff": None if top1 is None else float(top1["flowrate_diff"]),
            "similarity_top1_temperature_diff": None if top1 is None else float(top1["temperature_diff"]),
            "similarity_top1_column_length_diff": None if top1 is None else float(top1["column_length_diff"]),
            "similarity_top1_gradient_time_diff": None if top1 is None else float(top1["gradient_time_diff"]),
            "similarity_top2_method": top2["method_code"] if top2 else "",
            "similarity_top2_family": top2["family_id"] if top2 else "",
            "similarity_top2_score": None if top2 is None else float(top2["score"]),
            "similarity_score_margin": score_margin,
            "similarity_top_methods": [row["method_code"] for row in similarity_rows[:5]],
            "similarity_top_families": sorted(set(row["family_id"] for row in similarity_rows[:5])),
            "similarity_top_candidates": top_candidates,
        }

        # Condition-weighted mode: the top-1 is always returned as a *soft* candidate and
        # alpha is gated downstream by the weighted similarity. Legacy mode keeps the old
        # raw-cosine 0.95 auto-promotion to a hard hint.
        soft = self.condition_sim is not None
        auto_use_top1 = bool(top1 is not None and float(top1["score"]) >= LOCAL_TEMPLATE_AUTO_SCORE_THRESHOLD)

        route_matches = self.by_route_key.get(route_key, [])
        if route_matches:
            if len(route_matches) == 1:
                row = route_matches[0]
                return {
                    "match_type": "exact_method",
                    "match_source": "route_signature_exact",
                    "method_code": row["method_code"],
                    "candidate_methods": [row["method_code"]],
                    "candidate_families": [row["family_id"]],
                    **similarity_summary,
                }
            if soft:
                return {
                    "match_type": "condition_similarity_soft" if top1 else "no_hint",
                    "match_source": "condition_similarity_route_collision_top1" if top1 else "route_signature_collision_no_candidate",
                    "method_code": top1["method_code"] if top1 else None,
                    "candidate_methods": sorted([row["method_code"] for row in route_matches]),
                    "candidate_families": sorted(set(row["family_id"] for row in route_matches)),
                    **similarity_summary,
                }
            return {
                "match_type": "similarity_method" if auto_use_top1 else "no_hint",
                "match_source": "route_signature_collision_top1" if auto_use_top1 else "route_signature_collision_below_threshold",
                "method_code": (top1["method_code"] if top1 else route_matches[0]["method_code"]) if auto_use_top1 else None,
                "candidate_methods": sorted([row["method_code"] for row in route_matches]),
                "candidate_families": sorted(set(row["family_id"] for row in route_matches)),
                **similarity_summary,
            }

        meta_matches = self.by_meta_key.get(meta_key, [])
        if meta_matches:
            if len(meta_matches) == 1:
                row = meta_matches[0]
                return {
                    "match_type": "exact_method",
                    "match_source": "meta_signature_exact",
                    "method_code": row["method_code"],
                    "candidate_methods": [row["method_code"]],
                    "candidate_families": [row["family_id"]],
                    **similarity_summary,
                }
            if soft:
                return {
                    "match_type": "condition_similarity_soft" if top1 else "no_hint",
                    "match_source": "condition_similarity_meta_collision_top1" if top1 else "meta_signature_collision_no_candidate",
                    "method_code": top1["method_code"] if top1 else None,
                    "candidate_methods": sorted([row["method_code"] for row in meta_matches]),
                    "candidate_families": sorted(set(row["family_id"] for row in meta_matches)),
                    **similarity_summary,
                }
            return {
                "match_type": "similarity_method" if auto_use_top1 else "no_hint",
                "match_source": "meta_signature_collision_top1" if auto_use_top1 else "meta_signature_collision_below_threshold",
                "method_code": (top1["method_code"] if top1 else meta_matches[0]["method_code"]) if auto_use_top1 else None,
                "candidate_methods": sorted([row["method_code"] for row in meta_matches]),
                "candidate_families": sorted(set(row["family_id"] for row in meta_matches)),
                **similarity_summary,
            }

        if top1 is not None:
            if soft:
                return {
                    "match_type": "condition_similarity_soft",
                    "match_source": "condition_similarity_top1",
                    "method_code": top1["method_code"],
                    "candidate_methods": [row["method_code"] for row in similarity_rows[:5]],
                    "candidate_families": sorted(set(row["family_id"] for row in similarity_rows[:5])),
                    **similarity_summary,
                }
            return {
                "match_type": "similarity_method" if auto_use_top1 else "no_hint",
                "match_source": "route_signature_cosine_top1" if auto_use_top1 else "route_signature_cosine_below_threshold",
                "method_code": top1["method_code"] if auto_use_top1 else None,
                "candidate_methods": [row["method_code"] for row in similarity_rows[:5]],
                "candidate_families": sorted(set(row["family_id"] for row in similarity_rows[:5])),
                **similarity_summary,
            }

        return {
            "match_type": "no_hint",
            "match_source": "no_template_match",
            "method_code": None,
            "candidate_methods": [],
            "candidate_families": [],
            **similarity_summary,
        }

    def recommend(self, meta_dict: Dict[str, Any], metadata_path: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        method_dir = None
        method_code = None
        if metadata_path and os.path.exists(metadata_path):
            parent = os.path.dirname(metadata_path)
            if os.path.basename(parent).isdigit():
                method_dir = parent
                method_code = os.path.basename(parent)

        meta_vec = np.asarray(
            self.dataset_mod.extract_method_meta_vector(meta_dict, method_dir=method_dir, method_code=method_code),
            dtype=np.float32,
        )
        route_signature = np.asarray(
            self.dataset_mod.extract_route_signature(meta_dict, method_dir=method_dir, method_code=method_code),
            dtype=np.float32,
        )
        query_profile = self._extract_template_profile(meta_dict)
        query_condition_vec = self.condition_sim.vector(meta_dict) if self.condition_sim is not None else None
        rows = [
            self._build_similarity(meta_vec, route_signature, query_profile, row, query_condition_vec=query_condition_vec)
            for row in self.rows
        ]
        rows.sort(key=lambda item: item["score"], reverse=True)
        result = []
        for item in rows[: max(1, top_k)]:
            row = self.by_method_code.get(item["method_code"], {})
            result.append(
                {
                    "template_name": f"{item['method_code']} | {row.get('column_name_norm', '') or 'local_method'}",
                    "method_code": item["method_code"],
                    "family_id": item["family_id"],
                    "score": item["score"],
                    "meta_l1": item["meta_l1"],
                    "route_l1": item["route_l1"],
                    "column_name_match": item["column_name_match"],
                    "flowrate_diff": item["flowrate_diff"],
                    "temperature_diff": item["temperature_diff"],
                    "column_length_diff": item["column_length_diff"],
                    "gradient_time_diff": item["gradient_time_diff"],
                    "metadata_path": row.get("metadata_path"),
                }
            )
        return result


class UnifiedV48Predictor:
    def __init__(
        self,
        v47_train_script: str,
        v47_manifest: str,
        adapter_train_script: str,
        adapter_manifest: str,
        v47_dataset_script: str,
        mcmrt_split_script: str,
        mcmrt_compare_script: str,
        public_safe_fallback_manifest: Optional[str] = DEFAULT_PUBLIC_SAFE_FALLBACK_MANIFEST,
        public_safe_fallback: bool = True,
        condition_weights_path: Optional[str] = DEFAULT_CONDITION_WEIGHTS,
        use_condition_similarity: bool = True,
    ):
        self.v47_train_mod = _import_module_from_path("v48_v47_train", v47_train_script)
        self.adapter_train_mod = _import_module_from_path("v48_mcmrt_adapter_train", adapter_train_script)
        self.dataset_mod = _import_module_from_path("v48_v47_dataset", v47_dataset_script)
        self.model_mod = _import_module_from_path("v48_v47_model", DEFAULT_V47_MODEL_SCRIPT)
        self.mcmrt_split_mod = _import_module_from_path("v48_mcmrt_split", mcmrt_split_script)
        self.mcmrt_compare_mod = _import_module_from_path("v48_mcmrt_compare", mcmrt_compare_script)

        if os.path.isdir(LOCAL_MCMRT_DIR):
            self.mcmrt_split_mod.ROOT = ROOT
        self.adapter_train_mod.BACKBONE_TRAIN_SCRIPT = v47_train_script
        self.adapter_train_mod.BACKBONE_ROUTER_MANIFEST = v47_manifest
        if hasattr(self.adapter_train_mod, "adapter"):
            self.adapter_train_mod.adapter.BACKBONE_TRAIN_SCRIPT = v47_train_script
            self.adapter_train_mod.adapter.BACKBONE_ROUTER_MANIFEST = v47_manifest

        self.v47_predictor = self.v47_train_mod.build_predictor(v47_manifest)
        self.public_safe_fallback = bool(public_safe_fallback)
        self.public_safe_fallback_manifest = public_safe_fallback_manifest or ""
        self.public_safe_predictor = None
        if (
            self.public_safe_fallback
            and self.public_safe_fallback_manifest
            and os.path.exists(self.public_safe_fallback_manifest)
            and os.path.abspath(self.public_safe_fallback_manifest) != os.path.abspath(v47_manifest)
        ):
            self.public_safe_predictor = self.v47_train_mod.build_predictor(self.public_safe_fallback_manifest)
        self.adapter_predictor = self.adapter_train_mod.build_predictor(adapter_manifest)
        self.condition_sim = None
        if use_condition_similarity and condition_weights_path:
            from core.cross_condition.condition_similarity import load_condition_similarity

            self.condition_sim = load_condition_similarity(condition_weights_path)
            if self.condition_sim is None:
                print(f"[v48] condition similarity weights not found; using legacy cosine routing ({condition_weights_path})")
            else:
                print(f"[v48] condition-weighted similarity routing enabled ({condition_weights_path})")
        self.matcher = McmrtMetadataMatcher(self.dataset_mod, self.mcmrt_split_mod)
        self.local_method_matcher = LocalMethodTemplateMatcher(
            self.dataset_mod, getattr(self.v47_predictor, "manifest", {}), condition_sim=self.condition_sim
        )

        self.smiles_to_fp = getattr(self.v47_train_mod, "smiles_to_fp", None)
        self.extract_method_meta_vector = getattr(self.v47_train_mod, "extract_method_meta_vector", None)
        self.extract_route_signature = getattr(self.v47_train_mod, "extract_route_signature", None)
        if self.smiles_to_fp is None or self.extract_method_meta_vector is None or self.extract_route_signature is None:
            raise RuntimeError("v47 training script does not expose smiles_to_fp, extract_method_meta_vector, or extract_route_signature.")
        self._meta_vec_cache: Dict[str, Any] = {}

    @staticmethod
    def _get_column_meta(meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        return meta_dict.get("column", {}) if isinstance(meta_dict, dict) else {}

    @staticmethod
    def _get_gradient_meta(meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        return meta_dict.get("gradient", {}) if isinstance(meta_dict, dict) else {}

    def _extract_method_meta_vector_cached(self, meta_dict: Dict[str, Any]):
        cache_key = json.dumps(meta_dict, sort_keys=True, ensure_ascii=True, default=str)
        cached = self._meta_vec_cache.get(cache_key)
        if cached is None:
            cached = self.extract_method_meta_vector(meta_dict)
            self._meta_vec_cache[cache_key] = cached
        return cached

    @staticmethod
    def _attach_route_fields(
        out: Dict[str, Any],
        *,
        route_reason: str,
        is_mcmrt: bool,
        query_family_id: Optional[str],
        best_mcmrt_method: Optional[str],
        best_mcmrt_family_id: Optional[str],
        best_mcmrt_l1: Optional[float],
        best_mcmrt_l2: Optional[float],
        selected_expert: str,
        route_family: str,
        fallback_used: bool,
        local_router_top1: Optional[str] = None,
    ) -> Dict[str, Any]:
        out["v48_route_reason"] = route_reason
        out["v48_is_mcmrt"] = bool(is_mcmrt)
        out["v48_query_family_id"] = query_family_id
        out["v48_best_mcmrt_method"] = best_mcmrt_method
        out["v48_best_mcmrt_family_id"] = best_mcmrt_family_id
        out["v48_best_mcmrt_l1"] = best_mcmrt_l1
        out["v48_best_mcmrt_l2"] = best_mcmrt_l2
        out["v48_selected_expert"] = selected_expert
        out["v48_route_family"] = route_family
        out["v48_fallback_used"] = bool(fallback_used)
        out["v48_local_router_top1"] = local_router_top1 or out.get("router_top1")
        out["v48_router_gate_pass"] = bool(out.get("router_gate_pass", False))
        out["v48_gate_strength"] = float(out.get("gate_strength", 0.0) or 0.0)
        out["v48_alpha"] = float(out.get("alpha", 0.0) or 0.0)
        out["v48_alpha_base"] = float(out.get("alpha_base", 0.0) or 0.0)
        out["v48_route_similarity"] = out.get("route_similarity")
        out["v48_similarity_gate"] = out.get("similarity_gate")
        out["v48_final_reason"] = out.get("final_reason", "")
        out["v48_experts"] = out.get("experts", [])
        out["v48_expert_weights"] = out.get("weights", {})
        out["v48_family_weights"] = out.get("family_weights", {})
        out["v48_delta_preds"] = out.get("delta_preds", {})
        out["v48_delta_experts_used"] = out.get("delta_used", [])
        out["v48_confidence_tier"] = str(out.get("confidence_tier", "low") or "low")
        out["v48_family_confidence"] = float(out.get("family_confidence", 0.0) or 0.0)
        out["v48_correction_cap"] = float(out.get("correction_cap", 0.0) or 0.0)
        calibration = out.get("calibration", {}) if isinstance(out.get("calibration"), dict) else {}
        out["v48_family_top1"] = calibration.get("family_top1")
        out["v48_family_margin"] = calibration.get("family_margin")
        out["v48_top_expert_weight"] = calibration.get("top_expert_weight")
        out["v48_expert_margin"] = calibration.get("expert_margin")
        out["v48_collision_candidates"] = out.get("collision_candidates", [])
        out.setdefault("v48_local_match_type", "")
        out.setdefault("v48_local_match_source", "")
        out.setdefault("v48_local_candidate_methods", [])
        out.setdefault("v48_local_candidate_families", [])
        out.setdefault("v48_method_hint_used", "")
        out.setdefault("v48_local_similarity_top1_method", "")
        out.setdefault("v48_local_similarity_top1_template_name", "")
        out.setdefault("v48_local_similarity_top1_family", "")
        out.setdefault("v48_local_similarity_top1_score", None)
        out.setdefault("v48_local_similarity_top1_meta_l1", None)
        out.setdefault("v48_local_similarity_top1_route_l1", None)
        out.setdefault("v48_local_similarity_top1_column_name_match", False)
        out.setdefault("v48_local_similarity_top1_flowrate_diff", None)
        out.setdefault("v48_local_similarity_top1_temperature_diff", None)
        out.setdefault("v48_local_similarity_top1_column_length_diff", None)
        out.setdefault("v48_local_similarity_top1_gradient_time_diff", None)
        out.setdefault("v48_local_similarity_top2_method", "")
        out.setdefault("v48_local_similarity_top2_family", "")
        out.setdefault("v48_local_similarity_top2_score", None)
        out.setdefault("v48_local_similarity_score_margin", None)
        out.setdefault("v48_local_similarity_top_methods", [])
        out.setdefault("v48_local_similarity_top_families", [])
        out.setdefault("v48_template_name", "")
        out.setdefault("v48_template_method_code", "")
        out.setdefault("v48_template_match_status", "")
        out.setdefault("v48_public_safe_fallback", False)
        out.setdefault("v48_public_safe_fallback_reason", "")
        out.setdefault("v48_public_safe_fallback_manifest", "")
        return out

    @staticmethod
    def _attach_template_fields(out: Dict[str, Any], meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        template_name = _norm_str(meta_dict.get("template_name"))
        template_method_code = _normalize_method_code_hint(meta_dict.get("template_method_code"))
        template_match_status = _norm_str(meta_dict.get("template_match_status"))
        if template_name:
            out["v48_template_name"] = template_name
        if template_method_code:
            out["v48_template_method_code"] = template_method_code
        if template_match_status:
            out["v48_template_match_status"] = template_match_status
        return out

    def _predict_with_local_v47_router(
        self,
        smiles: str,
        meta_dict: Dict[str, Any],
        route: Dict[str, Any],
        route_reason_override: Optional[str] = None,
        method_code_hint: Optional[str] = None,
        metadata_path: Optional[str] = None,
        allow_method_hint: bool = True,
        predictor=None,
        model_label: str = "v47_local_expert_router",
        route_similarity: Optional[float] = None,
        soft_method_prior: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        fp = self.smiles_to_fp(smiles)
        if fp is None:
            return {
                "ok": False,
                "error": f"invalid_smiles: {smiles}",
                "v48_model": model_label,
                "router_top1": None,
                "router_gate_pass": False,
                "gate_strength": 0.0,
                "alpha": 0.0,
                "alpha_base": 0.0,
            }
        meta_vec = self._extract_method_meta_vector_cached(meta_dict)
        route_signature = self.extract_route_signature(
            meta_dict,
            method_dir=self._infer_method_dir(meta_dict, metadata_path, allow_method_hint=allow_method_hint),
            method_code=(method_code_hint if allow_method_hint else None),
        )
        active_predictor = predictor or self.v47_predictor
        out = active_predictor.predict(
            fp,
            meta_vec,
            method_code_hint=(method_code_hint if allow_method_hint else None),
            route_signature=route_signature,
            route_similarity=route_similarity,
            soft_method_prior=soft_method_prior,
        )
        if not isinstance(out, dict):
            out = {"pred_final": float(out)}
        out["ok"] = True
        out["v48_model"] = model_label
        out["router_top1"] = out.get("experts", [None])[0] if out.get("experts") else None
        return out

    @staticmethod
    def _should_use_public_safe_fallback(local_match: Dict[str, Any], local_method_code_hint: Optional[str]) -> bool:
        match_type = _norm_str(local_match.get("match_type"))
        match_source = _norm_str(local_match.get("match_source"))
        if match_type in {"explicit_method_code", "exact_method"}:
            return False
        matched_method = _normalize_method_code_hint(local_match.get("method_code")) or _normalize_method_code_hint(local_method_code_hint)
        if match_type == "similarity_method":
            return _is_public_expert_code(matched_method)
        if match_type == "condition_similarity_soft":
            if _is_public_expert_code(matched_method):
                return True
            score = _safe_float(local_match.get("similarity_top1_score"), default=float("nan"))
            return not np.isfinite(score) or score < LOCAL_TEMPLATE_AUTO_SCORE_THRESHOLD
        if "below_threshold" in match_source:
            return True
        return match_source in {"no_template_match", "route_signature_cosine_below_threshold", "meta_signature_collision_below_threshold"}

    @staticmethod
    def _soft_method_prior_from_local_match(local_match: Dict[str, Any]) -> Dict[str, float]:
        prior: Dict[str, float] = {}
        candidates = local_match.get("similarity_top_candidates")
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                code = _normalize_method_code_hint(item.get("method_code"))
                score = _safe_float(item.get("score"), default=float("nan"))
                if code and np.isfinite(score) and score > 0:
                    prior[code] = max(prior.get(code, 0.0), float(score))
        if not prior:
            code = _normalize_method_code_hint(local_match.get("similarity_top1_method"))
            score = _safe_float(local_match.get("similarity_top1_score"), default=float("nan"))
            if code and np.isfinite(score) and score > 0:
                prior[code] = float(score)
        return prior

    def _extract_local_method_code_hint(
        self,
        meta_dict: Dict[str, Any],
        metadata_path: Optional[str],
        allow_method_hint: bool = True,
    ) -> Optional[str]:
        if not allow_method_hint:
            return None
        code = _normalize_method_code_hint(meta_dict.get("method_code_hint")) or _normalize_method_code_hint(meta_dict.get("id"))
        if not code and metadata_path:
            base = os.path.basename(metadata_path)
            m = re.match(r"^(\d{4})_", base)
            if m:
                code = m.group(1)
            elif re.match(r"^\d{4}\.", base):
                code = os.path.splitext(base)[0]
        experts = getattr(self.v47_predictor, "manifest", {}).get("experts", {})
        if code and str(code) in experts:
            return code

        candidates = []
        meta_id = _norm_str(meta_dict.get("id")).lower()
        if meta_id:
            candidates.append(meta_id)
        if metadata_path:
            path_obj = os.path.normpath(metadata_path).replace("/", "\\").lower()
            parts = [part for part in path_obj.split("\\") if part]
            candidates.extend(parts)
            stem = os.path.splitext(os.path.basename(path_obj))[0]
            if stem:
                candidates.append(stem)
        for key in candidates:
            mapped = PUBLIC_BENCHMARK_METHOD_HINTS.get(key)
            if mapped and str(mapped) in experts:
                return mapped
        return None

    def _infer_method_dir(
        self,
        meta_dict: Dict[str, Any],
        metadata_path: Optional[str],
        allow_method_hint: bool = True,
    ) -> Optional[str]:
        if not allow_method_hint:
            return None
        if metadata_path and os.path.exists(metadata_path):
            parent = os.path.dirname(metadata_path)
            if os.path.basename(parent).isdigit():
                return parent
        code = self._extract_local_method_code_hint(meta_dict, metadata_path, allow_method_hint=allow_method_hint)
        if code:
            candidate = os.path.join(ROOT, "data", "processed_data", code)
            if os.path.isdir(candidate):
                return candidate
        return None

    def predict_one(
        self,
        smiles: str,
        meta_dict: Dict[str, Any],
        route_mode: str,
        metadata_path: Optional[str],
        input_path: Optional[str],
        allow_method_hint: bool = True,
    ) -> Dict[str, Any]:
        route = self.matcher.classify(
            meta_dict=meta_dict,
            mode=("mcmrt" if route_mode == "mcmrt_adapter" else route_mode),
            metadata_path=metadata_path,
            input_path=input_path,
        )
        if route["is_mcmrt"] and route_mode == "mcmrt_adapter":
            out = self.adapter_predictor.predict(smiles, meta_dict)
            if not isinstance(out, dict):
                out = {"pred_final": float(out)}
            out["ok"] = bool(out.get("ok", True))
            out["v48_model"] = "v47_mcmrt_adapter"
            out = self._attach_route_fields(
                out,
                route_reason=route["reason"],
                is_mcmrt=True,
                query_family_id=route["query_family_id"],
                best_mcmrt_method=route["best_method_name"],
                best_mcmrt_family_id=route["best_family_id"],
                best_mcmrt_l1=route["best_l1"],
                best_mcmrt_l2=route["best_l2"],
                selected_expert=f"mcmrt_{route['best_method_name']}" if route.get("best_method_name") else "mcmrt_adapter",
                route_family="mcmrt",
                fallback_used=True,
            )
            return self._attach_template_fields(out, meta_dict)
        elif route["is_mcmrt"]:
            out = self._predict_with_local_v47_router(
                smiles=smiles,
                meta_dict=meta_dict,
                route=route,
                route_reason_override="mcmrt_to_local_expert_router",
                method_code_hint=None,
                metadata_path=metadata_path,
            )
            out = self._attach_route_fields(
                out,
                route_reason="mcmrt_to_local_expert_router",
                is_mcmrt=True,
                query_family_id=route["query_family_id"],
                best_mcmrt_method=route["best_method_name"],
                best_mcmrt_family_id=route["best_family_id"],
                best_mcmrt_l1=route["best_l1"],
                best_mcmrt_l2=route["best_l2"],
                selected_expert=f"mcmrt_{route['best_method_name']}" if route.get("best_method_name") else "mcmrt_local_router",
                route_family="mcmrt",
                fallback_used=False,
                local_router_top1=out.get("router_top1"),
            )
            return self._attach_template_fields(out, meta_dict)
        else:
            local_method_code_hint = self._extract_local_method_code_hint(
                meta_dict,
                metadata_path,
                allow_method_hint=allow_method_hint,
            )
            local_match = {
                "match_type": "explicit_method_code" if local_method_code_hint else "no_hint",
                "match_source": "input_or_metadata_id" if local_method_code_hint else "no_template_match",
                "method_code": local_method_code_hint,
                "candidate_methods": ([local_method_code_hint] if local_method_code_hint else []),
                "candidate_families": [],
            }
            if local_method_code_hint is None:
                local_match = self.local_method_matcher.classify(meta_dict, metadata_path=metadata_path)
                local_method_code_hint = _normalize_method_code_hint(local_match.get("method_code"))
            if not local_method_code_hint and _should_promote_public_expert_similarity(
                local_match,
                metadata_path=metadata_path,
            ):
                local_method_code_hint = _normalize_method_code_hint(local_match.get("similarity_top1_method"))
                local_match = dict(local_match)
                local_match["match_type"] = "public_expert_similarity"
                local_match["match_source"] = "public_expert_similarity_top1"
                local_match["method_code"] = local_method_code_hint
                local_match["candidate_methods"] = [local_method_code_hint]
                if local_match.get("similarity_top1_family"):
                    local_match["candidate_families"] = [local_match.get("similarity_top1_family")]
            use_public_safe = bool(
                self.public_safe_predictor is not None
                and self._should_use_public_safe_fallback(local_match, local_method_code_hint)
            )
            active_predictor = self.public_safe_predictor if use_public_safe else self.v47_predictor
            model_label = "v47_public_safe_legacy_router" if use_public_safe else "v47_local_expert_router"
            match_type = _norm_str(local_match.get("match_type"))
            soft_condition_match = bool(
                self.condition_sim is not None
                and not use_public_safe
                and match_type == "condition_similarity_soft"
            )
            routed_method_code_hint = None if (use_public_safe or soft_condition_match) else local_method_code_hint
            soft_method_prior = self._soft_method_prior_from_local_match(local_match) if soft_condition_match else None
            # Soft, condition-weighted matches gate alpha by similarity instead of being
            # promoted to a hard "explicit method code" (alpha=1.0). Only applies when the
            # learned condition weights are loaded; otherwise behaviour is unchanged.
            route_similarity = None
            if (
                self.condition_sim is not None
                and not use_public_safe
                and match_type in {"condition_similarity_soft", "public_expert_similarity"}
            ):
                _sim = _safe_float(local_match.get("similarity_top1_score"), default=float("nan"))
                route_similarity = float(_sim) if np.isfinite(_sim) else None
            out = self._predict_with_local_v47_router(
                smiles=smiles,
                meta_dict=meta_dict,
                route=route,
                route_reason_override=("public_safe_fallback_to_legacy_v47" if use_public_safe else route["reason"]),
                method_code_hint=routed_method_code_hint,
                metadata_path=metadata_path,
                allow_method_hint=(allow_method_hint and not use_public_safe),
                predictor=active_predictor,
                model_label=model_label,
                route_similarity=route_similarity,
                soft_method_prior=soft_method_prior,
            )
            routed = self._attach_route_fields(
                out,
                route_reason=("public_safe_fallback_to_legacy_v47" if use_public_safe else route["reason"]),
                is_mcmrt=False,
                query_family_id=route["query_family_id"],
                best_mcmrt_method=route["best_method_name"],
                best_mcmrt_family_id=route["best_family_id"],
                best_mcmrt_l1=route["best_l1"],
                best_mcmrt_l2=route["best_l2"],
                selected_expert=out.get("router_top1") or local_method_code_hint or "global_only",
                route_family="local",
                fallback_used=use_public_safe,
                local_router_top1=out.get("router_top1"),
            )
            routed["v48_public_safe_fallback"] = bool(use_public_safe)
            routed["v48_public_safe_fallback_reason"] = local_match.get("match_source", "") if use_public_safe else ""
            routed["v48_public_safe_fallback_manifest"] = self.public_safe_fallback_manifest if use_public_safe else ""
            routed["v48_local_match_type"] = local_match.get("match_type", "")
            routed["v48_local_match_source"] = local_match.get("match_source", "")
            routed["v48_local_candidate_methods"] = local_match.get("candidate_methods", [])
            routed["v48_local_candidate_families"] = local_match.get("candidate_families", [])
            routed["v48_method_hint_used"] = routed_method_code_hint or ""
            routed["v48_local_similarity_top1_method"] = local_match.get("similarity_top1_method", "")
            routed["v48_local_similarity_top1_template_name"] = local_match.get("similarity_top1_template_name", "")
            routed["v48_local_similarity_top1_family"] = local_match.get("similarity_top1_family", "")
            routed["v48_local_similarity_top1_score"] = local_match.get("similarity_top1_score")
            routed["v48_local_similarity_top1_meta_l1"] = local_match.get("similarity_top1_meta_l1")
            routed["v48_local_similarity_top1_route_l1"] = local_match.get("similarity_top1_route_l1")
            routed["v48_local_similarity_top1_column_name_match"] = local_match.get("similarity_top1_column_name_match", False)
            routed["v48_local_similarity_top1_flowrate_diff"] = local_match.get("similarity_top1_flowrate_diff")
            routed["v48_local_similarity_top1_temperature_diff"] = local_match.get("similarity_top1_temperature_diff")
            routed["v48_local_similarity_top1_column_length_diff"] = local_match.get("similarity_top1_column_length_diff")
            routed["v48_local_similarity_top1_gradient_time_diff"] = local_match.get("similarity_top1_gradient_time_diff")
            routed["v48_local_similarity_top2_method"] = local_match.get("similarity_top2_method", "")
            routed["v48_local_similarity_top2_family"] = local_match.get("similarity_top2_family", "")
            routed["v48_local_similarity_top2_score"] = local_match.get("similarity_top2_score")
            routed["v48_local_similarity_score_margin"] = local_match.get("similarity_score_margin")
            routed["v48_local_similarity_top_methods"] = local_match.get("similarity_top_methods", [])
            routed["v48_local_similarity_top_families"] = local_match.get("similarity_top_families", [])
            if not use_public_safe and not routed.get("v48_template_name") and local_match.get("method_code"):
                routed["v48_template_name"] = local_match.get("similarity_top1_template_name", "")
                routed["v48_template_method_code"] = _normalize_method_code_hint(local_match.get("method_code"))
                routed["v48_template_match_status"] = "auto_template_top1"
            return self._attach_template_fields(routed, meta_dict)
 

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Unified v48 predictor: local v47 expert router by default, optional legacy MCMRT adapter.")
    parser.add_argument("--input", default=None, help="Input TSV, CSV, or Excel file.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name. Defaults to the first sheet.")
    parser.add_argument("--sep", default=None, help="Separator. Auto-detected from extension by default.")
    parser.add_argument("--metadata_path", default=None, help="Optional YAML, JSON, or MCMRT Dataset_xx.xlsx metadata file applied to all rows.")
    parser.add_argument("--route_mode", choices=["auto", "v47", "mcmrt", "mcmrt_adapter"], default="auto", help="Route selection mode.")
    parser.add_argument("--v47_train_script", default=DEFAULT_V47_TRAIN_SCRIPT, help="Path to the original v47 training script.")
    parser.add_argument("--v47_manifest", default=DEFAULT_V47_MANIFEST, help="Path to the original v47 manifest.")
    parser.add_argument("--public_safe_fallback_manifest", default=DEFAULT_PUBLIC_SAFE_FALLBACK_MANIFEST, help="Legacy/no-embedding v47 manifest used for unknown public-style inputs.")
    parser.add_argument("--disable_public_safe_fallback", action="store_true", help="Disable public-safe fallback for unknown methods and below-threshold template matches.")
    parser.add_argument("--condition_weights_path", default=DEFAULT_CONDITION_WEIGHTS, help="Condition-similarity weights JSON used for soft local-template routing.")
    parser.add_argument("--disable_condition_similarity", action="store_true", help="Disable condition-weighted similarity routing and use legacy raw-cosine template matching.")
    parser.add_argument("--adapter_train_script", default=DEFAULT_MCMRT_ADAPTER_TRAIN_SCRIPT, help="Path to the v47 MCMRT adapter training script.")
    parser.add_argument("--adapter_manifest", default=DEFAULT_MCMRT_ADAPTER_MANIFEST, help="Path to the v47 MCMRT adapter manifest.")
    parser.add_argument("--out_csv", default=None, help="Output CSV path.")
    parser.add_argument("--disable_method_hint", action="store_true", help="Ignore method_code / metadata_id hints and route only from chromatographic conditions.")
    parser.add_argument("--explode_multi_smiles", action="store_true", help="Split a multi-SMILES cell into multiple output rows.")
    parser.add_argument("--include_detail_json", action="store_true", help="Include verbose per-row detail JSON columns.")
    parser.add_argument("--simple_output", action="store_true", help="Write only input columns plus predicted_rt.")
    parser.add_argument("--enable_cross_condition_branch", action="store_true", help="Enable gated source-RT to target-condition transfer branch. The branch also auto-enables when source RT and source metadata are provided.")
    parser.add_argument("--disable_cross_condition_branch", action="store_true", help="Disable the gated source-RT to target-condition transfer branch.")
    parser.add_argument("--cross_condition_model", default=DEFAULT_CROSS_CONDITION_MODEL, help="Joblib model for the cross-condition transfer branch.")
    parser.add_argument("--cross_condition_source_metadata_path", default=None, help="Source-condition YAML/JSON metadata for source RT values.")
    parser.add_argument("--cross_condition_source_rt_col", default="source_rt_minutes", help="Column containing source-condition RT in minutes.")
    parser.add_argument("--cross_condition_max_predicted_error", type=float, default=2.0, help="Only apply cross-condition branch when estimated absolute error is at or below this threshold.")
    parser.add_argument("--enable_public_global_branch", action="store_true", help="Enable gated public-aware global branch for public-safe fallback rows. Kept for compatibility; enabled by default when the model exists.")
    parser.add_argument("--disable_public_global_branch", action="store_true", help="Disable the gated public-aware global branch.")
    parser.add_argument("--public_global_model", default=DEFAULT_PUBLIC_AWARE_GLOBAL_MODEL, help="Joblib model for the public-aware global branch.")
    parser.add_argument("--public_global_max_predicted_error", type=float, default=2.0, help="Only apply public-aware global branch when estimated absolute error is at or below this threshold.")
    parser.add_argument("--enable_family_condition_branch", action="store_true", help="Enable gated family-level condition-aware residual expert branch. Enabled by default when the model exists.")
    parser.add_argument("--disable_family_condition_branch", action="store_true", help="Disable the family-level condition-aware residual expert branch.")
    parser.add_argument("--family_condition_model", default=DEFAULT_FAMILY_CONDITION_EXPERT_MODEL, help="Joblib model for the family-level condition-aware residual expert branch.")
    parser.add_argument("--family_condition_max_neighbor_distance", type=float, default=None, help="Optional override for the family expert condition-neighbor distance guard.")
    parser.add_argument("--public_expert_route_min_rows", type=int, default=PUBLIC_EXPERT_ROUTE_MIN_ROWS_DEFAULT, help="Protect exact public-expert routes with at least this many training rows from public fallback overrides.")
    parser.add_argument("--calibration_adapter", default=None, help="Optional lightweight calibration adapter.joblib applied after prediction.")
    parser.add_argument("--calibration_adapter_mode", choices=["auto", "blend", "report", "force"], default="auto", help="Adapter apply mode. report adds columns without changing final predictions.")
    parser.add_argument("--calibration_adapter_final_col", default=None, help="Prediction column updated by the adapter. Defaults to pred_rt_v48_mean or pred_rt_v48.")
    args = parser.parse_args(argv)

    input_path = args.input or _guess_default_input()
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(
            "Input file not found. Use --input or place an integrated_compound_data_*.tsv file under LCQ."
        )

    df = _load_table(input_path, args.sep, args.sheet)
    if "compound_id" in df.columns:
        df = df[df["compound_id"].astype(str).str.lower() != "compound_id"].copy()
    if "smiles" in df.columns:
        df = df[df["smiles"].astype(str).str.lower() != "smiles"].copy()
    df = df.reset_index(drop=True)

    if "smiles" not in df.columns:
        raise RuntimeError("Input file is missing the smiles column.")

    predictor = UnifiedV48Predictor(
        v47_train_script=args.v47_train_script,
        v47_manifest=args.v47_manifest,
        adapter_train_script=args.adapter_train_script,
        adapter_manifest=args.adapter_manifest,
        v47_dataset_script=DEFAULT_V47_DATASET_SCRIPT,
        mcmrt_split_script=DEFAULT_MCMRT_SPLIT_SCRIPT,
        mcmrt_compare_script=DEFAULT_MCMRT_COMPARE_SCRIPT,
        public_safe_fallback_manifest=args.public_safe_fallback_manifest,
        public_safe_fallback=(not args.disable_public_safe_fallback),
        condition_weights_path=args.condition_weights_path,
        use_condition_similarity=(not args.disable_condition_similarity),
    )

    cross_condition_predictor = None
    cross_condition_auto = bool(
        (not args.disable_cross_condition_branch)
        and args.metadata_path
        and args.cross_condition_source_metadata_path
        and args.cross_condition_source_rt_col in df.columns
    )
    cross_condition_requested = bool(args.enable_cross_condition_branch or cross_condition_auto)
    if args.disable_cross_condition_branch:
        cross_condition_requested = False
    if cross_condition_requested:
        missing = []
        if not args.metadata_path:
            missing.append("--metadata_path")
        if not args.cross_condition_source_metadata_path:
            missing.append("--cross_condition_source_metadata_path")
        if not os.path.exists(args.cross_condition_model):
            missing.append(f"cross-condition model not found: {args.cross_condition_model}")
        if args.cross_condition_source_metadata_path and not os.path.exists(args.cross_condition_source_metadata_path):
            missing.append(f"cross-condition source metadata not found: {args.cross_condition_source_metadata_path}")
        if missing:
            if args.enable_cross_condition_branch:
                raise RuntimeError("--enable_cross_condition_branch cannot run: " + "; ".join(missing))
            print("[WARN] cross-condition branch skipped: " + "; ".join(missing))
        else:
            from core.cross_condition.transfer_model import CrossConditionTransferPredictor

            cross_condition_predictor = CrossConditionTransferPredictor(args.cross_condition_model)
    family_condition_predictor = None
    family_condition_requested = bool(args.enable_family_condition_branch or not args.disable_family_condition_branch)
    if family_condition_requested:
        family_condition_explicit = bool(
            args.enable_family_condition_branch
            or os.path.abspath(args.family_condition_model) != os.path.abspath(DEFAULT_FAMILY_CONDITION_EXPERT_MODEL)
        )
        if not os.path.exists(args.family_condition_model):
            if family_condition_explicit:
                raise FileNotFoundError(f"Family condition expert model not found: {args.family_condition_model}")
            print(f"[WARN] family-condition expert branch skipped; model not found: {args.family_condition_model}")
        else:
            from core.family_condition.family_condition_model import FamilyConditionExpertPredictor

            family_condition_predictor = FamilyConditionExpertPredictor(
                args.family_condition_model,
                global_bundle=predictor.v47_predictor.global_bundle,
                dataset_mod=predictor.dataset_mod,
            )

    public_global_predictor = None
    public_global_requested = bool(args.enable_public_global_branch or not args.disable_public_global_branch)
    if public_global_requested:
        public_global_explicit = bool(
            args.enable_public_global_branch
            or os.path.abspath(args.public_global_model) != os.path.abspath(DEFAULT_PUBLIC_AWARE_GLOBAL_MODEL)
        )
        if not os.path.exists(args.public_global_model):
            if public_global_explicit:
                raise FileNotFoundError(f"Public-aware global model not found: {args.public_global_model}")
            print(f"[WARN] public-aware global branch skipped; model not found: {args.public_global_model}")
        else:
            from core.public_global.public_global_model import PublicAwareGlobalPredictor

            public_global_predictor = PublicAwareGlobalPredictor(args.public_global_model)

    shared_meta = None
    if args.metadata_path:
        shared_meta = load_metadata_any(args.metadata_path, predictor.v47_train_mod, predictor.mcmrt_compare_mod)
    elif not _has_any_meta_columns(df):
        raise RuntimeError(
            "Input file has no chromatographic condition columns. "
            "Provide --metadata_path <yaml/json/xlsx> to apply one method setup to all rows."
        )
    cross_condition_source_meta = None
    if cross_condition_predictor is not None:
        cross_condition_source_meta = load_metadata_any(
            args.cross_condition_source_metadata_path,
            predictor.v47_train_mod,
            predictor.mcmrt_compare_mod,
        )

    rows_out: List[Dict[str, Any]] = []
    total_rows = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        base_row = row.to_dict()
        meta_dict = shared_meta if shared_meta is not None else build_meta_dict_from_row(row)
        smiles_raw = _norm_str(row.get("smiles"))
        smiles_list = _split_smiles(smiles_raw)

        preds = []
        models = []
        details = []
        for smi in smiles_list:
            out = predictor.predict_one(
                smiles=smi,
                meta_dict=meta_dict,
                route_mode=args.route_mode,
                metadata_path=args.metadata_path,
                input_path=input_path,
                allow_method_hint=(not args.disable_method_hint),
            )
            details.append({"smiles": smi, **out})
            if out.get("ok") and out.get("pred_final") is not None:
                preds.append(float(out["pred_final"]))
                models.append(str(out.get("v48_model", "")))

            if args.explode_multi_smiles:
                row_out = dict(base_row)
                row_out["smiles"] = smi
                row_out["pred_rt_v48"] = float(out["pred_final"]) if out.get("pred_final") is not None else np.nan
                row_out["pred_rt_v48_model"] = out.get("v48_model", "")
                row_out["pred_rt_v48_route_reason"] = out.get("v48_route_reason", "")
                row_out["pred_rt_v48_is_mcmrt"] = bool(out.get("v48_is_mcmrt", False))
                row_out["pred_rt_v48_selected_expert"] = out.get("v48_selected_expert", "")
                row_out["pred_rt_v48_route_family"] = out.get("v48_route_family", "")
                row_out["pred_rt_v48_fallback_used"] = bool(out.get("v48_fallback_used", False))
                row_out["pred_rt_v48_router_top1"] = out.get("v48_local_router_top1", out.get("router_top1", out.get("pred_routed_top1")))
                row_out["pred_rt_v48_router_gate_pass"] = bool(out.get("v48_router_gate_pass", False))
                row_out["pred_rt_v48_gate_strength"] = out.get("v48_gate_strength")
                row_out["pred_rt_v48_alpha"] = out.get("v48_alpha")
                row_out["pred_rt_v48_alpha_base"] = out.get("v48_alpha_base")
                row_out["pred_rt_v48_confidence_tier"] = out.get("v48_confidence_tier", "")
                row_out["pred_rt_v48_family_confidence"] = out.get("v48_family_confidence")
                row_out["pred_rt_v48_correction_cap"] = out.get("v48_correction_cap")
                row_out["pred_rt_v48_family_top1"] = out.get("v48_family_top1")
                row_out["pred_rt_v48_family_margin"] = out.get("v48_family_margin")
                row_out["pred_rt_v48_top_expert_weight"] = out.get("v48_top_expert_weight")
                row_out["pred_rt_v48_expert_margin"] = out.get("v48_expert_margin")
                row_out["pred_rt_v48_local_match_type"] = out.get("v48_local_match_type", "")
                row_out["pred_rt_v48_local_match_source"] = out.get("v48_local_match_source", "")
                row_out["pred_rt_v48_local_candidate_methods"] = ";".join(out.get("v48_local_candidate_methods", []))
                row_out["pred_rt_v48_local_candidate_families"] = ";".join(out.get("v48_local_candidate_families", []))
                row_out["pred_rt_v48_method_hint_used"] = out.get("v48_method_hint_used", "")
                row_out["pred_rt_v48_collision_candidates"] = ";".join(out.get("v48_collision_candidates", []))
                row_out["pred_rt_v48_local_similarity_top1_method"] = out.get("v48_local_similarity_top1_method", "")
                row_out["pred_rt_v48_local_similarity_top1_template_name"] = out.get("v48_local_similarity_top1_template_name", "")
                row_out["pred_rt_v48_local_similarity_top1_family"] = out.get("v48_local_similarity_top1_family", "")
                row_out["pred_rt_v48_local_similarity_top1_score"] = out.get("v48_local_similarity_top1_score")
                row_out["pred_rt_v48_local_similarity_top1_meta_l1"] = out.get("v48_local_similarity_top1_meta_l1")
                row_out["pred_rt_v48_local_similarity_top1_route_l1"] = out.get("v48_local_similarity_top1_route_l1")
                row_out["pred_rt_v48_local_similarity_top1_column_name_match"] = out.get("v48_local_similarity_top1_column_name_match")
                row_out["pred_rt_v48_local_similarity_top1_flowrate_diff"] = out.get("v48_local_similarity_top1_flowrate_diff")
                row_out["pred_rt_v48_local_similarity_top1_temperature_diff"] = out.get("v48_local_similarity_top1_temperature_diff")
                row_out["pred_rt_v48_local_similarity_top1_column_length_diff"] = out.get("v48_local_similarity_top1_column_length_diff")
                row_out["pred_rt_v48_local_similarity_top1_gradient_time_diff"] = out.get("v48_local_similarity_top1_gradient_time_diff")
                row_out["pred_rt_v48_local_similarity_top2_method"] = out.get("v48_local_similarity_top2_method", "")
                row_out["pred_rt_v48_local_similarity_top2_family"] = out.get("v48_local_similarity_top2_family", "")
                row_out["pred_rt_v48_local_similarity_top2_score"] = out.get("v48_local_similarity_top2_score")
                row_out["pred_rt_v48_local_similarity_score_margin"] = out.get("v48_local_similarity_score_margin")
                row_out["pred_rt_v48_local_similarity_top_methods"] = ";".join(out.get("v48_local_similarity_top_methods", []))
                row_out["pred_rt_v48_local_similarity_top_families"] = ";".join(out.get("v48_local_similarity_top_families", []))
                row_out["pred_rt_v48_template_name"] = out.get("v48_template_name", "")
                row_out["pred_rt_v48_template_method_code"] = out.get("v48_template_method_code", "")
                row_out["pred_rt_v48_template_match_status"] = out.get("v48_template_match_status", "")
                row_out["pred_rt_v48_public_safe_fallback"] = bool(out.get("v48_public_safe_fallback", False))
                row_out["pred_rt_v48_public_safe_fallback_reason"] = out.get("v48_public_safe_fallback_reason", "")
                row_out["pred_rt_v48_public_safe_fallback_manifest"] = out.get("v48_public_safe_fallback_manifest", "")
                protected_public_expert_route = _is_protected_public_expert_route(
                    out,
                    min_rows=args.public_expert_route_min_rows,
                )
                protected_known_method_route = _is_protected_known_method_route(out)
                row_out["pred_rt_v48_public_expert_route_protected"] = protected_public_expert_route
                row_out["pred_rt_v48_known_method_route_protected"] = protected_known_method_route
                if cross_condition_predictor is not None:
                    row_out["pred_rt_v48_cross_condition"] = np.nan
                    row_out["pred_rt_v48_cross_condition_error_est"] = np.nan
                    row_out["pred_rt_v48_cross_condition_accepted"] = False
                    row_out["pred_rt_v48_cross_condition_used"] = False
                    row_out["pred_rt_v48_cross_condition_admitted"] = _should_admit_cross_condition_branch(
                        public_safe_fallback=bool(out.get("v48_public_safe_fallback", False)),
                        meta_dict=meta_dict,
                        metadata_path=args.metadata_path,
                        input_path=input_path,
                        protected_public_expert_route=protected_public_expert_route,
                        protected_known_method_route=protected_known_method_route,
                    )
                    row_out["pred_rt_v48_cross_condition_threshold"] = args.cross_condition_max_predicted_error
                    row_out["pred_rt_v48_cross_condition_model"] = ""
                    source_rt = _safe_float(row.get(args.cross_condition_source_rt_col), default=np.nan)
                    if row_out["pred_rt_v48_cross_condition_admitted"] and np.isfinite(source_rt):
                        cc_input = pd.DataFrame(
                            [
                                {
                                    "smiles": smi,
                                    args.cross_condition_source_rt_col: source_rt,
                                }
                            ]
                        )
                        cc_pred = cross_condition_predictor.predict_frame(
                            cc_input,
                            source_rt_col=args.cross_condition_source_rt_col,
                            source_metadata=cross_condition_source_meta,
                            target_metadata=meta_dict,
                            smiles_col="smiles",
                            max_predicted_error=args.cross_condition_max_predicted_error,
                        ).iloc[0]
                        cc_value = float(cc_pred.get("pred_rt_cross_condition", np.nan))
                        cc_error = float(cc_pred.get("pred_rt_cross_condition_error_est", np.nan))
                        cc_accepted = bool(cc_pred.get("pred_rt_cross_condition_accepted", False))
                        row_out["pred_rt_v48_cross_condition"] = cc_value
                        row_out["pred_rt_v48_cross_condition_error_est"] = cc_error
                        row_out["pred_rt_v48_cross_condition_accepted"] = cc_accepted
                        row_out["pred_rt_v48_cross_condition_model"] = cc_pred.get("pred_rt_cross_condition_model", "")
                        if cc_accepted and np.isfinite(cc_value):
                            row_out["pred_rt_v48"] = cc_value
                            row_out["pred_rt_v48_cross_condition_used"] = True
                            row_out["pred_rt_v48_model"] = "cross_condition_transfer"
                            row_out["pred_rt_v48_route_reason"] = "cross_condition_transfer_branch"
                            row_out["pred_rt_v48_selected_expert"] = "cross_condition_transfer"
                if family_condition_predictor is not None:
                    row_out["pred_rt_v48_family_condition"] = np.nan
                    row_out["pred_rt_v48_family_condition_global"] = np.nan
                    row_out["pred_rt_v48_family_condition_delta"] = np.nan
                    row_out["pred_rt_v48_family_condition_error_est"] = np.nan
                    row_out["pred_rt_v48_family_condition_accepted"] = False
                    row_out["pred_rt_v48_family_condition_used"] = False
                    row_out["pred_rt_v48_family_condition_model"] = args.family_condition_model
                    row_out["pred_rt_v48_family_condition_admitted"] = _should_admit_family_condition_branch(
                        public_safe_fallback=bool(out.get("v48_public_safe_fallback", False)),
                        meta_dict=meta_dict,
                        metadata_path=args.metadata_path,
                        input_path=input_path,
                        protected_public_expert_route=protected_public_expert_route,
                    )
                    row_out["pred_rt_v48_family_condition_family_id"] = ""
                    row_out["pred_rt_v48_family_condition_status"] = ""
                    row_out["pred_rt_v48_family_condition_blend_weight"] = 0.0
                    row_out["pred_rt_v48_family_condition_neighbor_distance"] = np.nan
                    row_out["pred_rt_v48_family_condition_neighbor_threshold"] = np.nan
                    row_out["pred_rt_v48_family_condition_train_methods"] = 0
                    row_out["pred_rt_v48_family_condition_train_rows"] = 0
                    if (
                        row_out["pred_rt_v48_family_condition_admitted"]
                        and not bool(row_out.get("pred_rt_v48_cross_condition_used", False))
                    ):
                        fc_input = pd.DataFrame([{"smiles": smi}])
                        fc_pred = family_condition_predictor.predict_frame(
                            fc_input,
                            metadata_dict=meta_dict,
                            smiles_col="smiles",
                            max_neighbor_distance=args.family_condition_max_neighbor_distance,
                        ).iloc[0]
                        fc_value = float(fc_pred.get("pred_rt_family_condition", np.nan))
                        fc_accepted = bool(fc_pred.get("pred_rt_family_condition_accepted", False))
                        row_out["pred_rt_v48_family_condition"] = fc_value
                        row_out["pred_rt_v48_family_condition_global"] = float(fc_pred.get("pred_rt_family_condition_global", np.nan))
                        row_out["pred_rt_v48_family_condition_delta"] = float(fc_pred.get("pred_rt_family_condition_delta", np.nan))
                        row_out["pred_rt_v48_family_condition_error_est"] = float(fc_pred.get("pred_rt_family_condition_error_est", np.nan))
                        row_out["pred_rt_v48_family_condition_accepted"] = fc_accepted
                        row_out["pred_rt_v48_family_condition_family_id"] = fc_pred.get("pred_rt_family_condition_family_id", "")
                        row_out["pred_rt_v48_family_condition_status"] = fc_pred.get("pred_rt_family_condition_status", "")
                        row_out["pred_rt_v48_family_condition_blend_weight"] = fc_pred.get("pred_rt_family_condition_blend_weight")
                        row_out["pred_rt_v48_family_condition_neighbor_distance"] = fc_pred.get("pred_rt_family_condition_neighbor_distance")
                        row_out["pred_rt_v48_family_condition_neighbor_threshold"] = fc_pred.get("pred_rt_family_condition_neighbor_threshold")
                        row_out["pred_rt_v48_family_condition_train_methods"] = fc_pred.get("pred_rt_family_condition_train_methods")
                        row_out["pred_rt_v48_family_condition_train_rows"] = fc_pred.get("pred_rt_family_condition_train_rows")
                        if fc_accepted and np.isfinite(fc_value):
                            row_out["pred_rt_v48"] = fc_value
                            row_out["pred_rt_v48_family_condition_used"] = True
                            row_out["pred_rt_v48_model"] = "family_condition_aware_expert"
                            row_out["pred_rt_v48_route_reason"] = "family_condition_aware_expert_branch"
                            row_out["pred_rt_v48_selected_expert"] = f"family_condition:{row_out['pred_rt_v48_family_condition_family_id']}"
                if public_global_predictor is not None:
                    row_out["pred_rt_v48_public_global"] = np.nan
                    row_out["pred_rt_v48_public_global_error_est"] = np.nan
                    row_out["pred_rt_v48_public_global_accepted"] = False
                    row_out["pred_rt_v48_public_global_used"] = False
                    row_out["pred_rt_v48_public_global_threshold"] = args.public_global_max_predicted_error
                    row_out["pred_rt_v48_public_global_model"] = args.public_global_model
                    row_out["pred_rt_v48_public_global_admitted"] = _should_admit_public_global_branch(
                        public_safe_fallback=bool(out.get("v48_public_safe_fallback", False)),
                        meta_dict=meta_dict,
                        metadata_path=args.metadata_path,
                        input_path=input_path,
                        protected_public_expert_route=protected_public_expert_route,
                    )
                    if (
                        row_out["pred_rt_v48_public_global_admitted"]
                        and not bool(row_out.get("pred_rt_v48_cross_condition_used", False))
                        and not bool(row_out.get("pred_rt_v48_family_condition_used", False))
                    ):
                        pg_input = pd.DataFrame([{"smiles": smi}])
                        pg_pred = public_global_predictor.predict_frame(
                            pg_input,
                            metadata_dict=meta_dict,
                            smiles_col="smiles",
                            max_predicted_error=args.public_global_max_predicted_error,
                        ).iloc[0]
                        pg_value = float(pg_pred.get("pred_rt_public_global", np.nan))
                        pg_error = float(pg_pred.get("pred_rt_public_global_error_est", np.nan))
                        pg_accepted = bool(pg_pred.get("pred_rt_public_global_accepted", False))
                        row_out["pred_rt_v48_public_global"] = pg_value
                        row_out["pred_rt_v48_public_global_error_est"] = pg_error
                        row_out["pred_rt_v48_public_global_accepted"] = pg_accepted
                        if pg_accepted and np.isfinite(pg_value):
                            row_out["pred_rt_v48"] = pg_value
                            row_out["pred_rt_v48_public_global_used"] = True
                            row_out["pred_rt_v48_model"] = "public_aware_global"
                            row_out["pred_rt_v48_route_reason"] = "public_aware_global_branch"
                            row_out["pred_rt_v48_selected_expert"] = "public_aware_global"
                if bool(row_out.get("pred_rt_v48_cross_condition_used", False)):
                    row_out["pred_rt_v48_strategy"] = "cross_condition_transfer"
                elif bool(row_out.get("pred_rt_v48_family_condition_used", False)):
                    row_out["pred_rt_v48_strategy"] = "family_condition_aware_expert"
                elif bool(row_out.get("pred_rt_v48_public_global_used", False)):
                    row_out["pred_rt_v48_strategy"] = "public_aware_global"
                elif bool(row_out.get("pred_rt_v48_public_safe_fallback", False)):
                    row_out["pred_rt_v48_strategy"] = "legacy_public_safe_fallback"
                elif bool(row_out.get("pred_rt_v48_public_expert_route_protected", False)):
                    row_out["pred_rt_v48_strategy"] = "public_expert"
                else:
                    row_out["pred_rt_v48_strategy"] = "rt_predict"
                row_out["pred_rt_v48_ok"] = bool(out.get("ok", False))
                if args.include_detail_json:
                    row_out["pred_rt_v48_detail_json"] = json.dumps(out, ensure_ascii=False)
                rows_out.append(row_out)

        if not args.explode_multi_smiles:
            row_out = dict(base_row)
            row_out["n_smiles"] = len(smiles_list)
            row_out["pred_rt_v48_first"] = preds[0] if preds else np.nan
            row_out["pred_rt_v48_mean"] = float(np.mean(preds)) if preds else np.nan
            row_out["pred_rt_v48_base_mean"] = row_out["pred_rt_v48_mean"]
            row_out["pred_rt_v48_all"] = ";".join(f"{x:.6f}" for x in preds) if preds else ""
            row_out["pred_rt_v48_model_first"] = models[0] if models else ""
            row_out["pred_rt_v48_models"] = ";".join(models)
            row_out["pred_rt_v48_route_reason"] = details[0].get("v48_route_reason", "") if details else ""
            row_out["pred_rt_v48_is_mcmrt"] = bool(details[0].get("v48_is_mcmrt", False)) if details else False
            row_out["pred_rt_v48_selected_expert"] = details[0].get("v48_selected_expert", "") if details else ""
            row_out["pred_rt_v48_route_family"] = details[0].get("v48_route_family", "") if details else ""
            row_out["pred_rt_v48_fallback_used"] = bool(details[0].get("v48_fallback_used", False)) if details else False
            row_out["pred_rt_v48_router_top1"] = details[0].get("v48_local_router_top1", "") if details else ""
            row_out["pred_rt_v48_router_gate_pass"] = bool(details[0].get("v48_router_gate_pass", False)) if details else False
            row_out["pred_rt_v48_gate_strength"] = details[0].get("v48_gate_strength") if details else None
            row_out["pred_rt_v48_alpha"] = details[0].get("v48_alpha") if details else None
            row_out["pred_rt_v48_alpha_base"] = details[0].get("v48_alpha_base") if details else None
            row_out["pred_rt_v48_confidence_tier"] = details[0].get("v48_confidence_tier", "") if details else ""
            row_out["pred_rt_v48_family_confidence"] = details[0].get("v48_family_confidence") if details else None
            row_out["pred_rt_v48_correction_cap"] = details[0].get("v48_correction_cap") if details else None
            row_out["pred_rt_v48_family_top1"] = details[0].get("v48_family_top1") if details else None
            row_out["pred_rt_v48_family_margin"] = details[0].get("v48_family_margin") if details else None
            row_out["pred_rt_v48_top_expert_weight"] = details[0].get("v48_top_expert_weight") if details else None
            row_out["pred_rt_v48_expert_margin"] = details[0].get("v48_expert_margin") if details else None
            row_out["pred_rt_v48_local_match_type"] = details[0].get("v48_local_match_type", "") if details else ""
            row_out["pred_rt_v48_local_match_source"] = details[0].get("v48_local_match_source", "") if details else ""
            row_out["pred_rt_v48_local_candidate_methods"] = ";".join(details[0].get("v48_local_candidate_methods", [])) if details else ""
            row_out["pred_rt_v48_local_candidate_families"] = ";".join(details[0].get("v48_local_candidate_families", [])) if details else ""
            row_out["pred_rt_v48_method_hint_used"] = details[0].get("v48_method_hint_used", "") if details else ""
            row_out["pred_rt_v48_collision_candidates"] = ";".join(details[0].get("v48_collision_candidates", [])) if details else ""
            row_out["pred_rt_v48_local_similarity_top1_method"] = details[0].get("v48_local_similarity_top1_method", "") if details else ""
            row_out["pred_rt_v48_local_similarity_top1_template_name"] = details[0].get("v48_local_similarity_top1_template_name", "") if details else ""
            row_out["pred_rt_v48_local_similarity_top1_family"] = details[0].get("v48_local_similarity_top1_family", "") if details else ""
            row_out["pred_rt_v48_local_similarity_top1_score"] = details[0].get("v48_local_similarity_top1_score") if details else None
            row_out["pred_rt_v48_local_similarity_top1_meta_l1"] = details[0].get("v48_local_similarity_top1_meta_l1") if details else None
            row_out["pred_rt_v48_local_similarity_top1_route_l1"] = details[0].get("v48_local_similarity_top1_route_l1") if details else None
            row_out["pred_rt_v48_local_similarity_top1_column_name_match"] = details[0].get("v48_local_similarity_top1_column_name_match") if details else None
            row_out["pred_rt_v48_local_similarity_top1_flowrate_diff"] = details[0].get("v48_local_similarity_top1_flowrate_diff") if details else None
            row_out["pred_rt_v48_local_similarity_top1_temperature_diff"] = details[0].get("v48_local_similarity_top1_temperature_diff") if details else None
            row_out["pred_rt_v48_local_similarity_top1_column_length_diff"] = details[0].get("v48_local_similarity_top1_column_length_diff") if details else None
            row_out["pred_rt_v48_local_similarity_top1_gradient_time_diff"] = details[0].get("v48_local_similarity_top1_gradient_time_diff") if details else None
            row_out["pred_rt_v48_local_similarity_top2_method"] = details[0].get("v48_local_similarity_top2_method", "") if details else ""
            row_out["pred_rt_v48_local_similarity_top2_family"] = details[0].get("v48_local_similarity_top2_family", "") if details else ""
            row_out["pred_rt_v48_local_similarity_top2_score"] = details[0].get("v48_local_similarity_top2_score") if details else None
            row_out["pred_rt_v48_local_similarity_score_margin"] = details[0].get("v48_local_similarity_score_margin") if details else None
            row_out["pred_rt_v48_local_similarity_top_methods"] = ";".join(details[0].get("v48_local_similarity_top_methods", [])) if details else ""
            row_out["pred_rt_v48_local_similarity_top_families"] = ";".join(details[0].get("v48_local_similarity_top_families", [])) if details else ""
            row_out["pred_rt_v48_template_name"] = details[0].get("v48_template_name", "") if details else ""
            row_out["pred_rt_v48_template_method_code"] = details[0].get("v48_template_method_code", "") if details else ""
            row_out["pred_rt_v48_template_match_status"] = details[0].get("v48_template_match_status", "") if details else ""
            row_out["pred_rt_v48_public_safe_fallback"] = bool(details[0].get("v48_public_safe_fallback", False)) if details else False
            row_out["pred_rt_v48_public_safe_fallback_reason"] = details[0].get("v48_public_safe_fallback_reason", "") if details else ""
            row_out["pred_rt_v48_public_safe_fallback_manifest"] = details[0].get("v48_public_safe_fallback_manifest", "") if details else ""
            protected_public_expert_route = bool(
                details
                and _is_protected_public_expert_route(
                    details[0],
                    min_rows=args.public_expert_route_min_rows,
                )
            )
            protected_known_method_route = bool(
                details and _is_protected_known_method_route(details[0])
            )
            row_out["pred_rt_v48_public_expert_route_protected"] = protected_public_expert_route
            row_out["pred_rt_v48_known_method_route_protected"] = protected_known_method_route
            if cross_condition_predictor is not None:
                row_out["pred_rt_v48_cross_condition"] = np.nan
                row_out["pred_rt_v48_cross_condition_error_est"] = np.nan
                row_out["pred_rt_v48_cross_condition_accepted"] = False
                row_out["pred_rt_v48_cross_condition_used"] = False
                row_out["pred_rt_v48_cross_condition_admitted"] = _should_admit_cross_condition_branch(
                    public_safe_fallback=bool(details and details[0].get("v48_public_safe_fallback", False)),
                    meta_dict=meta_dict,
                    metadata_path=args.metadata_path,
                    input_path=input_path,
                    protected_public_expert_route=protected_public_expert_route,
                    protected_known_method_route=protected_known_method_route,
                )
                row_out["pred_rt_v48_cross_condition_threshold"] = args.cross_condition_max_predicted_error
                row_out["pred_rt_v48_cross_condition_model"] = ""
                source_rt_raw = row.get(args.cross_condition_source_rt_col)
                source_rt = _safe_float(source_rt_raw, default=np.nan)
                if row_out["pred_rt_v48_cross_condition_admitted"] and smiles_list and np.isfinite(source_rt):
                    cc_input = pd.DataFrame(
                        [
                            {
                                "smiles": smi,
                                args.cross_condition_source_rt_col: source_rt,
                            }
                            for smi in smiles_list
                        ]
                    )
                    cc_frame = cross_condition_predictor.predict_frame(
                        cc_input,
                        source_rt_col=args.cross_condition_source_rt_col,
                        source_metadata=cross_condition_source_meta,
                        target_metadata=meta_dict,
                        smiles_col="smiles",
                        max_predicted_error=args.cross_condition_max_predicted_error,
                    )
                    cc_values = pd.to_numeric(cc_frame["pred_rt_cross_condition"], errors="coerce").to_numpy(dtype=float)
                    cc_errors = pd.to_numeric(cc_frame["pred_rt_cross_condition_error_est"], errors="coerce").to_numpy(dtype=float)
                    cc_accepted_values = cc_frame["pred_rt_cross_condition_accepted"].astype(bool).to_numpy()
                    cc_all_accepted = bool(
                        len(cc_values) == len(smiles_list)
                        and np.all(np.isfinite(cc_values))
                        and np.all(cc_accepted_values)
                    )
                    row_out["pred_rt_v48_cross_condition"] = float(np.mean(cc_values)) if len(cc_values) else np.nan
                    row_out["pred_rt_v48_cross_condition_error_est"] = float(np.nanmax(cc_errors)) if len(cc_errors) else np.nan
                    row_out["pred_rt_v48_cross_condition_accepted"] = cc_all_accepted
                    row_out["pred_rt_v48_cross_condition_model"] = cc_frame["pred_rt_cross_condition_model"].iloc[0] if len(cc_frame) else ""
                    if cc_all_accepted:
                        row_out["pred_rt_v48_mean"] = float(np.mean(cc_values))
                        row_out["pred_rt_v48_first"] = float(cc_values[0])
                        row_out["pred_rt_v48_all"] = ";".join(f"{x:.6f}" for x in cc_values)
                        row_out["pred_rt_v48_cross_condition_used"] = True
                        row_out["pred_rt_v48_model_first"] = "cross_condition_transfer"
                        row_out["pred_rt_v48_models"] = ";".join(["cross_condition_transfer"] * len(cc_values))
                        row_out["pred_rt_v48_route_reason"] = "cross_condition_transfer_branch"
                        row_out["pred_rt_v48_selected_expert"] = "cross_condition_transfer"
            if family_condition_predictor is not None:
                row_out["pred_rt_v48_family_condition"] = np.nan
                row_out["pred_rt_v48_family_condition_global"] = np.nan
                row_out["pred_rt_v48_family_condition_delta"] = np.nan
                row_out["pred_rt_v48_family_condition_error_est"] = np.nan
                row_out["pred_rt_v48_family_condition_accepted"] = False
                row_out["pred_rt_v48_family_condition_used"] = False
                row_out["pred_rt_v48_family_condition_model"] = args.family_condition_model
                row_out["pred_rt_v48_family_condition_admitted"] = _should_admit_family_condition_branch(
                    public_safe_fallback=bool(details and details[0].get("v48_public_safe_fallback", False)),
                    meta_dict=meta_dict,
                    metadata_path=args.metadata_path,
                    input_path=input_path,
                    protected_public_expert_route=protected_public_expert_route,
                )
                row_out["pred_rt_v48_family_condition_family_id"] = ""
                row_out["pred_rt_v48_family_condition_status"] = ""
                row_out["pred_rt_v48_family_condition_blend_weight"] = 0.0
                row_out["pred_rt_v48_family_condition_neighbor_distance"] = np.nan
                row_out["pred_rt_v48_family_condition_neighbor_threshold"] = np.nan
                row_out["pred_rt_v48_family_condition_train_methods"] = 0
                row_out["pred_rt_v48_family_condition_train_rows"] = 0
                if (
                    row_out["pred_rt_v48_family_condition_admitted"]
                    and not bool(row_out.get("pred_rt_v48_cross_condition_used", False))
                    and smiles_list
                ):
                    fc_input = pd.DataFrame({"smiles": smiles_list})
                    fc_frame = family_condition_predictor.predict_frame(
                        fc_input,
                        metadata_dict=meta_dict,
                        smiles_col="smiles",
                        max_neighbor_distance=args.family_condition_max_neighbor_distance,
                    )
                    fc_values = pd.to_numeric(fc_frame["pred_rt_family_condition"], errors="coerce").to_numpy(dtype=float)
                    fc_accepted_values = fc_frame["pred_rt_family_condition_accepted"].astype(bool).to_numpy()
                    fc_all_accepted = bool(
                        len(fc_values) == len(smiles_list)
                        and np.all(np.isfinite(fc_values))
                        and np.all(fc_accepted_values)
                    )
                    row_out["pred_rt_v48_family_condition"] = float(np.mean(fc_values)) if len(fc_values) else np.nan
                    row_out["pred_rt_v48_family_condition_global"] = float(np.mean(pd.to_numeric(fc_frame["pred_rt_family_condition_global"], errors="coerce"))) if len(fc_frame) else np.nan
                    row_out["pred_rt_v48_family_condition_delta"] = float(np.mean(pd.to_numeric(fc_frame["pred_rt_family_condition_delta"], errors="coerce"))) if len(fc_frame) else np.nan
                    row_out["pred_rt_v48_family_condition_error_est"] = _nanmax_or_nan(fc_frame["pred_rt_family_condition_error_est"]) if len(fc_frame) else np.nan
                    row_out["pred_rt_v48_family_condition_accepted"] = fc_all_accepted
                    row_out["pred_rt_v48_family_condition_family_id"] = fc_frame["pred_rt_family_condition_family_id"].iloc[0] if len(fc_frame) else ""
                    row_out["pred_rt_v48_family_condition_status"] = fc_frame["pred_rt_family_condition_status"].iloc[0] if len(fc_frame) else ""
                    row_out["pred_rt_v48_family_condition_blend_weight"] = fc_frame["pred_rt_family_condition_blend_weight"].iloc[0] if len(fc_frame) else 0.0
                    row_out["pred_rt_v48_family_condition_neighbor_distance"] = fc_frame["pred_rt_family_condition_neighbor_distance"].iloc[0] if len(fc_frame) else np.nan
                    row_out["pred_rt_v48_family_condition_neighbor_threshold"] = fc_frame["pred_rt_family_condition_neighbor_threshold"].iloc[0] if len(fc_frame) else np.nan
                    row_out["pred_rt_v48_family_condition_train_methods"] = fc_frame["pred_rt_family_condition_train_methods"].iloc[0] if len(fc_frame) else 0
                    row_out["pred_rt_v48_family_condition_train_rows"] = fc_frame["pred_rt_family_condition_train_rows"].iloc[0] if len(fc_frame) else 0
                    if fc_all_accepted:
                        row_out["pred_rt_v48_mean"] = float(np.mean(fc_values))
                        row_out["pred_rt_v48_first"] = float(fc_values[0])
                        row_out["pred_rt_v48_all"] = ";".join(f"{x:.6f}" for x in fc_values)
                        row_out["pred_rt_v48_family_condition_used"] = True
                        row_out["pred_rt_v48_model_first"] = "family_condition_aware_expert"
                        row_out["pred_rt_v48_models"] = ";".join(["family_condition_aware_expert"] * len(fc_values))
                        row_out["pred_rt_v48_route_reason"] = "family_condition_aware_expert_branch"
                        row_out["pred_rt_v48_selected_expert"] = f"family_condition:{row_out['pred_rt_v48_family_condition_family_id']}"
            if public_global_predictor is not None:
                row_out["pred_rt_v48_public_global"] = np.nan
                row_out["pred_rt_v48_public_global_error_est"] = np.nan
                row_out["pred_rt_v48_public_global_accepted"] = False
                row_out["pred_rt_v48_public_global_used"] = False
                row_out["pred_rt_v48_public_global_threshold"] = args.public_global_max_predicted_error
                row_out["pred_rt_v48_public_global_model"] = args.public_global_model
                row_out["pred_rt_v48_public_global_admitted"] = _should_admit_public_global_branch(
                    public_safe_fallback=bool(details and details[0].get("v48_public_safe_fallback", False)),
                    meta_dict=meta_dict,
                    metadata_path=args.metadata_path,
                    input_path=input_path,
                    protected_public_expert_route=protected_public_expert_route,
                )
                if (
                    row_out["pred_rt_v48_public_global_admitted"]
                    and not bool(row_out.get("pred_rt_v48_cross_condition_used", False))
                    and not bool(row_out.get("pred_rt_v48_family_condition_used", False))
                    and smiles_list
                ):
                    pg_input = pd.DataFrame({"smiles": smiles_list})
                    pg_frame = public_global_predictor.predict_frame(
                        pg_input,
                        metadata_dict=meta_dict,
                        smiles_col="smiles",
                        max_predicted_error=args.public_global_max_predicted_error,
                    )
                    pg_values = pd.to_numeric(pg_frame["pred_rt_public_global"], errors="coerce").to_numpy(dtype=float)
                    pg_errors = pd.to_numeric(pg_frame["pred_rt_public_global_error_est"], errors="coerce").to_numpy(dtype=float)
                    pg_accepted_values = pg_frame["pred_rt_public_global_accepted"].astype(bool).to_numpy()
                    pg_all_accepted = bool(
                        len(pg_values) == len(smiles_list)
                        and np.all(np.isfinite(pg_values))
                        and np.all(pg_accepted_values)
                    )
                    row_out["pred_rt_v48_public_global"] = float(np.mean(pg_values)) if len(pg_values) else np.nan
                    row_out["pred_rt_v48_public_global_error_est"] = float(np.nanmax(pg_errors)) if len(pg_errors) else np.nan
                    row_out["pred_rt_v48_public_global_accepted"] = pg_all_accepted
                    if pg_all_accepted:
                        row_out["pred_rt_v48_mean"] = float(np.mean(pg_values))
                        row_out["pred_rt_v48_first"] = float(pg_values[0])
                        row_out["pred_rt_v48_all"] = ";".join(f"{x:.6f}" for x in pg_values)
                        row_out["pred_rt_v48_public_global_used"] = True
                        row_out["pred_rt_v48_model_first"] = "public_aware_global"
                        row_out["pred_rt_v48_models"] = ";".join(["public_aware_global"] * len(pg_values))
                        row_out["pred_rt_v48_route_reason"] = "public_aware_global_branch"
                        row_out["pred_rt_v48_selected_expert"] = "public_aware_global"
            if bool(row_out.get("pred_rt_v48_cross_condition_used", False)):
                row_out["pred_rt_v48_strategy"] = "cross_condition_transfer"
            elif bool(row_out.get("pred_rt_v48_family_condition_used", False)):
                row_out["pred_rt_v48_strategy"] = "family_condition_aware_expert"
            elif bool(row_out.get("pred_rt_v48_public_global_used", False)):
                row_out["pred_rt_v48_strategy"] = "public_aware_global"
            elif bool(row_out.get("pred_rt_v48_public_safe_fallback", False)):
                row_out["pred_rt_v48_strategy"] = "legacy_public_safe_fallback"
            elif bool(row_out.get("pred_rt_v48_public_expert_route_protected", False)):
                row_out["pred_rt_v48_strategy"] = "public_expert"
            else:
                row_out["pred_rt_v48_strategy"] = "rt_predict"
            if args.include_detail_json:
                row_out["pred_rt_v48_detail_json"] = json.dumps(details, ensure_ascii=False)
            rows_out.append(row_out)

        if idx == total_rows or idx % 25 == 0:
            print(f"[PREDICT] {idx}/{total_rows}")

    out_df = pd.DataFrame(rows_out)
    if args.calibration_adapter:
        from core.calibration.lightweight_adapter import apply_adapter_to_frame

        final_col = args.calibration_adapter_final_col
        if not final_col:
            final_col = "pred_rt_v48_mean" if "pred_rt_v48_mean" in out_df.columns else "pred_rt_v48"
        out_df = apply_adapter_to_frame(
            out_df,
            args.calibration_adapter,
            mode=args.calibration_adapter_mode,
            update_final=(args.calibration_adapter_mode != "report"),
            final_col=final_col,
        )
        if args.calibration_adapter_mode != "report" and "pred_rt_v48_strategy" in out_df.columns:
            out_df["pred_rt_v48_strategy_before_calibration_adapter"] = out_df["pred_rt_v48_strategy"]
            used = out_df["pred_rt_calibration_adapter_used"].astype(bool)
            out_df.loc[used, "pred_rt_v48_strategy"] = (
                out_df.loc[used, "pred_rt_v48_strategy_before_calibration_adapter"].astype(str)
                + "+lightweight_calibration_adapter"
            )
    from core.routing.route_registry import annotate_canonical_routes

    out_df = annotate_canonical_routes(out_df)
    if args.simple_output:
        if args.explode_multi_smiles or len(out_df) != len(df):
            raise ValueError("--simple_output requires one output row per input row")
        final_column = next(
            (name for name in ("pred_rt_v48_mean", "pred_rt_v48", "pred_rt_v48_first") if name in out_df.columns),
            None,
        )
        if final_column is None:
            raise ValueError("No final RT prediction column is available for --simple_output")
        simple_df = df.reset_index(drop=True).drop(columns=["predicted_rt"], errors="ignore").copy()
        smiles_position = list(simple_df.columns).index("smiles") + 1
        simple_df.insert(
            smiles_position,
            "predicted_rt",
            pd.to_numeric(out_df[final_column], errors="coerce").to_numpy(),
        )
        out_df = simple_df
    out_csv = args.out_csv
    if not out_csv:
        root, _ = os.path.splitext(input_path)
        out_csv = root + "_with_v48_rt.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] input: {input_path}")
    print(f"[DONE] rows : {len(out_df)}")
    print(f"[SAVE]      {out_csv}")


if __name__ == "__main__":
    main()
