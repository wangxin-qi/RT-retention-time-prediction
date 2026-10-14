"""Canonical names and output annotations for the six deployed RouteRT routes.

The route is assigned only from inference-time evidence.  The query's measured RT
is never consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import pandas as pd


class RouteId(str, Enum):
    KNOWN_METHOD_EXPERT = "known_method_expert"
    PUBLIC_EXPERT = "public_expert"
    CROSS_CONDITION_TRANSFER = "cross_condition_transfer"
    FAMILY_EXPERT = "family_expert"
    PUBLIC_AWARE_GLOBAL = "public_aware_global"
    GLOBAL_FALLBACK = "global_fallback"


@dataclass(frozen=True)
class RouteDefinition:
    route_id: RouteId
    english_name: str
    chinese_name: str


ROUTE_REGISTRY: dict[RouteId, RouteDefinition] = {
    RouteId.KNOWN_METHOD_EXPERT: RouteDefinition(
        RouteId.KNOWN_METHOD_EXPERT, "Known-method expert", "已知方法专家"
    ),
    RouteId.PUBLIC_EXPERT: RouteDefinition(
        RouteId.PUBLIC_EXPERT, "Public expert", "公共数据专家"
    ),
    RouteId.CROSS_CONDITION_TRANSFER: RouteDefinition(
        RouteId.CROSS_CONDITION_TRANSFER, "Cross-condition transfer", "跨条件迁移"
    ),
    RouteId.FAMILY_EXPERT: RouteDefinition(
        RouteId.FAMILY_EXPERT, "Family expert", "家族专家"
    ),
    RouteId.PUBLIC_AWARE_GLOBAL: RouteDefinition(
        RouteId.PUBLIC_AWARE_GLOBAL, "Public-aware global", "公共感知全局"
    ),
    RouteId.GLOBAL_FALLBACK: RouteDefinition(
        RouteId.GLOBAL_FALLBACK, "Global fallback", "全局回退"
    ),
}


_EXACT_MATCH_TYPES = {
    "explicit_method_code",
    "exact_method",
    "method_template_exact",
}
_EXACT_MATCH_SOURCES = {
    "explicit_method_code",
    "exact_method",
    "meta_signature_exact",
    "route_signature_exact",
}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return bool(value)


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def classify_output_row(row: Mapping[str, Any]) -> tuple[RouteId, str]:
    """Map legacy diagnostic fields to exactly one canonical deployed route."""

    if _truthy(row.get("pred_rt_v48_cross_condition_used", False)):
        return RouteId.CROSS_CONDITION_TRANSFER, "accepted_source_rt_transfer"
    if _truthy(row.get("pred_rt_v48_family_condition_used", False)):
        return RouteId.FAMILY_EXPERT, "accepted_family_condition_expert"
    if _truthy(row.get("pred_rt_v48_public_global_used", False)):
        return RouteId.PUBLIC_AWARE_GLOBAL, "accepted_public_aware_global"
    if _truthy(row.get("pred_rt_v48_public_expert_route_protected", False)):
        return RouteId.PUBLIC_EXPERT, "validated_public_expert"

    match_type = _text(row, "pred_rt_v48_local_match_type")
    match_source = _text(row, "pred_rt_v48_local_match_source")
    if (
        not _truthy(row.get("pred_rt_v48_public_safe_fallback", False))
        and (
            _truthy(row.get("pred_rt_v48_known_method_route_protected", False))
            or match_type in _EXACT_MATCH_TYPES
            or match_source in _EXACT_MATCH_SOURCES
        )
    ):
        return RouteId.KNOWN_METHOD_EXPERT, "exact_local_method_match"

    return RouteId.GLOBAL_FALLBACK, "higher_priority_routes_ineligible_or_rejected"


def annotate_canonical_routes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable route names and adapter status to a prediction frame."""

    out = frame.copy()
    decisions = [classify_output_row(row) for row in out.to_dict("records")]
    route_ids = [route_id for route_id, _ in decisions]
    out["pred_rt_route_id"] = [route_id.value for route_id in route_ids]
    out["pred_rt_route_name"] = [
        ROUTE_REGISTRY[route_id].english_name for route_id in route_ids
    ]
    out["pred_rt_route_name_cn"] = [
        ROUTE_REGISTRY[route_id].chinese_name for route_id in route_ids
    ]
    out["pred_rt_route_decision_reason"] = [reason for _, reason in decisions]
    if "pred_rt_calibration_adapter_used" in out.columns:
        out["pred_rt_anchor_adapter_applied"] = (
            out["pred_rt_calibration_adapter_used"].fillna(False).astype(bool)
        )
    else:
        out["pred_rt_anchor_adapter_applied"] = False
    if "pred_rt_calibration_adapter_evaluated" in out.columns:
        out["pred_rt_anchor_adapter_evaluated"] = (
            out["pred_rt_calibration_adapter_evaluated"].fillna(False).astype(bool)
        )
    else:
        out["pred_rt_anchor_adapter_evaluated"] = False
    return out
