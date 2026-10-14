import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(CORE_ROOT)
for _path in (SCRIPT_DIR, CORE_ROOT, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analysis.compare.compare_v47_vs_retention_time_gnn as base
from core.data.dataset_v11_pro_meta_threehead_v50_mcmrt_transfer import (
    GNN_REPO,
    METHOD_IDS,
    ROOT,
    build_feature_matrix,
    load_gnn_train_test_predictions,
    load_method_split,
    predict_backbone_rows,
)
from core.models.model_v11_pro_meta_threehead_v50_mcmrt_transfer import (
    GnnMethodBundle,
    PrototypeRouterV50,
    V50TransferPredictor,
)


RUN_MODE = "train"

OUT_DIR = os.path.join(
    ROOT,
    "artifacts",
    "archive",
    "runs",
    "v11_pro_meta_threehead_v50_mcmrt_transfer",
)
ROUTER_MANIFEST = os.path.join(OUT_DIR, "router_manifest_v50.json")
MANIFEST_PATH = ROUTER_MANIFEST
LEGACY_MANIFEST_PATH = os.path.join(OUT_DIR, "transfer_manifest_v50.json")
EVAL_CSV = os.path.join(OUT_DIR, "v50_eval_mcmrt10.csv")
EVAL_JSON = os.path.join(OUT_DIR, "v50_eval_mcmrt10.json")

BACKBONE_TRAIN_SCRIPT = os.path.join(ROOT, "train_v50_backbone.py")
BACKBONE_ROUTER_MANIFEST = os.path.join(OUT_DIR, "router_manifest_v50_backbone.json")

FEATURE_NAMES = [
    "gnn_pred",
    "backbone_global_pred",
    "backbone_delta_pred",
    "backbone_final_pred",
    "backbone_top1_pred",
    "backbone_minus_gnn",
]
RIDGE_LAMBDA = 0.1
CAL_SLOPE_PENALTY = 0.25
CAL_INTERCEPT_PENALTY = 0.05
CAL_SLOPE_MIN = 0.7
CAL_SLOPE_MAX = 1.3
USE_BACKBONE_TRAIN_AUGMENT = False

SMILES = "CC1=C(C=C(OC1=O)CC(C(C)C(C(C)C)O)O)O"
METHOD_METADATA_PATH = os.path.join(ROOT, "public_riken_metadata.yaml")


def fit_linear_residual_head(train_x, train_y, gnn_pred, ridge_lambda=RIDGE_LAMBDA, sample_weight=None):
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    gnn_pred = np.asarray(gnn_pred, dtype=np.float64).reshape(-1)
    target = train_y - gnn_pred
    design = np.column_stack([train_x, np.ones(len(train_x), dtype=np.float64)])
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        sqrt_w = np.sqrt(np.clip(sample_weight, 1e-12, None))
        design = design * sqrt_w[:, None]
        target = target * sqrt_w
    reg = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ target)
    return coef


def fit_method_calibrator(base_pred, true):
    base_pred = np.asarray(base_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if len(base_pred) == 0:
        return 1.0, 0.0
    if len(base_pred) == 1 or float(np.std(base_pred)) < 1e-8:
        return 1.0, float(np.mean(true) - np.mean(base_pred))

    x = np.column_stack([base_pred, np.ones_like(base_pred)])
    reg = np.diag([CAL_SLOPE_PENALTY, CAL_INTERCEPT_PENALTY]).astype(np.float64)
    prior = np.asarray([1.0, 0.0], dtype=np.float64)
    coef = np.linalg.solve(x.T @ x + reg, x.T @ true + reg @ prior)
    slope = float(np.clip(coef[0], CAL_SLOPE_MIN, CAL_SLOPE_MAX))
    intercept = float(np.mean(true) - slope * np.mean(base_pred))
    return slope, intercept


def apply_residual_head(x_std, coef, gnn_pred):
    x_std = np.asarray(x_std, dtype=np.float64)
    gnn_pred = np.asarray(gnn_pred, dtype=np.float64).reshape(-1)
    residual = np.column_stack([x_std, np.ones(len(x_std), dtype=np.float64)]) @ coef
    return gnn_pred + residual, residual


def apply_calibrator(pred, slope, intercept):
    pred = np.asarray(pred, dtype=np.float64)
    return slope * pred + intercept


def collect_backbone_augment_rows(backbone_mod, backbone_predictor, router, methods):
    family_map = backbone_mod.load_method_family_map(backbone_mod.FAMILY_MAP_PATH)
    method_dirs = backbone_mod.list_method_dirs(backbone_mod.SEARCH_ROOT)
    print(f"[AUGMENT] scanning local methods under {backbone_mod.SEARCH_ROOT}")
    print(f"[AUGMENT] found candidate directories={len(method_dirs)}")

    selected_pairs = []
    skipped = []
    total_dirs = len(method_dirs)
    for idx, method_dir in enumerate(method_dirs, start=1):
        ds, info = backbone_mod.load_method_dataset(
            method_dir,
            min_rows=backbone_mod.MIN_ROWS,
            family_map=family_map,
        )
        if ds is None:
            skipped.append(info)
        else:
            selected_pairs.append((ds, info))
        if idx == total_dirs or idx % 50 == 0:
            print(
                f"[AUGMENT][SCAN] {idx}/{total_dirs} "
                f"selected={len(selected_pairs)} skipped={len(skipped)}"
            )

    selected_pairs.sort(key=lambda x: x[1]["rows"], reverse=True)
    max_experts = getattr(backbone_mod, "MAX_EXPERTS", None)
    if max_experts is not None:
        selected_pairs = selected_pairs[:max_experts]

    print(
        f"[AUGMENT] usable local methods={len(selected_pairs)} "
        f"after min_rows={backbone_mod.MIN_ROWS}"
    )
    if not selected_pairs:
        return [], skipped

    gnn_cache = {}
    rows = []
    total = len(selected_pairs)
    for idx, (ds, info) in enumerate(selected_pairs, start=1):
        meta_vec = np.asarray(ds.meta[0], dtype=np.float32)
        route = router.route(meta_vec)
        top1 = route.get("top1")
        if top1 is None:
            continue
        method_name = str(top1["method_name"])
        print(
            f"[AUGMENT][START] {idx}/{total} "
            f"local_method={info['method_code']} routed_to={method_name} "
            f"source_rows={len(ds.smiles)}"
        )
        if method_name not in gnn_cache:
            method_info = methods[method_name]
            gnn_cache[method_name] = GnnMethodBundle(
                dataset_name=method_info["dataset_name"],
                model_path=method_info["gnn_model_path"],
                gnn_repo=GNN_REPO,
                device="cpu",
            )
        gnn_bundle = gnn_cache[method_name]
        backbone_rows = backbone_mod.predict_backbone_rows(
            backbone_predictor,
            backbone_mod.smiles_to_fp,
            meta_vec,
            ds.smiles.tolist(),
        ) if hasattr(backbone_mod, "predict_backbone_rows") else None
        if backbone_rows is None:
            backbone_rows = {}
            pred_global = []
            pred_delta = []
            pred_final = []
            pred_top1 = []
            for fp in ds.fps:
                out = backbone_predictor.predict(fp, meta_vec)
                pred_global.append(np.nan if out.get("pred_global") is None else float(out.get("pred_global")))
                pred_delta.append(0.0 if out.get("pred_delta") is None else float(out.get("pred_delta")))
                pred_final.append(np.nan if out.get("pred_final") is None else float(out.get("pred_final")))
                pred_top1.append(np.nan if out.get("pred_routed_top1") is None else float(out.get("pred_routed_top1")))
            backbone_rows = {
                "pred_global": np.asarray(pred_global, dtype=np.float64),
                "pred_delta": np.asarray(pred_delta, dtype=np.float64),
                "pred_final": np.asarray(pred_final, dtype=np.float64),
                "pred_routed_top1": np.asarray(pred_top1, dtype=np.float64),
            }

        gnn_pred = []
        keep_idx = []
        skipped_unsupported = 0
        skipped_reason_counts = {}
        for sample_idx, smi in enumerate(ds.smiles.tolist()):
            try:
                gnn_pred.append(float(gnn_bundle.predict(str(smi))))
                keep_idx.append(sample_idx)
            except ValueError as exc:
                skipped_unsupported += 1
                reason = str(exc).strip() or exc.__class__.__name__
                skipped_reason_counts[reason] = skipped_reason_counts.get(reason, 0) + 1
                continue
        if not keep_idx:
            print(
                f"[AUGMENT][SKIP] local_method={info['method_code']} "
                f"routed_to={method_name} no gnn-compatible rows "
                f"reasons={json.dumps(skipped_reason_counts, ensure_ascii=False)}"
            )
            continue

        keep_idx = np.asarray(keep_idx, dtype=np.int64)
        gnn_pred = np.asarray(gnn_pred, dtype=np.float64)
        filtered_backbone_rows = {
            "pred_global": np.asarray(backbone_rows["pred_global"], dtype=np.float64)[keep_idx],
            "pred_delta": np.asarray(backbone_rows["pred_delta"], dtype=np.float64)[keep_idx],
            "pred_final": np.asarray(backbone_rows["pred_final"], dtype=np.float64)[keep_idx],
            "pred_routed_top1": np.asarray(backbone_rows["pred_routed_top1"], dtype=np.float64)[keep_idx],
        }
        filtered_rt = np.asarray(ds.rts, dtype=np.float64)[keep_idx]
        x = build_feature_matrix(gnn_pred, filtered_backbone_rows)
        rows.append(
            {
                "method_code": str(info["method_code"]),
                "routed_method": method_name,
                "rows": int(len(filtered_rt)),
                "skipped_unsupported_rows": int(skipped_unsupported),
                "x": x,
                "rt": filtered_rt,
                "gnn_pred": gnn_pred,
            }
        )
        print(
            f"[AUGMENT] {idx}/{total} "
            f"local_method={info['method_code']} routed_to={method_name} "
            f"rows={len(filtered_rt)} skipped_unsupported={skipped_unsupported}"
        )
        if skipped_reason_counts:
            print(
                f"[AUGMENT][DETAIL] local_method={info['method_code']} "
                f"reasons={json.dumps(skipped_reason_counts, ensure_ascii=False)}"
            )
    return rows, skipped


def build_predictor(manifest_path=None):
    manifest_path = manifest_path or ROUTER_MANIFEST
    backbone_mod = base.import_module_from_path("v50_backbone_train_module", BACKBONE_TRAIN_SCRIPT)
    backbone_predictor = backbone_mod.build_predictor(BACKBONE_ROUTER_MANIFEST)
    return V50TransferPredictor(
        manifest_path=manifest_path,
        backbone_predictor=backbone_predictor,
        smiles_to_fp=backbone_mod.smiles_to_fp,
        extract_method_meta_vector=backbone_mod.extract_method_meta_vector,
        gnn_repo=GNN_REPO,
        device="cpu",
    )


def evaluate_manifest(manifest_path):
    predictor = build_predictor(manifest_path)
    rows = []
    for method_name in METHOD_IDS:
        split = load_method_split(method_name)
        metadata = split["metadata"]
        test_df = split["test_df"]
        preds = []
        routed = []
        for smiles in test_df["smiles"].tolist():
            out = predictor.predict(smiles, metadata)
            preds.append(float(out["pred_final"]))
            routed.append(str(out.get("router_top1", "")))
        metrics = base.compute_metrics(test_df["rt"].to_numpy(), np.asarray(preds))
        rows.append(
            {
                "dataset": method_name,
                "model_name": "v50_transfer",
                **metrics,
                "router_top1": method_name,
                "router_top1_acc": float(np.mean([1.0 if t == method_name else 0.0 for t in routed])),
            }
        )

    eval_df = pd.DataFrame(rows)
    summary = {
        "mae_macro": float(eval_df["mae"].mean()),
        "rmse_macro": float(eval_df["rmse"].mean()),
        "r2_macro": float(eval_df["r2"].mean()),
        "router_top1_acc": float(eval_df["router_top1_acc"].mean()),
    }
    eval_df.to_csv(EVAL_CSV, index=False, encoding="utf-8-sig")
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_method": rows}, f, ensure_ascii=False, indent=2)
    print("[EVAL]", json.dumps(summary, ensure_ascii=False))
    print("[SAVE]", EVAL_CSV)
    print("[SAVE]", EVAL_JSON)
    return summary


def train_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    backbone_mod = base.import_module_from_path("v50_train_support_module", BACKBONE_TRAIN_SCRIPT)
    backbone_predictor = backbone_mod.build_predictor(BACKBONE_ROUTER_MANIFEST)

    pooled_x = []
    pooled_rt = []
    pooled_gnn = []
    pooled_weight = []
    methods = {}

    for method_name in METHOD_IDS:
        split = load_method_split(method_name)
        metadata = split["metadata"]
        meta_vec = backbone_mod.extract_method_meta_vector(metadata)
        train_df = split["train_df"]
        test_df = split["test_df"]
        gnn_blob = load_gnn_train_test_predictions(method_name)

        train_backbone = predict_backbone_rows(backbone_predictor, backbone_mod.smiles_to_fp, meta_vec, train_df["smiles"].tolist())
        test_backbone = predict_backbone_rows(backbone_predictor, backbone_mod.smiles_to_fp, meta_vec, test_df["smiles"].tolist())

        train_true = train_df["rt"].to_numpy(dtype=np.float64)
        test_true = test_df["rt"].to_numpy(dtype=np.float64)
        if len(train_true) != len(gnn_blob["train_pred"]) or len(test_true) != len(gnn_blob["test_pred"]):
            raise RuntimeError(f"{method_name}: split length mismatch")
        if not np.allclose(train_true, gnn_blob["train_true"], atol=1e-5):
            raise RuntimeError(f"{method_name}: train truth mismatch between backbone and GNN")
        if not np.allclose(test_true, gnn_blob["test_true"], atol=1e-5):
            raise RuntimeError(f"{method_name}: test truth mismatch between backbone and GNN")

        train_x = build_feature_matrix(gnn_blob["train_pred"], train_backbone)
        test_x = build_feature_matrix(gnn_blob["test_pred"], test_backbone)

        pooled_x.append(train_x)
        pooled_rt.append(train_true)
        pooled_gnn.append(np.asarray(gnn_blob["train_pred"], dtype=np.float64))
        pooled_weight.append(np.ones(len(train_true), dtype=np.float64))

        methods[method_name] = {
            "dataset_name": method_name,
            "prototype": np.asarray(meta_vec, dtype=np.float32).tolist(),
            "gnn_model_path": str(gnn_blob["model_path"]),
            "metadata_source": os.path.abspath(split["path"]),
            "train_rows": int(len(train_true)),
            "test_rows": int(len(test_true)),
            "train_true": train_true.tolist(),
            "test_true": test_true.tolist(),
            "train_gnn_pred": np.asarray(gnn_blob["train_pred"], dtype=np.float64).tolist(),
            "test_gnn_pred": np.asarray(gnn_blob["test_pred"], dtype=np.float64).tolist(),
            "train_features": train_x.tolist(),
            "test_features": test_x.tolist(),
        }
        print(f"[COLLECT] {method_name} train={len(train_true)} test={len(test_true)}")

    augment_rows = []
    augment_skipped = []
    if USE_BACKBONE_TRAIN_AUGMENT:
        router_manifest = {
            "methods": {
                method_name: {
                    "dataset_name": info["dataset_name"],
                    "prototype": info["prototype"],
                    "gnn_model_path": info["gnn_model_path"],
                }
                for method_name, info in methods.items()
            },
            "router_stats": {
                "meta_mean": np.stack([np.asarray(info["prototype"], dtype=np.float32) for info in methods.values()], axis=0).mean(axis=0).tolist(),
                "meta_std": (np.stack([np.asarray(info["prototype"], dtype=np.float32) for info in methods.values()], axis=0).std(axis=0) + 1e-6).tolist(),
            },
        }
        router = PrototypeRouterV50(router_manifest)
        augment_rows, augment_skipped = collect_backbone_augment_rows(backbone_mod, backbone_predictor, router, methods)
        if augment_rows:
            n_mcmrt = int(sum(len(x) for x in pooled_rt))
            n_local = int(sum(int(row["rows"]) for row in augment_rows))
            local_weight = float(n_mcmrt / max(n_local, 1))
            for row in augment_rows:
                pooled_x.append(row["x"])
                pooled_rt.append(row["rt"])
                pooled_gnn.append(row["gnn_pred"])
                pooled_weight.append(np.full(int(row["rows"]), local_weight, dtype=np.float64))
            print(f"[AUGMENT] local_methods={len(augment_rows)} local_rows={n_local} sample_weight={local_weight:.6f}")
        else:
            print("[AUGMENT] no local backbone rows added")

    pooled_x = np.concatenate(pooled_x, axis=0)
    pooled_rt = np.concatenate(pooled_rt, axis=0)
    pooled_gnn = np.concatenate(pooled_gnn, axis=0)
    pooled_weight = np.concatenate(pooled_weight, axis=0)

    feature_mean = pooled_x.mean(axis=0)
    feature_std = pooled_x.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    pooled_x_std = (pooled_x - feature_mean) / feature_std
    coef = fit_linear_residual_head(
        pooled_x_std,
        pooled_rt,
        pooled_gnn,
        ridge_lambda=RIDGE_LAMBDA,
        sample_weight=pooled_weight,
    )
    train_pred_base, train_residual = apply_residual_head(pooled_x_std, coef, pooled_gnn)
    train_metrics_base = base.compute_metrics(pooled_rt, train_pred_base)

    per_method_details = {}
    for method_name, info in methods.items():
        train_x = np.asarray(info.pop("train_features"), dtype=np.float64)
        test_x = np.asarray(info.pop("test_features"), dtype=np.float64)
        train_true = np.asarray(info.pop("train_true"), dtype=np.float64)
        test_true = np.asarray(info.pop("test_true"), dtype=np.float64)
        train_gnn_pred = np.asarray(info.pop("train_gnn_pred"), dtype=np.float64)
        test_gnn_pred = np.asarray(info.pop("test_gnn_pred"), dtype=np.float64)

        train_x_std = (train_x - feature_mean) / feature_std
        test_x_std = (test_x - feature_mean) / feature_std
        train_base_pred, _ = apply_residual_head(train_x_std, coef, train_gnn_pred)
        test_base_pred, _ = apply_residual_head(test_x_std, coef, test_gnn_pred)

        slope, intercept = fit_method_calibrator(train_base_pred, train_true)
        train_final_pred = apply_calibrator(train_base_pred, slope, intercept)
        test_final_pred = apply_calibrator(test_base_pred, slope, intercept)

        info["calibrator"] = {
            "slope": float(slope),
            "intercept": float(intercept),
        }
        info["train_metrics_base"] = base.compute_metrics(train_true, train_base_pred)
        info["train_metrics_final"] = base.compute_metrics(train_true, train_final_pred)
        info["test_metrics_final"] = base.compute_metrics(test_true, test_final_pred)
        info["train_backbone_mae"] = float(np.mean(np.abs(train_x[:, 3] - train_true)))
        info["test_backbone_mae"] = float(np.mean(np.abs(test_x[:, 3] - test_true)))
        per_method_details[method_name] = {
            "calibrator": info["calibrator"],
            "train_metrics_base": info["train_metrics_base"],
            "train_metrics_final": info["train_metrics_final"],
            "test_metrics_final": info["test_metrics_final"],
        }
        print(
            f"[FIT] {method_name} "
            f"slope={slope:.4f} intercept={intercept:.4f} "
            f"test_mae={info['test_metrics_final']['mae']:.4f}"
        )

    proto_arr = np.stack([np.asarray(info["prototype"], dtype=np.float32) for info in methods.values()], axis=0)
    manifest = {
        "version": "v50_mcmrt_transfer",
        "root": ROOT,
        "gnn_repo": GNN_REPO,
        "backbone_train_script": BACKBONE_TRAIN_SCRIPT,
        "backbone_router_manifest": BACKBONE_ROUTER_MANIFEST,
        "method_ids": METHOD_IDS,
        "methods": methods,
        "router_stats": {
            "meta_mean": proto_arr.mean(axis=0).tolist(),
            "meta_std": (proto_arr.std(axis=0) + 1e-6).tolist(),
        },
        "hybrid": {
            "feature_names": FEATURE_NAMES,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "weights": coef[:-1].tolist(),
            "bias": float(coef[-1]),
            "ridge_lambda": float(RIDGE_LAMBDA),
            "train_metrics_base": train_metrics_base,
            "train_rows_total": int(len(pooled_rt)),
        },
        "calibration": {
            "slope_penalty": float(CAL_SLOPE_PENALTY),
            "intercept_penalty": float(CAL_INTERCEPT_PENALTY),
            "slope_bounds": [float(CAL_SLOPE_MIN), float(CAL_SLOPE_MAX)],
        },
        "augment": {
            "use_backbone_train_augment": bool(USE_BACKBONE_TRAIN_AUGMENT),
            "local_method_count": int(len(augment_rows)),
            "local_row_count": int(sum(int(row["rows"]) for row in augment_rows)),
            "skipped_local_method_count": int(len(augment_skipped)),
        },
        "details": per_method_details,
    }

    with open(ROUTER_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(LEGACY_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[DONE] manifest:", ROUTER_MANIFEST)
    print("[TRAIN]", json.dumps(train_metrics_base, ensure_ascii=False))
    evaluate_manifest(ROUTER_MANIFEST)


def predict_once():
    predictor = build_predictor(ROUTER_MANIFEST)
    backbone_mod = base.import_module_from_path("v50_demo_backbone_train_mod", BACKBONE_TRAIN_SCRIPT)
    meta = backbone_mod.load_yaml_or_json(METHOD_METADATA_PATH)
    out = predictor.predict(SMILES, meta)
    out["smiles"] = SMILES
    out["method_metadata_path"] = METHOD_METADATA_PATH
    save_path = os.path.join(OUT_DIR, "v50_last_prediction.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("[SAVE]", save_path)


def eval_existing():
    os.makedirs(OUT_DIR, exist_ok=True)
    evaluate_manifest(ROUTER_MANIFEST)


def main():
    if RUN_MODE == "train":
        train_all()
    elif RUN_MODE == "predict":
        predict_once()
    elif RUN_MODE == "eval":
        eval_existing()
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
