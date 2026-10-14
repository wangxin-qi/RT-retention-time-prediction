import importlib.util
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(CORE_ROOT)
for _path in (SCRIPT_DIR, CORE_ROOT, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analysis.compare.compare_v47_vs_gnn_mcmrt as mcmrt
import analysis.compare.compare_v47_vs_retention_time_gnn as base


ROOT = base.ROOT
GNN_REPO = base.GNN_REPO
METHOD_IDS = [f"dataset_{i:02d}" for i in range(1, 31)]


def import_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_method_split(method_name):
    path = os.path.join(ROOT, "data", "MCMRT", method_name.replace("dataset_", "Dataset_") + ".xlsx")
    rt_df = mcmrt.extract_rt_df(path)
    metadata = mcmrt.extract_metadata_from_setup_sheet(path)
    test_idx = base.get_test_indices(len(rt_df))
    train_mask = np.ones(len(rt_df), dtype=bool)
    train_mask[test_idx] = False
    train_df = rt_df.iloc[np.where(train_mask)[0]].reset_index(drop=True)
    test_df = rt_df.iloc[test_idx].reset_index(drop=True)
    return {
        "path": path,
        "metadata": metadata,
        "rt_df": rt_df,
        "train_df": train_df,
        "test_df": test_df,
        "test_idx": test_idx,
    }


def load_gnn_train_test_predictions(dataset_name):
    if GNN_REPO not in sys.path:
        sys.path.insert(0, GNN_REPO)

    cwd = os.getcwd()
    os.chdir(GNN_REPO)
    try:
        from dataset import GraphDataset
        from util import collate_2D_graphs
        from gnn import GINPredictor

        model_mod = import_module_from_path(
            f"v50_train_gnn_model_{dataset_name}",
            os.path.join(GNN_REPO, "model.py"),
        )

        Trainer = model_mod.Trainer
        trndata = GraphDataset(dataset_name, cv_id=base.CV_ID, split="trn", seed=base.CV_SEED)
        tstdata = GraphDataset(dataset_name, cv_id=base.CV_ID, split="tst", seed=base.CV_SEED)

        net = GINPredictor(
            node_in_feats=trndata.node_attr.shape[1],
            edge_in_feats=trndata.edge_attr.shape[1],
        )
        trainer = Trainer(net, os.path.join(GNN_REPO, "model", "model_SMRT.pt"), "cpu")
        model_path = os.path.join(GNN_REPO, "model", f"model_{dataset_name}_cv_{base.CV_ID}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        trainer.load(model_path)

        trn_y = trndata.label.reshape(-1)
        trn_y_mean, trn_y_std = np.mean(trn_y), np.std(trn_y)
        trn_loader = DataLoader(trndata, batch_size=len(trndata), shuffle=False, collate_fn=collate_2D_graphs)
        tst_loader = DataLoader(tstdata, batch_size=len(tstdata), shuffle=False, collate_fn=collate_2D_graphs)
        train_pred = trainer.inference(trn_loader, trn_y_mean, trn_y_std).reshape(-1)
        test_pred = trainer.inference(tst_loader, trn_y_mean, trn_y_std).reshape(-1)
        return {
            "model_path": os.path.abspath(model_path),
            "train_true": trndata.label.reshape(-1),
            "train_pred": train_pred,
            "test_true": tstdata.label.reshape(-1),
            "test_pred": test_pred,
        }
    finally:
        os.chdir(cwd)


def predict_backbone_rows(backbone_predictor, smiles_to_fp, meta_vec, smiles_list):
    pred_global = []
    pred_delta = []
    pred_final = []
    pred_top1 = []
    for smiles in smiles_list:
        fp = smiles_to_fp(smiles)
        out = backbone_predictor.predict(fp, meta_vec)
        global_pred = out.get("pred_global")
        delta_pred = out.get("pred_delta")
        final_pred = out.get("pred_final")
        top1_pred = out.get("pred_routed_top1")
        pred_global.append(np.nan if global_pred is None else float(global_pred))
        pred_delta.append(0.0 if delta_pred is None else float(delta_pred))
        pred_final.append(np.nan if final_pred is None else float(final_pred))
        pred_top1.append(np.nan if top1_pred is None else float(top1_pred))
    return {
        "pred_global": np.asarray(pred_global, dtype=np.float64),
        "pred_delta": np.asarray(pred_delta, dtype=np.float64),
        "pred_final": np.asarray(pred_final, dtype=np.float64),
        "pred_routed_top1": np.asarray(pred_top1, dtype=np.float64),
    }


def build_feature_matrix(gnn_pred, backbone_preds):
    gnn_pred = np.asarray(gnn_pred, dtype=np.float64).reshape(-1)
    pred_global = np.asarray(backbone_preds["pred_global"], dtype=np.float64).reshape(-1)
    pred_delta = np.asarray(backbone_preds["pred_delta"], dtype=np.float64).reshape(-1)
    pred_final = np.asarray(backbone_preds["pred_final"], dtype=np.float64).reshape(-1)
    pred_top1 = np.asarray(backbone_preds["pred_routed_top1"], dtype=np.float64).reshape(-1)

    pred_global = np.where(np.isfinite(pred_global), pred_global, pred_final)
    pred_final = np.where(np.isfinite(pred_final), pred_final, pred_global)
    pred_top1 = np.where(np.isfinite(pred_top1), pred_top1, pred_final)

    return np.column_stack(
        [
            gnn_pred,
            pred_global,
            pred_delta,
            pred_final,
            pred_top1,
            pred_final - gnn_pred,
        ]
    )
