import os

import core.train.train_v11_pro_meta_threehead_v50_mcmrt_transfer as adapter


WEIGHTS_ROOT = os.path.join(adapter.ROOT, "core", "models", "weights")
adapter.RUN_MODE = "train"
adapter.OUT_DIR = os.path.join(adapter.ROOT, "artifacts", "archive", "runs", "v47_mcmrt_adapter")
adapter.ROUTER_MANIFEST = os.path.join(adapter.OUT_DIR, "router_manifest_v47_mcmrt_adapter.json")
adapter.MANIFEST_PATH = adapter.ROUTER_MANIFEST
adapter.LEGACY_MANIFEST_PATH = os.path.join(adapter.OUT_DIR, "transfer_manifest_v47_mcmrt_adapter.json")
adapter.EVAL_CSV = os.path.join(adapter.OUT_DIR, "v47_mcmrt_adapter_eval_mcmrt10.csv")
adapter.EVAL_JSON = os.path.join(adapter.OUT_DIR, "v47_mcmrt_adapter_eval_mcmrt10.json")
adapter.BACKBONE_TRAIN_SCRIPT = os.path.join(adapter.ROOT, "core", "train", "train_v11_pro_meta_threehead_v47_calibrated_gate.py")
runtime_backbone_manifest = os.path.join(WEIGHTS_ROOT, "v47_main", "router_manifest_v47.json")
adapter.BACKBONE_ROUTER_MANIFEST = runtime_backbone_manifest if os.path.exists(runtime_backbone_manifest) else os.path.join(
    adapter.ROOT,
    "artifacts",
    "archive",
    "runs",
    "v11_pro_meta_threehead_v47_calibrated_gate",
    "router_manifest_v47.json",
)
adapter.USE_BACKBONE_TRAIN_AUGMENT = False

ROOT = adapter.ROOT
OUT_DIR = adapter.OUT_DIR
ROUTER_MANIFEST = adapter.ROUTER_MANIFEST
MANIFEST_PATH = adapter.MANIFEST_PATH
LEGACY_MANIFEST_PATH = adapter.LEGACY_MANIFEST_PATH
EVAL_CSV = adapter.EVAL_CSV
EVAL_JSON = adapter.EVAL_JSON
BACKBONE_TRAIN_SCRIPT = adapter.BACKBONE_TRAIN_SCRIPT
BACKBONE_ROUTER_MANIFEST = adapter.BACKBONE_ROUTER_MANIFEST


def build_predictor(manifest_path=None):
    return adapter.build_predictor(manifest_path)


def evaluate_manifest(manifest_path):
    return adapter.evaluate_manifest(manifest_path)


def eval_existing():
    return adapter.eval_existing()


if __name__ == "__main__":
    adapter.main()
