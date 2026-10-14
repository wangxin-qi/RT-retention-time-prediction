import json
import os
import sys
import importlib.util
from typing import Any, Dict, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from dgl import graph


def _resolve_manifest_path(manifest_dir: str, path_value: str):
    if not path_value:
        return path_value
    path_value = os.path.expanduser(str(path_value))
    candidate_bases = []
    current = os.path.abspath(manifest_dir)
    for _ in range(6):
        if current and current not in candidate_bases:
            candidate_bases.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    if os.path.isabs(path_value):
        resolved = os.path.abspath(path_value)
        if os.path.exists(resolved):
            return resolved
        return resolved
    seen = set()
    candidates = []
    for base_dir in candidate_bases:
        if not base_dir:
            continue
        resolved = os.path.abspath(os.path.join(base_dir, path_value))
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    for resolved in candidates:
        if os.path.exists(resolved):
            return resolved
    return candidates[1] if len(candidates) > 1 else candidates[0]


def _ensure_gnn_repo_on_syspath(gnn_repo: str) -> None:
    gnn_repo = os.path.abspath(gnn_repo)
    if gnn_repo not in sys.path:
        sys.path.insert(0, gnn_repo)


def _import_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SingleMolGraphFeaturizer:
    atom_list = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
    charge_list = [1, -1, 0]
    degree_list = [1, 2, 3, 4, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ["SP", "SP2", "SP3", "S"]
    hydrogen_list = [1, 2, 3, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )

    @classmethod
    def _find_stereo(cls, mol):
        si = Chem.FindPotentialStereo(mol)
        for element in si:
            if str(element.type) == "Atom_Tetrahedral" and str(element.specified) == "Specified":
                mol.GetAtomWithIdx(element.centeredOn).SetProp("Chirality", str(element.descriptor))
            elif str(element.type) == "Bond_Double" and str(element.specified) == "Specified":
                mol.GetBondWithIdx(element.centeredOn).SetProp("Stereochemistry", str(element.descriptor))

    @classmethod
    def _donor_acceptor_lists(cls, mol):
        donors = []
        acceptors = []
        for feat in cls.chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == "Donor":
                donors.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == "Acceptor":
                acceptors.append(feat.GetAtomIds()[0])
        return donors, acceptors

    @staticmethod
    def _chirality(atom):
        if atom.HasProp("Chirality"):
            return [
                atom.GetProp("Chirality") == "Tet_CW",
                atom.GetProp("Chirality") == "Tet_CCW",
            ]
        return [0, 0]

    @staticmethod
    def _stereochemistry(bond):
        if bond.HasProp("Stereochemistry"):
            return [
                bond.GetProp("Stereochemistry") == "Bond_Cis",
                bond.GetProp("Stereochemistry") == "Bond_Trans",
            ]
        return [0, 0]

    @classmethod
    def smiles_to_graph(cls, smiles: str):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"invalid_smiles: {smiles}")
        cls._find_stereo(mol)
        if "." in Chem.MolToSmiles(mol):
            raise ValueError(f"multi_component_smiles_not_supported: {smiles}")
        atom_symbols = {a.GetSymbol() for a in mol.GetAtoms()}
        if not atom_symbols.issubset(set(cls.atom_list)):
            raise ValueError(f"unsupported_atoms: {sorted(atom_symbols - set(cls.atom_list))}")

        mol = Chem.RemoveHs(mol)
        donors, acceptors = cls._donor_acceptor_lists(mol)

        atom_fea1 = np.eye(len(cls.atom_list), dtype=bool)[[cls.atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(cls.charge_list), dtype=bool)[[cls.charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:, :-1]
        atom_fea3 = np.eye(len(cls.degree_list), dtype=bool)[[cls.degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:, :-1]
        atom_fea4 = np.eye(len(cls.hybridization_list), dtype=bool)[[cls.hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:, :-1]
        atom_fea5 = np.eye(len(cls.hydrogen_list), dtype=bool)[[cls.hydrogen_list.index(a.GetTotalNumHs(includeNeighbors=True)) for a in mol.GetAtoms()]][:, :-1]
        atom_fea6 = np.eye(len(cls.valence_list), dtype=bool)[[cls.valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:, :-1]
        atom_fea7 = np.array([[(j in donors), (j in acceptors)] for j in range(mol.GetNumAtoms())], dtype=bool)
        atom_fea8 = np.array([cls._chirality(a) for a in mol.GetAtoms()], dtype=bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in cls.ringsize_list] for a in mol.GetAtoms()], dtype=bool)
        atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype=bool)
        node_attr = np.concatenate(
            [atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10],
            axis=1,
        )

        if mol.GetNumBonds() == 0:
            edge_attr = np.zeros((0, len(cls.bond_list) + 4), dtype=bool)
            src = np.asarray([], dtype=np.int64)
            dst = np.asarray([], dtype=np.int64)
        else:
            bond_fea1 = np.eye(len(cls.bond_list), dtype=bool)[[cls.bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([cls._stereochemistry(b) for b in mol.GetBonds()], dtype=bool)
            bond_fea3 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype=bool)
            edge_attr_single = np.concatenate([bond_fea1, bond_fea2, bond_fea3], axis=1)
            edge_attr = np.vstack([edge_attr_single, edge_attr_single])
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=np.int64)
            src = np.hstack([bond_loc[:, 0], bond_loc[:, 1]])
            dst = np.hstack([bond_loc[:, 1], bond_loc[:, 0]])

        g = graph((src, dst), num_nodes=mol.GetNumAtoms())
        node_tensor = torch.from_numpy(node_attr.astype(np.float32))
        edge_tensor = torch.from_numpy(edge_attr.astype(np.float32))
        return g, node_tensor, edge_tensor


class GnnMethodBundle:
    def __init__(self, dataset_name: str, model_path: str, gnn_repo: str, device: str = "cpu"):
        _ensure_gnn_repo_on_syspath(gnn_repo)
        cwd = os.getcwd()
        os.chdir(gnn_repo)
        try:
            dataset_mod = _import_module_from_path(
                f"v50_gnn_dataset_{dataset_name}",
                os.path.join(gnn_repo, "dataset.py"),
            )
            from gnn import GINPredictor

            GraphDataset = dataset_mod.GraphDataset
            trndata = GraphDataset(dataset_name, cv_id=0, split="trn", seed=134)
            self.train_mean = float(np.mean(trndata.label))
            self.train_std = float(np.std(trndata.label) + 1e-8)
            self.dataset_name = str(dataset_name)
            self.model_path = os.path.abspath(model_path)
            self.device = device

            self.model = GINPredictor(
                node_in_feats=trndata.node_attr.shape[1],
                edge_in_feats=trndata.edge_attr.shape[1],
            )
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            self.model.to(device).eval()
        finally:
            os.chdir(cwd)

    def predict(self, smiles: str) -> float:
        g, node_attr, edge_attr = SingleMolGraphFeaturizer.smiles_to_graph(smiles)
        g = g.to(self.device)
        node_attr = node_attr.to(self.device)
        edge_attr = edge_attr.to(self.device)
        with torch.no_grad():
            pred_z = self.model(g, node_attr, edge_attr).detach().cpu().numpy().reshape(-1)
        return float(pred_z[0] * self.train_std + self.train_mean)


class PrototypeRouterV50:
    def __init__(self, manifest: Dict[str, Any]):
        self.methods = manifest.get("methods", {})
        stats = manifest.get("router_stats", {})
        self.meta_mean = np.asarray(stats.get("meta_mean"), dtype=np.float32)
        self.meta_std = np.asarray(stats.get("meta_std"), dtype=np.float32)

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.meta_mean) / (self.meta_std + 1e-6)

    @staticmethod
    def _cos(a, b):
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

    def route(self, meta_vec):
        query = self._normalize(meta_vec)
        scored = []
        for method_name, info in self.methods.items():
            proto = np.asarray(info["prototype"], dtype=np.float32)
            proto = self._normalize(proto)
            cos = self._cos(query, proto)
            l1 = float(np.mean(np.abs(query - proto)))
            score = 0.8 * cos - 0.2 * l1
            scored.append(
                {
                    "method_name": str(method_name),
                    "dataset_name": str(info["dataset_name"]),
                    "score": float(score),
                    "cosine": float(cos),
                    "l1": float(l1),
                    "gnn_model_path": str(info["gnn_model_path"]),
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        top1 = scored[0] if scored else None
        return {
            "top1": top1,
            "candidates": scored[: min(5, len(scored))],
        }


class ResidualHybridHeadV50:
    def __init__(self, manifest: Dict[str, Any]):
        hybrid = manifest.get("hybrid", {})
        self.feature_names = list(hybrid.get("feature_names", []))
        self.feature_mean = np.asarray(hybrid.get("feature_mean", []), dtype=np.float32)
        self.feature_std = np.asarray(hybrid.get("feature_std", []), dtype=np.float32)
        self.weights = np.asarray(hybrid.get("weights", []), dtype=np.float32)
        self.bias = float(hybrid.get("bias", 0.0))

    def predict(self, feature_vec: np.ndarray, gnn_pred: float):
        x = np.asarray(feature_vec, dtype=np.float32)
        x_std = (x - self.feature_mean) / np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        residual = float(np.dot(x_std, self.weights) + self.bias)
        return float(gnn_pred + residual), residual


class MethodCalibratorV50:
    def __init__(self, manifest: Dict[str, Any]):
        self.methods = manifest.get("methods", {})

    def calibrate(self, method_name: str, pred: float):
        info = self.methods.get(str(method_name), {})
        cal = info.get("calibrator", {})
        slope = float(cal.get("slope", 1.0))
        intercept = float(cal.get("intercept", 0.0))
        return float(slope * pred + intercept), slope, intercept


class V50TransferPredictor:
    def __init__(
        self,
        manifest_path: str,
        backbone_predictor,
        smiles_to_fp,
        extract_method_meta_vector,
        gnn_repo: str,
        device: str = "cpu",
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
            self.manifest = json.load(f)
        for _, info in self.manifest.get("methods", {}).items():
            if isinstance(info, dict):
                info["gnn_model_path"] = _resolve_manifest_path(self.manifest_dir, info.get("gnn_model_path"))
        self.backbone_predictor = backbone_predictor
        self.smiles_to_fp = smiles_to_fp
        self.extract_method_meta_vector = extract_method_meta_vector
        self.gnn_repo = os.path.abspath(gnn_repo)
        self.device = device
        self.router = PrototypeRouterV50(self.manifest)
        self.hybrid = ResidualHybridHeadV50(self.manifest)
        self.calibrator = MethodCalibratorV50(self.manifest)
        self._gnn_bundles: Dict[str, GnnMethodBundle] = {}

    def _get_bundle(self, method_name: str) -> GnnMethodBundle:
        method_name = str(method_name)
        if method_name not in self._gnn_bundles:
            info = self.manifest["methods"][method_name]
            self._gnn_bundles[method_name] = GnnMethodBundle(
                dataset_name=info["dataset_name"],
                model_path=info["gnn_model_path"],
                gnn_repo=self.gnn_repo,
                device=self.device,
            )
        return self._gnn_bundles[method_name]

    @staticmethod
    def _build_feature_vec(gnn_pred: float, backbone_out: Dict[str, Any]) -> np.ndarray:
        pred_global = backbone_out.get("pred_global")
        pred_delta = backbone_out.get("pred_delta")
        pred_final = backbone_out.get("pred_final")
        pred_top1 = backbone_out.get("pred_routed_top1")

        pred_global = float(pred_final if pred_global is None else pred_global)
        pred_delta = float(0.0 if pred_delta is None else pred_delta)
        pred_final = float(pred_global if pred_final is None else pred_final)
        pred_top1 = float(pred_final if pred_top1 is None else pred_top1)
        return np.asarray(
            [
                float(gnn_pred),
                pred_global,
                pred_delta,
                pred_final,
                pred_top1,
                pred_final - float(gnn_pred),
            ],
            dtype=np.float32,
        )

    def predict(self, smiles_or_fp, method_meta=None, smiles: Optional[str] = None, meta_vec=None) -> Dict[str, Any]:
        if isinstance(smiles_or_fp, str):
            smiles = str(smiles_or_fp)
            fp = self.smiles_to_fp(smiles)
            if fp is None:
                return {"ok": False, "error": f"invalid_smiles: {smiles}"}
            if meta_vec is None:
                if method_meta is None:
                    raise ValueError("method_meta is required when calling predict(smiles, method_meta)")
                meta_vec = self.extract_method_meta_vector(method_meta)
        else:
            fp = np.asarray(smiles_or_fp, dtype=np.float32)
            if meta_vec is None:
                if method_meta is None:
                    raise ValueError("meta_vec or method_meta is required")
                meta_vec = np.asarray(method_meta, dtype=np.float32)
            if smiles is not None:
                smiles = str(smiles)

        meta_vec = np.asarray(meta_vec, dtype=np.float32)
        backbone_out = self.backbone_predictor.predict(fp, meta_vec)
        if not isinstance(backbone_out, dict):
            backbone_out = {"pred_final": float(backbone_out)}
        backbone_pred = float(backbone_out.get("pred_final", np.nan))

        if smiles is None:
            out = dict(backbone_out)
            out.update(
                {
                    "ok": True,
                    "pred_backbone": backbone_pred,
                    "pred_gnn": None,
                    "pred_v50_base": backbone_pred,
                    "pred_v50_residual": 0.0,
                    "pred_final": backbone_pred,
                    "v50_reason": "missing_smiles_fallback_to_backbone",
                }
            )
            return out

        route = self.router.route(meta_vec)
        top1 = route.get("top1")
        if top1 is None:
            out = dict(backbone_out)
            out.update(
                {
                    "ok": True,
                    "pred_backbone": backbone_pred,
                    "pred_gnn": None,
                    "pred_v50_base": backbone_pred,
                    "pred_v50_residual": 0.0,
                    "pred_final": backbone_pred,
                    "v50_reason": "no_gnn_route_fallback_to_backbone",
                }
            )
            return out

        try:
            gnn_pred = self._get_bundle(top1["method_name"]).predict(smiles)
            feature_vec = self._build_feature_vec(gnn_pred, backbone_out)
            pred_base, residual = self.hybrid.predict(feature_vec, gnn_pred)
            pred_final, slope, intercept = self.calibrator.calibrate(top1["method_name"], pred_base)
            out = dict(backbone_out)
            out.update(
                {
                    "ok": True,
                    "pred_backbone": backbone_pred,
                    "pred_gnn": float(gnn_pred),
                    "pred_v50_base": float(pred_base),
                    "pred_v50_residual": float(residual),
                    "pred_final": float(pred_final),
                    "v50_reason": "gnn_backbone_residual_plus_method_calibration",
                    "router_top1": top1["method_name"],
                    "router_candidates": route["candidates"],
                    "v50_feature_vec": feature_vec.tolist(),
                    "v50_hybrid_weights": self.hybrid.weights.tolist(),
                    "v50_hybrid_bias": self.hybrid.bias,
                    "v50_method_slope": float(slope),
                    "v50_method_intercept": float(intercept),
                }
            )
            return out
        except Exception as exc:
            out = dict(backbone_out)
            out.update(
                {
                    "ok": True,
                    "pred_backbone": backbone_pred,
                    "pred_gnn": None,
                    "pred_v50_base": backbone_pred,
                    "pred_v50_residual": 0.0,
                    "pred_final": backbone_pred,
                    "v50_reason": "gnn_failed_fallback_to_backbone",
                    "gnn_error": str(exc),
                    "router_top1": top1["method_name"],
                    "router_candidates": route["candidates"],
                }
            )
            return out
