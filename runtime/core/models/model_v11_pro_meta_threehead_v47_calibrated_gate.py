import os
import json
import math
import numpy as np
import torch
import torch.nn as nn

from core.data.dataset_v11_pro_meta_threehead_v47_calibrated_gate import (
    extract_route_signature,
    load_yaml_or_json,
)


_META_IDX_TEMPERATURE = 5
_META_IDX_PH_A = 6
_META_IDX_PH_B = 7
_META_IDX_B_ACN = 12
_META_IDX_B_MEOH = 13
_EXTRA_PHASE_OFFSET = 18 + 13


def _resolve_manifest_path(manifest_dir, path_value):
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

    def _candidate_suffixes(abs_path):
        normalized = os.path.abspath(abs_path).replace("/", "\\")
        lower = normalized.lower()
        suffixes = []
        for marker in (
            "data\\train\\processed_data\\",
            "data\\processed_data\\",
            "processed_data\\",
            "runs\\",
            "runs_",
            "external\\third_party\\",
            "third_party\\",
            "data\\mcmrt\\",
            "mcmrt\\",
            "artifacts\\validation_results\\",
            "validation_results\\",
        ):
            idx = lower.find(marker)
            if idx >= 0:
                suffixes.append(normalized[idx:])
        suffixes.append(os.path.basename(normalized))
        seen = set()
        ordered = []
        for item in suffixes:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    if os.path.isabs(path_value):
        resolved_abs = os.path.abspath(path_value)
        if os.path.exists(resolved_abs):
            return resolved_abs
        seen = set()
        for suffix in _candidate_suffixes(resolved_abs):
            for base_dir in candidate_bases:
                if not base_dir:
                    continue
                candidate = os.path.abspath(os.path.join(base_dir, suffix))
                if candidate in seen:
                    continue
                seen.add(candidate)
                if os.path.exists(candidate):
                    return candidate
        return resolved_abs
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


def _to_meta_buffer(value, meta_dim, fill):
    if value is None:
        return torch.full((meta_dim,), float(fill), dtype=torch.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(arr) != meta_dim:
        return torch.full((meta_dim,), float(fill), dtype=torch.float32)
    return torch.from_numpy(arr).float()


def _load_state_compat(model, state_dict):
    result = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in getattr(result, "missing_keys", []) if not key.endswith("num_batches_tracked")]
    unexpected = [key for key in getattr(result, "unexpected_keys", []) if not key.endswith("num_batches_tracked")]
    critical_prefixes = ("net.", "meta_tower.encoder.", "meta_tower.decoder.")
    critical_missing = [key for key in missing if key.startswith(critical_prefixes)]
    critical_unexpected = [key for key in unexpected if key.startswith(critical_prefixes)]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"missing={critical_missing[:12]} unexpected={critical_unexpected[:12]}"
        )


class MetaEmbeddingTower(nn.Module):
    def __init__(self, meta_dim, embed_dim=8, hidden_dim=32, meta_mean=None, meta_std=None):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.embed_dim = int(max(embed_dim, 0))
        self.hidden_dim = int(max(hidden_dim, 1))
        self.register_buffer('meta_mean', _to_meta_buffer(meta_mean, self.meta_dim, 0.0))
        self.register_buffer('meta_std', _to_meta_buffer(meta_std, self.meta_dim, 1.0))
        if self.embed_dim <= 0:
            self.encoder = None
            self.decoder = None
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.meta_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.embed_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.meta_dim),
            )

    def forward(self, meta):
        meta_norm = (meta - self.meta_mean) / (self.meta_std + 1e-6)
        if self.encoder is None:
            batch_size = int(meta_norm.shape[0])
            embed = meta_norm.new_zeros((batch_size, 0))
            recon = meta_norm
        else:
            embed = self.encoder(meta_norm)
            recon = self.decoder(embed)
        return meta_norm, embed, recon


class GlobalMLP(nn.Module):
    def __init__(
        self,
        fp_dim=1024,
        meta_dim=18,
        hidden=384,
        dropout=0.2,
        meta_mean=None,
        meta_std=None,
        meta_embed_dim=0,
        meta_encoder_hidden=32,
    ):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.meta_embed_dim = int(max(meta_embed_dim, 0))
        self.meta_tower = MetaEmbeddingTower(
            meta_dim=self.meta_dim,
            embed_dim=self.meta_embed_dim,
            hidden_dim=meta_encoder_hidden,
            meta_mean=meta_mean,
            meta_std=meta_std,
        )
        self.net = nn.Sequential(
            nn.Linear(fp_dim + meta_dim + self.meta_embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, fp, meta, return_aux=False):
        meta_norm, meta_embed, meta_recon = self.meta_tower(meta)
        pred = self.net(torch.cat([fp, meta_norm, meta_embed], dim=1)).squeeze(1)
        if return_aux:
            return pred, {
                'meta_norm': meta_norm,
                'meta_embed': meta_embed,
                'meta_recon': meta_recon,
            }
        return pred


class ExpertDeltaMLP(nn.Module):
    def __init__(
        self,
        fp_dim=1024,
        meta_dim=18,
        hidden=256,
        dropout=0.2,
        meta_mean=None,
        meta_std=None,
        meta_embed_dim=0,
        meta_encoder_hidden=32,
    ):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.meta_embed_dim = int(max(meta_embed_dim, 0))
        self.meta_tower = MetaEmbeddingTower(
            meta_dim=self.meta_dim,
            embed_dim=self.meta_embed_dim,
            hidden_dim=meta_encoder_hidden,
            meta_mean=meta_mean,
            meta_std=meta_std,
        )
        self.net = nn.Sequential(
            nn.Linear(fp_dim + meta_dim + self.meta_embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, fp, meta, return_aux=False):
        meta_norm, meta_embed, meta_recon = self.meta_tower(meta)
        pred = self.net(torch.cat([fp, meta_norm, meta_embed], dim=1)).squeeze(1)
        if return_aux:
            return pred, {
                'meta_norm': meta_norm,
                'meta_embed': meta_embed,
                'meta_recon': meta_recon,
            }
        return pred


class GlobalBundle:
    def __init__(self, ckpt_path, device='cpu'):
        blob = torch.load(ckpt_path, map_location=device)
        meta = blob.get('meta', {})
        self.rt_mean = float(meta.get('rt_mean', 0.0))
        self.rt_std = float(meta.get('rt_std', 1.0))
        fp_dim = int(meta.get('fp_dim', 1024))
        meta_dim = int(meta.get('meta_dim', 18))
        hidden = int(meta.get('hidden', 384))
        dropout = float(meta.get('dropout', 0.2))
        meta_embed_dim = int(meta.get('meta_embed_dim', 0))
        meta_encoder_hidden = int(meta.get('meta_encoder_hidden', 32))
        self.model = GlobalMLP(
            fp_dim=fp_dim,
            meta_dim=meta_dim,
            hidden=hidden,
            dropout=dropout,
            meta_mean=meta.get('meta_mean'),
            meta_std=meta.get('meta_std'),
            meta_embed_dim=meta_embed_dim,
            meta_encoder_hidden=meta_encoder_hidden,
        )
        _load_state_compat(self.model, blob['model_state'])
        self.model.to(device).eval()
        self.device = device
        self.fp_dim = fp_dim
        self.meta_dim = meta_dim

    def predict_array(self, fps_np, metas_np, batch_size=1024):
        fps_np = np.asarray(fps_np, dtype=np.float32)[:, :self.fp_dim]
        metas_np = np.asarray(metas_np, dtype=np.float32)[:, :self.meta_dim]
        outs = []
        with torch.no_grad():
            for i in range(0, len(fps_np), batch_size):
                fp = torch.from_numpy(fps_np[i:i + batch_size]).float().to(self.device)
                meta = torch.from_numpy(metas_np[i:i + batch_size]).float().to(self.device)
                pred_z = self.model(fp, meta).detach().cpu().numpy()
                outs.append(pred_z * self.rt_std + self.rt_mean)
        return np.concatenate(outs, axis=0).astype(np.float32)

    def predict(self, fp_np, meta_np):
        return float(self.predict_array(
            np.asarray(fp_np, dtype=np.float32)[None, :],
            np.asarray(meta_np, dtype=np.float32)[None, :]
        )[0])


class FamilyHead(nn.Module):
    def __init__(
        self,
        fp_dim=1024,
        meta_dim=18,
        hidden=128,
        num_families=20,
        dropout=0.2,
        meta_mean=None,
        meta_std=None,
        meta_embed_dim=0,
        meta_encoder_hidden=32,
    ):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.num_families = int(num_families)
        self.meta_embed_dim = int(max(meta_embed_dim, 0))
        self.meta_tower = MetaEmbeddingTower(
            meta_dim=self.meta_dim,
            embed_dim=self.meta_embed_dim,
            hidden_dim=meta_encoder_hidden,
            meta_mean=meta_mean,
            meta_std=meta_std,
        )
        self.net = nn.Sequential(
            nn.Linear(fp_dim + meta_dim + self.meta_embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_families),
        )

    def forward(self, fp, meta, return_aux=False):
        meta_norm, meta_embed, meta_recon = self.meta_tower(meta)
        logits = self.net(torch.cat([fp, meta_norm, meta_embed], dim=1))
        if return_aux:
            return logits, {
                'meta_norm': meta_norm,
                'meta_embed': meta_embed,
                'meta_recon': meta_recon,
            }
        return logits


class FamilyHeadBundle:
    def __init__(self, ckpt_path, device='cpu'):
        blob = torch.load(ckpt_path, map_location=device)
        meta = blob.get('meta', {})
        fp_dim = int(meta.get('fp_dim', 1024))
        meta_dim = int(meta.get('meta_dim', 45))
        hidden = int(meta.get('hidden', 128))
        dropout = float(meta.get('dropout', 0.2))
        num_families = int(meta.get('num_families', 0))
        self.family_index = dict(meta.get('family_index', {}))
        meta_embed_dim = int(meta.get('meta_embed_dim', 0))
        meta_encoder_hidden = int(meta.get('meta_encoder_hidden', 32))
        self.model = FamilyHead(
            fp_dim=fp_dim,
            meta_dim=meta_dim,
            hidden=hidden,
            num_families=num_families,
            dropout=dropout,
            meta_mean=meta.get('meta_mean'),
            meta_std=meta.get('meta_std'),
            meta_embed_dim=meta_embed_dim,
            meta_encoder_hidden=meta_encoder_hidden,
        )
        _load_state_compat(self.model, blob['model_state'])
        self.model.to(device).eval()
        self.device = device
        self.fp_dim = fp_dim
        self.meta_dim = meta_dim
        self.num_families = num_families

    def predict_probs(self, fp_np, meta_np):
        fp = torch.from_numpy(np.asarray(fp_np, dtype=np.float32)[None, :self.fp_dim]).float().to(self.device)
        meta = torch.from_numpy(np.asarray(meta_np, dtype=np.float32)[None, :self.meta_dim]).float().to(self.device)
        with torch.no_grad():
            logits = self.model(fp, meta).detach().cpu().numpy()[0]
        e = np.exp(logits - np.max(logits))
        return (e / (e.sum() + 1e-8)).astype(np.float32)


class ExpertDeltaBundle:
    def __init__(self, ckpt_path, device='cpu'):
        blob = torch.load(ckpt_path, map_location=device)
        meta = blob.get('meta', {})
        self.method_code = str(meta.get('method_code', os.path.basename(os.path.dirname(ckpt_path)).replace('expert_', '')))
        self.family_id = str(meta.get('family_id', self.method_code))
        self.delta_mean = float(meta.get('delta_mean', 0.0))
        self.delta_std = float(meta.get('delta_std', 1.0))
        self.best_val_mae = float(meta.get('best_val_mae', 1e9))
        fp_dim = int(meta.get('fp_dim', 1024))
        meta_dim = int(meta.get('meta_dim', 18))
        hidden = int(meta.get('hidden', 256))
        dropout = float(meta.get('dropout', 0.2))
        meta_embed_dim = int(meta.get('meta_embed_dim', 0))
        meta_encoder_hidden = int(meta.get('meta_encoder_hidden', 32))
        self.model = ExpertDeltaMLP(
            fp_dim=fp_dim,
            meta_dim=meta_dim,
            hidden=hidden,
            dropout=dropout,
            meta_mean=meta.get('meta_mean'),
            meta_std=meta.get('meta_std'),
            meta_embed_dim=meta_embed_dim,
            meta_encoder_hidden=meta_encoder_hidden,
        )
        _load_state_compat(self.model, blob['model_state'])
        self.model.to(device).eval()
        self.device = device
        self.fp_dim = fp_dim
        self.meta_dim = meta_dim

    def predict_delta_array(self, fps_np, metas_np, batch_size=1024):
        fps_np = np.asarray(fps_np, dtype=np.float32)[:, :self.fp_dim]
        metas_np = np.asarray(metas_np, dtype=np.float32)[:, :self.meta_dim]
        outs = []
        with torch.no_grad():
            for i in range(0, len(fps_np), batch_size):
                fp = torch.from_numpy(fps_np[i:i + batch_size]).float().to(self.device)
                meta = torch.from_numpy(metas_np[i:i + batch_size]).float().to(self.device)
                pred_z = self.model(fp, meta).detach().cpu().numpy()
                outs.append(pred_z * self.delta_std + self.delta_mean)
        return np.concatenate(outs, axis=0).astype(np.float32)

    def predict_delta(self, fp_np, meta_np):
        return float(self.predict_delta_array(
            np.asarray(fp_np, dtype=np.float32)[None, :],
            np.asarray(meta_np, dtype=np.float32)[None, :]
        )[0])


def _sigmoid_gate(x, threshold, scale):
    return 1.0 / (1.0 + math.exp(-(float(x) - float(threshold)) * float(scale)))


def _safe_meta_value(vec, idx):
    if idx < len(vec):
        return float(vec[idx])
    return 0.0


def _effective_ph_from_meta(vec):
    ph_b = _safe_meta_value(vec, _META_IDX_PH_B)
    ph_a = _safe_meta_value(vec, _META_IDX_PH_A)
    if ph_b > 0:
        return ph_b
    if ph_a > 0:
        return ph_a
    return None


def _phase_label_from_meta(vec):
    if len(vec) < (_EXTRA_PHASE_OFFSET + 4):
        return "unknown"
    c18 = _safe_meta_value(vec, _EXTRA_PHASE_OFFSET + 0) > 0.5
    hss_t3 = _safe_meta_value(vec, _EXTRA_PHASE_OFFSET + 1) > 0.5
    hilic = _safe_meta_value(vec, _EXTRA_PHASE_OFFSET + 2) > 0.5
    phenyl = _safe_meta_value(vec, _EXTRA_PHASE_OFFSET + 3) > 0.5
    if hss_t3:
        return "hss_t3"
    if hilic:
        return "hilic"
    if phenyl:
        return "phenyl"
    if c18:
        return "c18"
    return "unknown"


def _solvent_label_from_meta(vec):
    b_acn = _safe_meta_value(vec, _META_IDX_B_ACN) > 0.5
    b_meoh = _safe_meta_value(vec, _META_IDX_B_MEOH) > 0.5
    if b_acn and not b_meoh:
        return "acn"
    if b_meoh and not b_acn:
        return "meoh"
    if b_acn or b_meoh:
        return "mixed"
    return "unknown"


def _phase_compatible(query_phase, expert_phase):
    if query_phase == "unknown" or expert_phase == "unknown":
        return False
    rp_like = {"c18", "hss_t3"}
    if query_phase in rp_like and expert_phase in rp_like:
        return True
    return query_phase == expert_phase


class CalibratedQualityFamilyRouter:
    def __init__(
        self,
        manifest_path,
        topk=3,
        temperature=0.8,
        alpha_max=0.95,
        quality_lambda=0.35,
        min_family_top1=0.52,
        min_family_margin=0.10,
        min_expert_margin=0.05,
        min_top_expert_weight=0.28,
        route_signature_lambda=0.35,
        high_family_confidence=0.78,
        medium_family_confidence=0.60,
        soft_gate=True,
        soft_gate_min=0.20,
        family_head_lambda=0.0,
        ph_hard_window=1.25,
        temperature_hard_window=15.0,
        enforce_phase_boundary=True,
        enforce_solvent_boundary=True,
        similarity_alpha_lo=0.20,
        similarity_alpha_hi=0.85,
        condition_prior_lambda=1.0,
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        with open(self.manifest_path, 'r', encoding='utf-8-sig') as f:
            self.manifest = json.load(f)
        self.manifest['global_ckpt_path'] = _resolve_manifest_path(
            self.manifest_dir,
            self.manifest.get('global_ckpt_path'),
        )
        self.manifest['search_root'] = _resolve_manifest_path(
            self.manifest_dir,
            self.manifest.get('search_root'),
        )
        for _, info in self.manifest.get('experts', {}).items():
            if isinstance(info, dict):
                info['ckpt_path'] = _resolve_manifest_path(self.manifest_dir, info.get('ckpt_path'))
                info['metadata_path'] = _resolve_manifest_path(self.manifest_dir, info.get('metadata_path'))

        self.topk = int(topk)
        self.temperature = float(temperature)
        self.alpha_max = float(alpha_max)
        self.quality_lambda = float(quality_lambda)
        self.min_family_top1 = float(min_family_top1)
        self.min_family_margin = float(min_family_margin)
        self.min_expert_margin = float(min_expert_margin)
        self.min_top_expert_weight = float(min_top_expert_weight)
        self.route_signature_lambda = float(route_signature_lambda)
        self.high_family_confidence = float(high_family_confidence)
        self.medium_family_confidence = float(medium_family_confidence)
        self.soft_gate = bool(soft_gate)
        self.soft_gate_min = float(soft_gate_min)
        self.family_head_lambda = float(family_head_lambda)
        self.ph_hard_window = float(ph_hard_window)
        self.temperature_hard_window = float(temperature_hard_window)
        self.enforce_phase_boundary = bool(enforce_phase_boundary)
        self.enforce_solvent_boundary = bool(enforce_solvent_boundary)
        # Soft condition-similarity -> alpha ramp. A similarity at/below lo gives alpha 0
        # (fall back to global); at/above hi gives full trust. Replaces the old
        # "looks similar -> treat as explicit method code -> alpha = 1.0" behaviour.
        self.similarity_alpha_lo = float(similarity_alpha_lo)
        self.similarity_alpha_hi = float(similarity_alpha_hi)
        self.condition_prior_lambda = float(condition_prior_lambda)
        self.family_index = self.manifest.get('family_index', {}) or {}

        self.experts = self.manifest.get('experts', {})
        self.protos = {
            str(k): np.asarray(v['prototype'], dtype=np.float32)
            for k, v in self.experts.items()
            if v.get('prototype') is not None
        }
        self.route_protos = {}
        for code, info in self.experts.items():
            route_signature = info.get('route_signature')
            if route_signature is not None:
                self.route_protos[str(code)] = np.asarray(route_signature, dtype=np.float32)
                continue
            metadata_path = info.get('metadata_path')
            if metadata_path and os.path.exists(metadata_path):
                try:
                    meta_dict = load_yaml_or_json(metadata_path)
                    method_dir = os.path.dirname(metadata_path)
                    inferred_method_code = os.path.basename(method_dir)
                    self.route_protos[str(code)] = extract_route_signature(
                        meta_dict,
                        method_dir=method_dir,
                        method_code=inferred_method_code if inferred_method_code.isdigit() else str(code),
                    )
                except Exception:
                    pass
        self.family_of = {str(k): str(v.get('family_id', k)) for k, v in self.experts.items()}
        self.route_collision_groups = {}
        collision_buckets = {}
        for code, proto in self.route_protos.items():
            key = tuple(np.round(np.asarray(proto, dtype=np.float32), 6).tolist())
            collision_buckets.setdefault(key, []).append(str(code))
        for _, codes in collision_buckets.items():
            if len(codes) > 1:
                self.route_collision_groups[tuple(sorted(codes))] = sorted(codes)

        router_stats = self.manifest.get('router_stats', {})
        mean = router_stats.get('meta_mean')
        std = router_stats.get('meta_std')
        if mean is None or std is None:
            arr = np.stack(list(self.protos.values()), axis=0)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0) + 1e-6
        self.meta_mean = np.asarray(mean, dtype=np.float32)
        self.meta_std = np.asarray(std, dtype=np.float32)

        route_mean = router_stats.get('route_mean')
        route_std = router_stats.get('route_std')
        if route_mean is not None and route_std is not None:
            self.route_mean = np.asarray(route_mean, dtype=np.float32)
            self.route_std = np.asarray(route_std, dtype=np.float32)
        elif self.route_protos:
            route_arr = np.stack(list(self.route_protos.values()), axis=0)
            self.route_mean = np.mean(route_arr, axis=0).astype(np.float32)
            self.route_std = (np.std(route_arr, axis=0) + 1e-6).astype(np.float32)
        else:
            self.route_mean = None
            self.route_std = None

        maes = []
        for _, info in self.experts.items():
            try:
                maes.append(float(info.get('best_val_mae', 1e9)))
            except Exception:
                pass
        if maes:
            arr = np.asarray(maes, dtype=np.float32)
            self.mae_lo = float(np.min(arr))
            self.mae_hi = float(np.max(arr))
        else:
            self.mae_lo = 0.0
            self.mae_hi = 1.0

    def update_runtime_config(
        self,
        *,
        topk=None,
        temperature=None,
        alpha_max=None,
        quality_lambda=None,
        min_family_top1=None,
        min_family_margin=None,
        min_expert_margin=None,
        min_top_expert_weight=None,
        soft_gate=None,
        soft_gate_min=None,
        family_head_lambda=None,
        ph_hard_window=None,
        temperature_hard_window=None,
        enforce_phase_boundary=None,
        enforce_solvent_boundary=None,
        condition_prior_lambda=None,
    ):
        if topk is not None:
            self.topk = int(topk)
        if temperature is not None:
            self.temperature = float(temperature)
        if alpha_max is not None:
            self.alpha_max = float(alpha_max)
        if quality_lambda is not None:
            self.quality_lambda = float(quality_lambda)
        if min_family_top1 is not None:
            self.min_family_top1 = float(min_family_top1)
        if min_family_margin is not None:
            self.min_family_margin = float(min_family_margin)
        if min_expert_margin is not None:
            self.min_expert_margin = float(min_expert_margin)
        if min_top_expert_weight is not None:
            self.min_top_expert_weight = float(min_top_expert_weight)
        if soft_gate is not None:
            self.soft_gate = bool(soft_gate)
        if soft_gate_min is not None:
            self.soft_gate_min = float(soft_gate_min)
        if family_head_lambda is not None:
            self.family_head_lambda = float(family_head_lambda)
        if ph_hard_window is not None:
            self.ph_hard_window = float(ph_hard_window)
        if temperature_hard_window is not None:
            self.temperature_hard_window = float(temperature_hard_window)
        if enforce_phase_boundary is not None:
            self.enforce_phase_boundary = bool(enforce_phase_boundary)
        if enforce_solvent_boundary is not None:
            self.enforce_solvent_boundary = bool(enforce_solvent_boundary)
        if condition_prior_lambda is not None:
            self.condition_prior_lambda = float(condition_prior_lambda)

    def _normalize_meta(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.meta_mean) / (self.meta_std + 1e-6)

    def _cos(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))

    def _quality_prior(self, code):
        info = self.experts.get(str(code), {})
        mae = float(info.get('best_val_mae', self.mae_hi + 1.0))
        if self.mae_hi <= self.mae_lo + 1e-8:
            return 0.5
        q = 1.0 - ((mae - self.mae_lo) / (self.mae_hi - self.mae_lo + 1e-8))
        return float(max(0.0, min(1.0, q)))

    def _normalize_route_signature(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.route_mean is None or self.route_std is None:
            return x
        if len(x) != len(self.route_mean):
            return x
        return (x - self.route_mean) / (self.route_std + 1e-6)

    def _sim_score(self, query_meta, proto_meta, query_route_signature=None, proto_route_signature=None):
        q = self._normalize_meta(query_meta)
        p = self._normalize_meta(proto_meta)
        cos = self._cos(q, p)
        l1 = float(np.mean(np.abs(q - p)))
        score = 0.75 * cos - 0.25 * l1
        for idx, bonus in [(10, 0.06), (11, 0.06), (12, 0.03), (13, 0.03), (14, 0.03)]:
            if idx < len(q) and idx < len(p) and abs(q[idx] - p[idx]) < 0.25:
                score += bonus
        if query_route_signature is not None and proto_route_signature is not None:
            qr = np.asarray(query_route_signature, dtype=np.float32)
            pr = np.asarray(proto_route_signature, dtype=np.float32)
            if len(qr) == len(pr):
                qr = self._normalize_route_signature(qr)
                pr = self._normalize_route_signature(pr)
                route_cos = self._cos(qr, pr)
                route_l1 = float(np.mean(np.abs(qr - pr)))
                route_score = 0.70 * route_cos - 0.30 * route_l1
                score = (1.0 - self.route_signature_lambda) * score + self.route_signature_lambda * route_score
        return float(score)

    def _softmax(self, scores, temperature=None):
        scores = np.asarray(scores, dtype=np.float32)
        temperature = self.temperature if temperature is None else float(temperature)
        if len(scores) == 0:
            return np.zeros((0,), dtype=np.float32)
        scores = scores - np.max(scores)
        w = np.exp(scores / max(temperature, 1e-4))
        w = w / (np.sum(w) + 1e-8)
        return w.astype(np.float32)

    def _domain_gate(self, query_meta, proto_meta):
        query_phase = _phase_label_from_meta(query_meta)
        expert_phase = _phase_label_from_meta(proto_meta)
        query_solvent = _solvent_label_from_meta(query_meta)
        expert_solvent = _solvent_label_from_meta(proto_meta)
        query_ph = _effective_ph_from_meta(query_meta)
        expert_ph = _effective_ph_from_meta(proto_meta)
        query_temp = _safe_meta_value(query_meta, _META_IDX_TEMPERATURE)
        expert_temp = _safe_meta_value(proto_meta, _META_IDX_TEMPERATURE)

        reasons = []
        if self.enforce_phase_boundary and not _phase_compatible(query_phase, expert_phase):
            reasons.append('phase_mismatch')

        if self.enforce_solvent_boundary:
            if query_solvent not in {'unknown', 'mixed'} and expert_solvent not in {'unknown', 'mixed'}:
                if query_solvent != expert_solvent:
                    reasons.append('solvent_mismatch')

        if query_ph is not None and expert_ph is not None:
            if abs(float(query_ph) - float(expert_ph)) > self.ph_hard_window:
                reasons.append('ph_mismatch')

        if query_temp > 0 and expert_temp > 0:
            if abs(float(query_temp) - float(expert_temp)) > self.temperature_hard_window:
                reasons.append('temperature_mismatch')

        return len(reasons) == 0, reasons

    def _similarity_gate(self, route_similarity):
        """Map a condition-weighted similarity in (0,1] to an alpha-scaling g in [0,1]."""
        if route_similarity is None:
            return 1.0
        sim = float(route_similarity)
        lo, hi = self.similarity_alpha_lo, self.similarity_alpha_hi
        if hi <= lo:
            return 1.0 if sim >= hi else 0.0
        return float(max(0.0, min(1.0, (sim - lo) / (hi - lo))))

    def route(
        self,
        meta_vec,
        method_code_hint=None,
        route_signature=None,
        family_probs=None,
        route_similarity=None,
        soft_method_prior=None,
    ):
        meta_vec = np.asarray(meta_vec, dtype=np.float32)
        family_probs_arr = None
        if family_probs is not None and self.family_index:
            family_probs_arr = np.asarray(family_probs, dtype=np.float32).reshape(-1)
        soft_prior = {}
        if soft_method_prior:
            items = soft_method_prior.items() if isinstance(soft_method_prior, dict) else soft_method_prior
            for item in items:
                try:
                    if isinstance(item, dict):
                        code = str(item.get('method_code') or item.get('expert') or '').strip()
                        value = float(item.get('score', item.get('weight', 0.0)))
                    else:
                        code = str(item[0]).strip()
                        value = float(item[1])
                    if code and np.isfinite(value) and value > 0:
                        soft_prior[code] = max(soft_prior.get(code, 0.0), float(value))
                except Exception:
                    continue
        soft_prior_active = bool(soft_prior)
        soft_prior_max = max(soft_prior.values()) if soft_prior else None
        if method_code_hint is None and route_signature is not None:
            route_key = tuple(np.round(np.asarray(route_signature, dtype=np.float32), 6).tolist())
            for codes in self.route_collision_groups.values():
                first_proto = self.route_protos.get(codes[0])
                if first_proto is None:
                    continue
                proto_key = tuple(np.round(np.asarray(first_proto, dtype=np.float32), 6).tolist())
                if route_key == proto_key:
                    return {
                        'mode': 'global_only',
                        'experts': [],
                        'families': sorted({self.family_of.get(code, code) for code in codes}),
                        'weights': {},
                        'family_weights': {},
                        'scores': [],
                        'alpha_base': 0.0,
                        'router_gate_pass': False,
                        'gate_strength': 0.0,
                        'reason': 'ambiguous_route_signature',
                        'collision_candidates': list(codes),
                        'confidence_tier': 'low',
                        'family_confidence': 0.0,
                        'correction_cap': 0.0,
                        'calibration': {
                            'collision_candidates': list(codes),
                            'family_top1': 0.0,
                            'family_top2': 0.0,
                            'family_margin': 0.0,
                            'family_entropy': 0.0,
                            'top_expert_weight': 0.0,
                            'top_expert': None,
                            'top_score': 0.0,
                            'second_score': 0.0,
                            'expert_margin': 0.0,
                            'checks': {
                                'family_top1_pass': False,
                                'family_margin_pass': False,
                                'expert_margin_pass': False,
                                'top_expert_weight_pass': False,
                            },
                        },
                    }
        scored = []
        filtered_out = []
        for code, proto in self.protos.items():
            eligible, reject_reasons = self._domain_gate(meta_vec, proto)
            if not eligible:
                filtered_out.append({
                    'expert': str(code),
                    'family_id': self.family_of.get(str(code), str(code)),
                    'reject_reasons': list(reject_reasons),
                })
                continue
            sim = self._sim_score(
                meta_vec,
                proto,
                query_route_signature=route_signature,
                proto_route_signature=self.route_protos.get(str(code)),
            )
            q = self._quality_prior(code)
            condition_prior = float(soft_prior.get(str(code), 0.0))
            total = sim + self.quality_lambda * q
            if soft_prior_active:
                total += self.condition_prior_lambda * condition_prior
            scored.append({
                'expert': str(code),
                'family_id': self.family_of.get(str(code), str(code)),
                'sim_score': float(sim),
                'quality': float(q),
                'condition_prior': float(condition_prior),
                'score': float(total),
            })
            scored.sort(key=lambda x: x['score'], reverse=True)

        if not scored:
            if filtered_out:
                return {
                    'mode': 'global_only',
                    'experts': [],
                    'families': sorted({row['family_id'] for row in filtered_out}),
                    'weights': {},
                    'family_weights': {},
                    'scores': [],
                    'alpha_base': 0.0,
                    'router_gate_pass': False,
                    'gate_strength': 0.0,
                    'reason': 'no_eligible_experts',
                    'collision_candidates': [],
                    'confidence_tier': 'low',
                    'family_confidence': 0.0,
                    'correction_cap': 0.0,
                    'domain_filter': {
                        'eligible_count': 0,
                        'filtered_count': len(filtered_out),
                        'filtered_examples': filtered_out[:8],
                    },
                    'calibration': {},
                }
            return {
                'mode': 'global_only',
                'experts': [],
                'families': [],
                'weights': {},
                'family_weights': {},
                'scores': [],
                'alpha_base': 0.0,
                'router_gate_pass': False,
                'gate_strength': 0.0,
                'reason': 'no_experts',
                'collision_candidates': [],
                'domain_filter': {
                    'eligible_count': 0,
                    'filtered_count': 0,
                    'filtered_examples': [],
                },
                'calibration': {},
            }

        family_members = {}
        for row in scored:
            family_members.setdefault(row['family_id'], []).append(row)

        family_rows = []
        for family_id, members in family_members.items():
            best_score = max(m['score'] for m in members)
            mean_score = float(np.mean([m['score'] for m in members]))
            family_rows.append({
                'family_id': family_id,
                'score': float(0.7 * best_score + 0.3 * mean_score),
                'n_members': len(members),
            })
        family_rows.sort(key=lambda x: x['score'], reverse=True)

        picked_families = family_rows[:max(1, self.topk)]
        fam_scores = [x['score'] for x in picked_families]
        fam_w_proto = self._softmax(fam_scores)
        fam_w = fam_w_proto.copy()
        family_head_used = False
        if family_probs_arr is not None and len(family_probs_arr) > 0:
            head_w_picked = np.zeros_like(fam_w_proto)
            any_hit = False
            for i, fam in enumerate(picked_families):
                idx = self.family_index.get(fam['family_id'])
                if idx is None:
                    continue
                idx = int(idx)
                if 0 <= idx < len(family_probs_arr):
                    head_w_picked[i] = float(family_probs_arr[idx])
                    any_hit = True
            if any_hit and head_w_picked.sum() > 1e-8:
                head_w_picked = head_w_picked / head_w_picked.sum()
                lam = max(0.0, min(1.0, self.family_head_lambda))
                fam_w = (1.0 - lam) * fam_w_proto + lam * head_w_picked
                fam_w = fam_w / (fam_w.sum() + 1e-8)
                family_head_used = True
        family_weights = {picked_families[i]['family_id']: float(fam_w[i]) for i in range(len(picked_families))}

        weights = {}
        experts = []
        ordered_experts = []
        for fam in picked_families:
            family_id = fam['family_id']
            members = family_members[family_id]
            member_scores = [m['score'] for m in members]
            member_w = self._softmax(member_scores, temperature=max(self.temperature * 0.8, 0.15))
            for i, m in enumerate(members):
                w = float(family_weights[family_id] * member_w[i])
                weights[m['expert']] = w
                experts.append(m['expert'])
                ordered_experts.append((m, w))
        ordered_experts.sort(key=lambda x: x[1], reverse=True)
        experts = [x[0]['expert'] for x in ordered_experts]

        fam_top1 = float(fam_w[0]) if len(fam_w) > 0 else 0.0
        fam_top2 = float(fam_w[1]) if len(fam_w) > 1 else 0.0
        fam_margin = max(0.0, fam_top1 - fam_top2)
        fam_entropy = 0.0
        if len(fam_w) > 1:
            fam_entropy = -float(np.sum(fam_w * np.log(fam_w + 1e-8)) / math.log(len(fam_w)))

        top_score = float(scored[0]['score']) if len(scored) > 0 else 0.0
        second_score = float(scored[1]['score']) if len(scored) > 1 else top_score
        expert_margin = max(0.0, top_score - second_score)
        top_expert_weight = float(ordered_experts[0][1]) if len(ordered_experts) > 0 else 0.0
        top_expert = ordered_experts[0][0]['expert'] if len(ordered_experts) > 0 else None

        confidence_raw = 0.40 * fam_top1 + 0.25 * fam_margin + 0.15 * top_expert_weight + 0.20 * (1.0 - fam_entropy)
        alpha_base = self.alpha_max * max(0.0, min(1.0, confidence_raw))
        family_confidence = float(0.65 * fam_top1 + 0.35 * fam_margin)
        if family_confidence >= self.high_family_confidence:
            confidence_tier = 'high'
            correction_cap = 1.0
        elif family_confidence >= self.medium_family_confidence:
            confidence_tier = 'medium'
            correction_cap = 0.45
        else:
            confidence_tier = 'low'
            correction_cap = 0.0

        gate_checks = {
            'family_top1_pass': fam_top1 >= self.min_family_top1,
            'family_margin_pass': fam_margin >= self.min_family_margin,
            'expert_margin_pass': expert_margin >= self.min_expert_margin,
            'top_expert_weight_pass': top_expert_weight >= self.min_top_expert_weight,
        }
        pass_count = sum(int(v) for v in gate_checks.values())

        if self.soft_gate:
            # 软门：四项 sigmoid 乘积；阈值取实际同集分布的 ~p30，scale 调到 sigmoid 在 ±0.1 内基本饱和
            s_fam_top1 = _sigmoid_gate(fam_top1, 0.40, 12.0)
            s_fam_margin = _sigmoid_gate(fam_margin, 0.05, 20.0)
            s_expert_margin = _sigmoid_gate(expert_margin, 0.02, 30.0)
            s_top_expert_weight = _sigmoid_gate(top_expert_weight, 0.08, 25.0)
            gate_soft = float(s_fam_top1 * s_fam_margin * s_expert_margin * s_top_expert_weight)
            router_gate_pass = gate_soft >= self.soft_gate_min
            gate_strength = gate_soft
            alpha_base = float(self.alpha_max * max(0.0, min(1.0, confidence_raw)) * gate_soft)
        else:
            strict_pass = pass_count == len(gate_checks)
            medium_pass = bool(gate_checks['family_top1_pass'] and gate_checks['top_expert_weight_pass'])
            if confidence_tier == 'high':
                router_gate_pass = strict_pass or medium_pass
            elif confidence_tier == 'medium':
                router_gate_pass = medium_pass
            else:
                router_gate_pass = False
            gate_strength = float(pass_count / max(len(gate_checks), 1))
            if not router_gate_pass:
                alpha_base = 0.0

        soft_condition_gate = None
        if soft_prior_active:
            soft_condition_gate = self._similarity_gate(route_similarity)
            alpha_base = float(alpha_base * soft_condition_gate)
            gate_strength = float(gate_strength * soft_condition_gate)
            correction_cap = float(min(correction_cap, soft_condition_gate))
            if alpha_base <= 1e-6:
                router_gate_pass = False

        mode = 'blend' if alpha_base > 1e-6 else 'global_only'
        if soft_prior_active:
            reason = 'condition_similarity_soft_ensemble' if mode == 'blend' else 'condition_similarity_soft_gate_too_weak'
        elif mode == 'blend':
            reason = 'soft_gate_pass' if self.soft_gate else 'calibrated_gate_pass'
        else:
            reason = 'soft_gate_too_weak' if self.soft_gate else 'router_gate_blocked'

        hint_code = None if method_code_hint is None else str(method_code_hint)
        if hint_code and hint_code in self.experts:
            hinted_score = next((row['score'] for row in scored if row['expert'] == hint_code), 0.0)
            second_best_score = max([row['score'] for row in scored if row['expert'] != hint_code], default=hinted_score)
            hinted_family = self.family_of.get(hint_code, hint_code)
            if route_similarity is None:
                # Explicit, user-supplied method code -> full trust (alpha = alpha_max).
                soft_g = 1.0
                hint_alpha_base = float(self.alpha_max)
                hint_reason = 'method_code_hint'
                hint_tier = 'high'
            else:
                # Soft condition-similarity hint: alpha ramps with the condition-weighted
                # similarity instead of jumping to full trust. This is the #1 fix.
                soft_g = self._similarity_gate(route_similarity)
                hint_alpha_base = float(self.alpha_max * soft_g)
                hint_reason = 'condition_similarity_soft_hint'
                hint_tier = 'high' if soft_g >= 0.66 else ('medium' if soft_g >= 0.33 else 'low')
            return {
                'mode': 'blend' if hint_alpha_base > 1e-6 else 'global_only',
                'experts': [hint_code],
                'families': [hinted_family],
                'weights': {hint_code: 1.0},
                'family_weights': {hinted_family: 1.0},
                'scores': scored,
                'alpha_base': float(hint_alpha_base),
                'router_gate_pass': bool(hint_alpha_base > 1e-6),
                'gate_strength': float(soft_g),
                'reason': hint_reason,
                'route_similarity': None if route_similarity is None else float(route_similarity),
                'similarity_gate': float(soft_g),
                'collision_candidates': [],
                'confidence_tier': hint_tier,
                'family_confidence': float(soft_g),
                'correction_cap': float(soft_g),
                'domain_filter': {
                    'eligible_count': len(scored),
                    'filtered_count': len(filtered_out),
                    'filtered_examples': filtered_out[:8],
                },
                'calibration': {
                    'family_top1': float(soft_g),
                    'family_top2': 0.0,
                    'family_margin': float(soft_g),
                    'family_entropy': 0.0,
                    'top_expert_weight': float(soft_g),
                    'top_expert': hint_code,
                    'top_score': float(hinted_score),
                    'second_score': float(second_best_score),
                    'expert_margin': float(max(0.0, hinted_score - second_best_score)),
                    'checks': {
                        'family_top1_pass': bool(hint_alpha_base > 1e-6),
                        'family_margin_pass': bool(hint_alpha_base > 1e-6),
                        'expert_margin_pass': bool(hint_alpha_base > 1e-6),
                        'top_expert_weight_pass': bool(hint_alpha_base > 1e-6),
                    },
                },
            }

        return {
            'mode': mode,
            'experts': experts,
            'families': [x['family_id'] for x in picked_families],
            'weights': {k: float(v) for k, v in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)},
            'family_weights': family_weights,
            'scores': scored,
            'alpha_base': float(alpha_base),
            'router_gate_pass': bool(router_gate_pass),
            'gate_strength': float(gate_strength),
            'reason': reason,
            'route_similarity': None if route_similarity is None else float(route_similarity),
            'similarity_gate': None if soft_condition_gate is None else float(soft_condition_gate),
            'condition_prior_active': bool(soft_prior_active),
            'condition_prior_max': None if soft_prior_max is None else float(soft_prior_max),
            'condition_prior_methods': {str(k): float(v) for k, v in sorted(soft_prior.items(), key=lambda kv: kv[1], reverse=True)[:8]},
            'collision_candidates': [],
            'confidence_tier': confidence_tier,
            'family_confidence': family_confidence,
            'correction_cap': correction_cap,
            'family_head_used': bool(family_head_used),
            'domain_filter': {
                'eligible_count': len(scored),
                'filtered_count': len(filtered_out),
                'filtered_examples': filtered_out[:8],
            },
            'calibration': {
                'family_top1': fam_top1,
                'family_top2': fam_top2,
                'family_margin': fam_margin,
                'family_entropy': fam_entropy,
                'top_expert_weight': top_expert_weight,
                'top_expert': top_expert,
                'top_score': top_score,
                'second_score': second_score,
                'expert_margin': expert_margin,
                'checks': gate_checks,
                'soft_gate': bool(self.soft_gate),
            },
        }


class UnifiedPredictorDeltaRouterV47:
    def __init__(
        self,
        router_manifest,
        device='cpu',
        topk=3,
        temperature=0.8,
        alpha_max=0.95,
        quality_lambda=0.35,
        disagreement_scale=0.6,
        delta_scale=0.75,
        min_family_top1=0.52,
        min_family_margin=0.10,
        min_expert_margin=0.05,
        min_top_expert_weight=0.28,
        min_effective_alpha=0.02,
        soft_gate=True,
        soft_gate_min=0.20,
        family_head_lambda=0.0,
        expert_rescue_enabled=False,
        expert_rescue_alpha_floor=0.95,
        expert_rescue_min_top_weight=0.60,
        expert_rescue_min_margin=0.08,
        expert_rescue_max_family_top1=0.40,
        expert_rescue_max_family_margin=0.05,
        similarity_alpha_lo=0.20,
        similarity_alpha_hi=0.85,
        condition_prior_lambda=1.0,
        max_delta_experts=3,
    ):
        self.device = device
        self.disagreement_scale = float(disagreement_scale)
        self.delta_scale = float(delta_scale)
        self.max_delta_experts = int(max_delta_experts)
        self.min_effective_alpha = float(min_effective_alpha)
        self.expert_rescue_enabled = bool(expert_rescue_enabled)
        self.expert_rescue_alpha_floor = float(expert_rescue_alpha_floor)
        self.expert_rescue_min_top_weight = float(expert_rescue_min_top_weight)
        self.expert_rescue_min_margin = float(expert_rescue_min_margin)
        self.expert_rescue_max_family_top1 = float(expert_rescue_max_family_top1)
        self.expert_rescue_max_family_margin = float(expert_rescue_max_family_margin)

        self.router = CalibratedQualityFamilyRouter(
            router_manifest,
            topk=topk,
            temperature=temperature,
            alpha_max=alpha_max,
            quality_lambda=quality_lambda,
            min_family_top1=min_family_top1,
            min_family_margin=min_family_margin,
            min_expert_margin=min_expert_margin,
            min_top_expert_weight=min_top_expert_weight,
            soft_gate=soft_gate,
            soft_gate_min=soft_gate_min,
            family_head_lambda=family_head_lambda,
            similarity_alpha_lo=similarity_alpha_lo,
            similarity_alpha_hi=similarity_alpha_hi,
            condition_prior_lambda=condition_prior_lambda,
        )
        self.manifest = self.router.manifest

        self.global_bundle = None
        global_ckpt = self.manifest.get('global_ckpt_path')
        if global_ckpt and os.path.exists(global_ckpt):
            self.global_bundle = GlobalBundle(global_ckpt, device=device)

        self.family_head_bundle = None
        family_head_ckpt = self.manifest.get('family_head_ckpt_path')
        if family_head_ckpt:
            family_head_ckpt = _resolve_manifest_path(self.router.manifest_dir, family_head_ckpt)
            if family_head_ckpt and os.path.exists(family_head_ckpt):
                self.family_head_bundle = FamilyHeadBundle(family_head_ckpt, device=device)

        self.expert_bundles = {}
        for code, info in self.manifest.get('experts', {}).items():
            ckpt = info.get('ckpt_path')
            if ckpt and os.path.exists(ckpt):
                self.expert_bundles[str(code)] = ExpertDeltaBundle(ckpt, device=device)

    def update_runtime_config(self, **cfg):
        if 'disagreement_scale' in cfg and cfg['disagreement_scale'] is not None:
            self.disagreement_scale = float(cfg['disagreement_scale'])
        if 'delta_scale' in cfg and cfg['delta_scale'] is not None:
            self.delta_scale = float(cfg['delta_scale'])
        if 'min_effective_alpha' in cfg and cfg['min_effective_alpha'] is not None:
            self.min_effective_alpha = float(cfg['min_effective_alpha'])
        if 'expert_rescue_enabled' in cfg and cfg['expert_rescue_enabled'] is not None:
            self.expert_rescue_enabled = bool(cfg['expert_rescue_enabled'])
        if 'expert_rescue_alpha_floor' in cfg and cfg['expert_rescue_alpha_floor'] is not None:
            self.expert_rescue_alpha_floor = float(cfg['expert_rescue_alpha_floor'])
        if 'expert_rescue_min_top_weight' in cfg and cfg['expert_rescue_min_top_weight'] is not None:
            self.expert_rescue_min_top_weight = float(cfg['expert_rescue_min_top_weight'])
        if 'expert_rescue_min_margin' in cfg and cfg['expert_rescue_min_margin'] is not None:
            self.expert_rescue_min_margin = float(cfg['expert_rescue_min_margin'])
        if 'expert_rescue_max_family_top1' in cfg and cfg['expert_rescue_max_family_top1'] is not None:
            self.expert_rescue_max_family_top1 = float(cfg['expert_rescue_max_family_top1'])
        if 'expert_rescue_max_family_margin' in cfg and cfg['expert_rescue_max_family_margin'] is not None:
            self.expert_rescue_max_family_margin = float(cfg['expert_rescue_max_family_margin'])
        if 'max_delta_experts' in cfg and cfg['max_delta_experts'] is not None:
            self.max_delta_experts = int(cfg['max_delta_experts'])
        self.router.update_runtime_config(
            topk=cfg.get('topk'),
            temperature=cfg.get('temperature'),
            alpha_max=cfg.get('alpha_max'),
            quality_lambda=cfg.get('quality_lambda'),
            min_family_top1=cfg.get('min_family_top1'),
            min_family_margin=cfg.get('min_family_margin'),
            min_expert_margin=cfg.get('min_expert_margin'),
            min_top_expert_weight=cfg.get('min_top_expert_weight'),
            soft_gate=cfg.get('soft_gate'),
            soft_gate_min=cfg.get('soft_gate_min'),
            family_head_lambda=cfg.get('family_head_lambda'),
            condition_prior_lambda=cfg.get('condition_prior_lambda'),
        )

    def _maybe_rescue_alpha(self, route, alpha_base):
        if not self.expert_rescue_enabled:
            return None, None
        if route.get('reason') == 'method_code_hint':
            return None, None
        if route.get('router_gate_pass', False):
            return None, None

        calibration = route.get('calibration', {}) or {}
        top_expert_weight = float(calibration.get('top_expert_weight', 0.0) or 0.0)
        expert_margin = float(calibration.get('expert_margin', 0.0) or 0.0)
        family_top1 = float(calibration.get('family_top1', 0.0) or 0.0)
        family_margin = float(calibration.get('family_margin', 0.0) or 0.0)
        if top_expert_weight < self.expert_rescue_min_top_weight:
            return None, None
        if expert_margin < self.expert_rescue_min_margin:
            return None, None
        if family_top1 > self.expert_rescue_max_family_top1:
            return None, None
        if family_margin > self.expert_rescue_max_family_margin:
            return None, None

        rescued_alpha = max(float(alpha_base), self.expert_rescue_alpha_floor)
        rescued_alpha = min(rescued_alpha, float(self.router.alpha_max))
        if rescued_alpha < self.min_effective_alpha:
            return None, None
        return rescued_alpha, 'expert_rescue_low_family_confidence'

    def _compute_weighted_delta(self, route, fp_np, meta_np):
        delta_preds = {}
        used = []
        experts = route.get('experts', [])
        if not experts:
            return None, delta_preds, used

        route_weights = route.get('weights', {}) or {}
        weighted_codes = []
        if route_weights:
            for code in experts:
                code = str(code)
                weight = float(route_weights.get(code, 0.0) or 0.0)
                if weight > 0:
                    weighted_codes.append((code, weight))
            weighted_codes.sort(key=lambda item: item[1], reverse=True)
            if self.max_delta_experts > 0:
                weighted_codes = weighted_codes[: self.max_delta_experts]
        if not weighted_codes:
            weighted_codes = [(str(experts[0]), 1.0)]

        total_weight = 0.0
        weighted_delta = 0.0
        for code, weight in weighted_codes:
            bundle = self.expert_bundles.get(str(code))
            if bundle is None:
                continue
            d = bundle.predict_delta(fp_np, meta_np)
            delta_preds[str(code)] = float(d)
            used.append((str(code), float(weight), d, bundle.family_id, bundle.best_val_mae))
            weighted_delta += float(weight) * float(d)
            total_weight += float(weight)
        if total_weight <= 1e-8:
            return None, delta_preds, used
        normalized_used = [
            (code, float(weight) / total_weight, delta, family_id, best_val_mae)
            for code, weight, delta, family_id, best_val_mae in used
        ]
        return float(weighted_delta / total_weight), delta_preds, normalized_used

    def predict(self, fp_np, meta_np, method_code_hint=None, route_signature=None, route_similarity=None, soft_method_prior=None):
        fp_np = np.asarray(fp_np, dtype=np.float32)
        meta_np = np.asarray(meta_np, dtype=np.float32)

        family_probs = None
        if self.family_head_bundle is not None:
            family_probs = self.family_head_bundle.predict_probs(fp_np, meta_np)

        route = self.router.route(
            meta_np,
            method_code_hint=method_code_hint,
            route_signature=route_signature,
            family_probs=family_probs,
            route_similarity=route_similarity,
            soft_method_prior=soft_method_prior,
        )
        pred_global = None if self.global_bundle is None else self.global_bundle.predict(fp_np, meta_np)

        weighted_delta, delta_preds, used = self._compute_weighted_delta(route, fp_np, meta_np)

        alpha_base = float(route.get('alpha_base', 0.0)) if weighted_delta is not None else 0.0
        confidence_tier = str(route.get('confidence_tier', 'low'))
        correction_cap = float(route.get('correction_cap', 0.0))
        def _cap_alpha(value):
            alpha_value = float(value)
            if np.isfinite(correction_cap):
                alpha_value = min(alpha_value, max(0.0, correction_cap))
            return alpha_value

        disagreement_std = None
        disagreement_penalty = None
        delta_magnitude_penalty = None
        alpha = 0.0
        pred_expert = None if weighted_delta is None or pred_global is None else float(pred_global + weighted_delta)
        pred_final = pred_global
        final_reason = 'global_only'
        top1_routed_pred = None
        delta_source = 'ensemble_delta' if len(used) > 1 else 'top1_delta'

        if route.get('experts'):
            top_code = route['experts'][0]
            top_bundle = self.expert_bundles.get(str(top_code))
            if pred_global is not None and top_bundle is not None:
                top1_routed_pred = float(pred_global + top_bundle.predict_delta(fp_np, meta_np))

        if pred_global is None and weighted_delta is None:
            pred_final = None
            final_reason = 'no_global_and_no_expert'
        elif pred_global is None:
            pred_final = None
            final_reason = 'no_global_for_delta_model'
        elif weighted_delta is None:
            pred_final = float(pred_global)
            final_reason = 'global_only'
        elif route.get('reason') == 'method_code_hint':
            alpha = _cap_alpha(alpha_base)
            if alpha < self.min_effective_alpha:
                alpha = 0.0
                pred_final = float(pred_global)
                final_reason = 'explicit_method_alpha_or_cap_too_small_back_to_global'
            else:
                pred_final = float(pred_global + alpha * weighted_delta)
                final_reason = 'explicit_method_expert'
        elif route.get('reason') == 'condition_similarity_soft_hint':
            alpha = _cap_alpha(alpha_base)
            if alpha < self.min_effective_alpha:
                alpha = 0.0
                pred_final = float(pred_global)
                final_reason = 'soft_similarity_alpha_or_cap_too_small_back_to_global'
            else:
                pred_final = float(pred_global + alpha * weighted_delta)
                final_reason = 'condition_similarity_soft_blend'
        elif route.get('reason') == 'condition_similarity_soft_ensemble':
            alpha = _cap_alpha(alpha_base)
            if alpha < self.min_effective_alpha:
                alpha = 0.0
                pred_final = float(pred_global)
                final_reason = 'soft_similarity_alpha_or_cap_too_small_back_to_global'
            else:
                pred_final = float(pred_global + alpha * weighted_delta)
                final_reason = f'condition_similarity_soft_{delta_source}'
        else:
            disagreement_std = 0.0
            disagreement_penalty = 1.0
            delta_magnitude_penalty = 1.0
            alpha = _cap_alpha(alpha_base)
            if alpha < self.min_effective_alpha:
                rescued_alpha, rescue_reason = self._maybe_rescue_alpha(route, alpha_base)
                if rescued_alpha is None:
                    alpha = 0.0
                    pred_final = float(pred_global)
                    final_reason = 'alpha_or_cap_too_small_back_to_global'
                else:
                    alpha = _cap_alpha(rescued_alpha)
                    if alpha < self.min_effective_alpha:
                        alpha = 0.0
                        pred_final = float(pred_global)
                        final_reason = 'rescued_alpha_capped_back_to_global'
                    else:
                        pred_final = float(pred_global + alpha * weighted_delta)
                        final_reason = rescue_reason
            else:
                pred_final = float(pred_global + alpha * weighted_delta)
                final_reason = f'simple_global_plus_{confidence_tier}_{delta_source}'

        return {
            'pred_global': None if pred_global is None else float(pred_global),
            'pred_expert': None if pred_expert is None else float(pred_expert),
            'pred_final': None if pred_final is None else float(pred_final),
            'pred_delta': None if weighted_delta is None else float(weighted_delta),
            'pred_routed_top1': None if top1_routed_pred is None else float(top1_routed_pred),
            'mode': route.get('mode', 'global_only'),
            'experts': route.get('experts', []),
            'families': route.get('families', []),
            'weights': route.get('weights', {}),
            'family_weights': route.get('family_weights', {}),
            'scores': route.get('scores', []),
            'alpha_base': float(alpha_base),
            'alpha': float(alpha),
            'reason': route.get('reason', 'unknown'),
            'final_reason': final_reason,
            'delta_preds': delta_preds,
            'delta_used': [
                {
                    'expert': str(code),
                    'weight': float(weight),
                    'delta': float(delta),
                    'family_id': str(family_id),
                    'best_val_mae': None if best_val_mae is None else float(best_val_mae),
                }
                for code, weight, delta, family_id, best_val_mae in used
            ],
            'disagreement_std': disagreement_std,
            'disagreement_penalty': disagreement_penalty,
            'delta_magnitude_penalty': delta_magnitude_penalty,
            'router_gate_pass': bool(route.get('router_gate_pass', False)),
            'gate_strength': float(route.get('gate_strength', 0.0)),
            'confidence_tier': confidence_tier,
            'family_confidence': float(route.get('family_confidence', 0.0)),
            'correction_cap': correction_cap,
            'family_head_used': bool(route.get('family_head_used', False)),
            'collision_candidates': route.get('collision_candidates', []),
            'route_similarity': route.get('route_similarity'),
            'similarity_gate': route.get('similarity_gate'),
            'calibration': route.get('calibration', {}),
        }

    def predict_modes(self, fp_np, meta_np, true_method_code=None, route_method_code_hint=None, route_signature=None):
        out = self.predict(fp_np, meta_np, method_code_hint=route_method_code_hint, route_signature=route_signature)
        pred_global = out.get('pred_global')
        pred_oracle = None
        pred_routed = out.get('pred_routed_top1')

        if pred_global is not None and true_method_code is not None:
            true_bundle = self.expert_bundles.get(str(true_method_code))
            if true_bundle is not None:
                pred_oracle = float(pred_global + true_bundle.predict_delta(fp_np, meta_np))

        return {
            'global_only': pred_global,
            'oracle_expert': pred_oracle,
            'routed_expert': pred_routed,
            'final_blend': out.get('pred_final'),
            'router_top1': out.get('experts', [None])[0] if out.get('experts') else None,
            'router_gate_pass': out.get('router_gate_pass', False),
            'predict_detail': out,
        }
