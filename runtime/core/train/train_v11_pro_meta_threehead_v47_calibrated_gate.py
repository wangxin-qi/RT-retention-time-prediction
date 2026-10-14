import os
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CORE_ROOT)
for _path in (HERE, CORE_ROOT, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.data.dataset_v11_pro_meta_threehead_v47_calibrated_gate import (
    load_yaml_or_json,
    resolve_metadata_path,
    extract_method_meta_vector,
    extract_route_signature,
    smiles_to_fp,
    list_method_dirs,
    load_method_dataset,
    build_combined_dataset,
    build_delta_dataset,
    split_indices_by_group,
    collate_fn,
    load_method_family_map,
)
from core.models.model_v11_pro_meta_threehead_v47_calibrated_gate import (
    GlobalMLP,
    ExpertDeltaMLP,
    GlobalBundle,
    FamilyHead,
    UnifiedPredictorDeltaRouterV47,
)

LOCAL_SEARCH_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed_data')
RESTORED_SEARCH_ROOT = os.environ.get('ROUTERT_PROCESSED_DATA', '')

# =========================
# 运行模式：'train' / 'predict' / 'eval' / 'family_head'
#   'family_head' = 复用已有 global/expert ckpt，单独训 family head 并重跑 eval
# =========================
RUN_MODE = 'train'

SEARCH_ROOT = LOCAL_SEARCH_ROOT if os.path.isdir(LOCAL_SEARCH_ROOT) else (
    RESTORED_SEARCH_ROOT if os.path.isdir(RESTORED_SEARCH_ROOT) else LOCAL_SEARCH_ROOT
)
OUT_DIR = os.path.join(
    PROJECT_ROOT,
    'artifacts',
    'archive',
    'runs',
    'v11_pro_meta_threehead_v47_calibrated_gate',
)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- train ----------
MIN_ROWS = 300
MAX_EXPERTS = 64
VAL_RATIO = 0.10
SEED = 42
EPOCHS_GLOBAL = 50
EPOCHS_EXPERT = 40
BATCH_SIZE = 256
PRED_BATCH_SIZE = 1024
LR_GLOBAL = 3e-4
LR_EXPERT = 2e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
GLOBAL_HIDDEN = 384
EXPERT_HIDDEN = 256
DROPOUT = 0.2
DELTA_L2 = 0.01
FAMILY_MAP_PATH = None

# ---------- router ----------
ROUTER_TOPK = 3
ROUTER_TEMPERATURE = 0.8
ROUTER_ALPHA_MAX = 0.95
ROUTER_QUALITY_LAMBDA = 0.35
ROUTER_DISAGREEMENT_SCALE = 0.6
ROUTER_DELTA_SCALE = 0.75
ROUTER_MIN_FAMILY_TOP1 = 0.52
ROUTER_MIN_FAMILY_MARGIN = 0.10
ROUTER_MIN_EXPERT_MARGIN = 0.05
ROUTER_MIN_TOP_EXPERT_WEIGHT = 0.28
ROUTER_MIN_EFFECTIVE_ALPHA = 0.02
ROUTER_SOFT_GATE = True
ROUTER_SOFT_GATE_MIN = 0.20
ROUTER_FAMILY_HEAD_LAMBDA = 0.0

# ---------- family head ----------
FAMILY_HEAD_HIDDEN = 128
FAMILY_HEAD_DROPOUT = 0.2
FAMILY_HEAD_EPOCHS = 12
FAMILY_HEAD_LR = 1e-3
FAMILY_HEAD_BATCH_SIZE = 256
FAMILY_HEAD_PATIENCE = 4

# ---------- learned meta embedding ----------
META_EMBED_DIM = 8
META_ENCODER_HIDDEN = 32
META_RECON_LAMBDA_GLOBAL = 0.05
META_RECON_LAMBDA_EXPERT = 0.05
META_RECON_LAMBDA_FAMILY = 0.02

# ---------- predict demo ----------
ROUTER_MANIFEST = os.path.join(OUT_DIR, 'router_manifest_v47.json')
SMILES = 'CC1=C(C=C(OC1=O)CC(C(C)C(C(C)C)O)O)O'
METHOD_METADATA_PATH = os.path.join(SEARCH_ROOT, '0390', '0390_metadata.yaml')
METHOD_CODE = None


def to_tensors(batch, device, mode='global'):
    fp = torch.from_numpy(batch['fp']).float().to(device)
    meta = torch.from_numpy(batch['meta']).float().to(device)
    rt = torch.from_numpy(batch['rt']).float().to(device)
    if mode == 'global':
        rt_z = torch.from_numpy(batch['rt_z']).float().to(device)
        return fp, meta, rt, rt_z
    base_pred = torch.from_numpy(batch['base_pred']).float().to(device)
    delta = torch.from_numpy(batch['delta']).float().to(device)
    delta_z = torch.from_numpy(batch['delta_z']).float().to(device)
    return fp, meta, rt, base_pred, delta, delta_z


def evaluate_global(model, loader, rt_mean, rt_std, device):
    model.eval()
    maes = []
    with torch.no_grad():
        for batch in loader:
            fp, meta, rt, _ = to_tensors(batch, device, mode='global')
            pred_z = model(fp, meta)
            pred = pred_z * rt_std + rt_mean
            maes.append(torch.mean(torch.abs(pred - rt)).item())
    return float(np.mean(maes)) if maes else 1e9


def evaluate_delta(model, loader, delta_mean, delta_std, device):
    model.eval()
    maes = []
    with torch.no_grad():
        for batch in loader:
            fp, meta, rt, base_pred, _, _ = to_tensors(batch, device, mode='delta')
            pred_delta_z = model(fp, meta)
            pred_delta = pred_delta_z * delta_std + delta_mean
            pred = base_pred + pred_delta
            maes.append(torch.mean(torch.abs(pred - rt)).item())
    return float(np.mean(maes)) if maes else 1e9


def _compute_meta_stats(meta_arr, idx):
    sub = np.asarray(meta_arr, dtype=np.float32)[idx]
    mean = sub.mean(axis=0).astype(np.float32)
    std = (sub.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def train_global(ds, out_dir):
    tr_idx, va_idx = split_indices_by_group(ds.smiles, val_ratio=VAL_RATIO, seed=SEED)
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    meta_mean, meta_std = _compute_meta_stats(ds.meta, tr_idx)
    print(f'[global] meta_dim={ds.meta.shape[1]} meta_mean[:6]={meta_mean[:6].tolist()} meta_std[:6]={meta_std[:6].tolist()}')

    model = GlobalMLP(
        fp_dim=ds.fps.shape[1],
        meta_dim=ds.meta.shape[1],
        hidden=GLOBAL_HIDDEN,
        dropout=DROPOUT,
        meta_mean=meta_mean,
        meta_std=meta_std,
        meta_embed_dim=META_EMBED_DIM,
        meta_encoder_hidden=META_ENCODER_HIDDEN,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_GLOBAL, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
    loss_fn = torch.nn.SmoothL1Loss()

    best_mae, best_epoch, bad = 1e9, -1, 0
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best_global.pt')

    for epoch in range(EPOCHS_GLOBAL):
        model.train()
        tr_losses, tr_recon_losses = [], []
        for batch in tr_loader:
            fp, meta, _, rt_z = to_tensors(batch, DEVICE, mode='global')
            pred_z, aux = model(fp, meta, return_aux=True)
            recon_loss = torch.mean((aux['meta_recon'] - aux['meta_norm']) ** 2)
            loss = loss_fn(pred_z, rt_z) + META_RECON_LAMBDA_GLOBAL * recon_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())
            tr_recon_losses.append(recon_loss.item())

        va_mae = evaluate_global(model, va_loader, ds.rt_mean, ds.rt_std, DEVICE)
        sched.step(va_mae)
        print(
            f"[global] epoch {epoch:03d} | train_loss={np.mean(tr_losses):.4f} "
            f"| recon_loss={np.mean(tr_recon_losses):.4f} | val_mae={va_mae:.4f} | best={best_mae:.4f}"
        )

        if va_mae < best_mae:
            best_mae, best_epoch, bad = va_mae, epoch, 0
            torch.save({
                'model_state': model.state_dict(),
                'meta': {
                    'rt_mean': ds.rt_mean,
                    'rt_std': ds.rt_std,
                    'fp_dim': int(ds.fps.shape[1]),
                    'meta_dim': int(ds.meta.shape[1]),
                    'hidden': GLOBAL_HIDDEN,
                    'dropout': DROPOUT,
                    'rows': len(ds),
                    'meta_mean': meta_mean.tolist(),
                    'meta_std': meta_std.tolist(),
                    'meta_embed_dim': META_EMBED_DIM,
                    'meta_encoder_hidden': META_ENCODER_HIDDEN,
                }
            }, ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'[EARLY STOP][global] epoch={epoch}')
                break

    return {
        'rows': len(ds),
        'best_val_mae': float(best_mae),
        'best_epoch': int(best_epoch),
        'ckpt_path': ckpt_path,
        'meta_mean': meta_mean.tolist(),
        'meta_std': meta_std.tolist(),
    }


def train_one_expert_delta(method_ds, info, out_dir, global_ckpt_path, meta_mean=None, meta_std=None):
    global_bundle = GlobalBundle(global_ckpt_path, device=DEVICE)
    base_preds = global_bundle.predict_array(method_ds.fps, method_ds.meta, batch_size=PRED_BATCH_SIZE)
    delta_ds = build_delta_dataset(method_ds, base_preds)

    tr_idx, va_idx = split_indices_by_group(delta_ds.smiles, val_ratio=VAL_RATIO, seed=SEED)
    tr_loader = DataLoader(Subset(delta_ds, tr_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    va_loader = DataLoader(Subset(delta_ds, va_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = ExpertDeltaMLP(
        fp_dim=delta_ds.fps.shape[1],
        meta_dim=delta_ds.meta.shape[1],
        hidden=EXPERT_HIDDEN,
        dropout=DROPOUT,
        meta_mean=meta_mean,
        meta_std=meta_std,
        meta_embed_dim=META_EMBED_DIM,
        meta_encoder_hidden=META_ENCODER_HIDDEN,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_EXPERT, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
    loss_fn = torch.nn.SmoothL1Loss()

    best_mae, best_epoch, bad = 1e9, -1, 0
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best.pt')

    for epoch in range(EPOCHS_EXPERT):
        model.train()
        tr_losses, tr_recon_losses = [], []
        for batch in tr_loader:
            fp, meta, _, _, _, delta_z = to_tensors(batch, DEVICE, mode='delta')
            pred_delta_z, aux = model(fp, meta, return_aux=True)
            pred_delta = pred_delta_z * delta_ds.delta_std + delta_ds.delta_mean
            recon_loss = torch.mean((aux['meta_recon'] - aux['meta_norm']) ** 2)
            loss = (
                loss_fn(pred_delta_z, delta_z)
                + DELTA_L2 * torch.mean(pred_delta ** 2)
                + META_RECON_LAMBDA_EXPERT * recon_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(loss.item())
            tr_recon_losses.append(recon_loss.item())

        va_mae = evaluate_delta(model, va_loader, delta_ds.delta_mean, delta_ds.delta_std, DEVICE)
        sched.step(va_mae)
        print(
            f"[expert-delta:{info['method_code']}] epoch {epoch:03d} | train_loss={np.mean(tr_losses):.4f} "
            f"| recon_loss={np.mean(tr_recon_losses):.4f} | val_mae={va_mae:.4f} | best={best_mae:.4f}"
        )

        if va_mae < best_mae:
            best_mae, best_epoch, bad = va_mae, epoch, 0
            torch.save({
                'model_state': model.state_dict(),
                'meta': {
                    'method_code': info['method_code'],
                    'family_id': info['family_id'],
                    'delta_mean': delta_ds.delta_mean,
                    'delta_std': delta_ds.delta_std,
                    'fp_dim': int(delta_ds.fps.shape[1]),
                    'meta_dim': int(delta_ds.meta.shape[1]),
                    'hidden': EXPERT_HIDDEN,
                    'dropout': DROPOUT,
                    'metadata_path': info['metadata_path'],
                    'prototype': info['prototype'],
                    'route_signature': info.get('route_signature'),
                    'rows': info['rows'],
                    'global_ckpt_path': global_ckpt_path,
                    'best_val_mae': float(va_mae),
                    'meta_mean': None if meta_mean is None else (meta_mean.tolist() if hasattr(meta_mean, 'tolist') else list(meta_mean)),
                    'meta_std': None if meta_std is None else (meta_std.tolist() if hasattr(meta_std, 'tolist') else list(meta_std)),
                    'meta_embed_dim': META_EMBED_DIM,
                    'meta_encoder_hidden': META_ENCODER_HIDDEN,
                }
            }, ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[EARLY STOP][expert-delta:{info['method_code']}] epoch={epoch}")
                break

    return {
        'method_code': info['method_code'],
        'family_id': info['family_id'],
        'rows': info['rows'],
        'best_val_mae': float(best_mae),
        'best_epoch': int(best_epoch),
        'metadata_path': info['metadata_path'],
        'prototype': info['prototype'],
        'route_signature': info.get('route_signature'),
        'ckpt_path': ckpt_path,
        'delta_mean': float(delta_ds.delta_mean),
        'delta_std': float(delta_ds.delta_std),
    }


def build_predictor(router_manifest):
    return UnifiedPredictorDeltaRouterV47(
        router_manifest=router_manifest,
        device=DEVICE,
        topk=ROUTER_TOPK,
        temperature=ROUTER_TEMPERATURE,
        alpha_max=ROUTER_ALPHA_MAX,
        quality_lambda=ROUTER_QUALITY_LAMBDA,
        disagreement_scale=ROUTER_DISAGREEMENT_SCALE,
        delta_scale=ROUTER_DELTA_SCALE,
        min_family_top1=ROUTER_MIN_FAMILY_TOP1,
        min_family_margin=ROUTER_MIN_FAMILY_MARGIN,
        min_expert_margin=ROUTER_MIN_EXPERT_MARGIN,
        min_top_expert_weight=ROUTER_MIN_TOP_EXPERT_WEIGHT,
        min_effective_alpha=ROUTER_MIN_EFFECTIVE_ALPHA,
        soft_gate=ROUTER_SOFT_GATE,
        soft_gate_min=ROUTER_SOFT_GATE_MIN,
        family_head_lambda=ROUTER_FAMILY_HEAD_LAMBDA,
    )


def evaluate_router_same_set(selected_pairs, manifest_path, out_dir, route_with_method_hint=False, prefix='v47_same_set_eval'):
    predictor = build_predictor(manifest_path)
    rows = []

    for ds, info in selected_pairs:
        meta_dict = load_yaml_or_json(info['metadata_path'])
        route_signature_full = extract_route_signature(
            meta_dict,
            method_dir=os.path.dirname(info['metadata_path']),
            method_code=info['method_code'],
        )
        # 只有显式 with_method_hint 模式才传 route_signature；without-identity 模式必须断掉
        # 这个隐式身份泄露通路（method 专属 route_signature → 路由相似度），否则 mae_routed 是有偏的。
        route_signature = route_signature_full if route_with_method_hint else None
        _, va_idx = split_indices_by_group(ds.smiles, val_ratio=VAL_RATIO, seed=SEED)
        for idx in va_idx.tolist():
            fp = ds.fps[idx]
            meta = ds.meta[idx]
            rt = float(ds.rts[idx])
            smi = str(ds.smiles[idx])
            modes = predictor.predict_modes(
                fp,
                meta,
                true_method_code=info['method_code'],
                route_method_code_hint=(info['method_code'] if route_with_method_hint else None),
                route_signature=route_signature,
            )
            detail = modes['predict_detail']
            rows.append({
                'method_code': info['method_code'],
                'family_id': info['family_id'],
                'smiles': smi,
                'rt_true': rt,
                'pred_global': modes['global_only'],
                'pred_oracle': modes['oracle_expert'],
                'pred_routed': modes['routed_expert'],
                'pred_final': modes['final_blend'],
                'router_top1': modes['router_top1'],
                'router_gate_pass': bool(modes['router_gate_pass']),
                'alpha': detail.get('alpha'),
                'alpha_base': detail.get('alpha_base'),
                'pred_delta': detail.get('pred_delta'),
                'confidence_tier': detail.get('confidence_tier', ''),
                'family_confidence': detail.get('family_confidence'),
                'correction_cap': detail.get('correction_cap'),
                'top_expert_weight': detail.get('calibration', {}).get('top_expert_weight'),
                'family_top1': detail.get('calibration', {}).get('family_top1'),
                'family_margin': detail.get('calibration', {}).get('family_margin'),
                'expert_margin': detail.get('calibration', {}).get('expert_margin'),
            })

    if not rows:
        print('[WARN] no rows for evaluation')
        return None

    df = pd.DataFrame(rows)
    for pred_col, err_col in [
        ('pred_global', 'mae_global'),
        ('pred_oracle', 'mae_oracle'),
        ('pred_routed', 'mae_routed'),
        ('pred_final', 'mae_final'),
    ]:
        df[err_col] = (df[pred_col] - df['rt_true']).abs()

    df['router_top1_correct'] = (df['router_top1'].astype(str) == df['method_code'].astype(str)).astype(np.int64)

    overall = {
        'n': int(len(df)),
        'route_with_method_hint': bool(route_with_method_hint),
        'mae_global': float(df['mae_global'].mean(skipna=True)),
        'mae_oracle': float(df['mae_oracle'].mean(skipna=True)),
        'mae_routed': float(df['mae_routed'].mean(skipna=True)),
        'mae_final': float(df['mae_final'].mean(skipna=True)),
        'router_top1_acc': float(df['router_top1_correct'].mean()),
        'gate_pass_rate': float(df['router_gate_pass'].mean()),
        'family_confidence_mean': float(df['family_confidence'].mean(skipna=True)),
    }

    per_method = []
    for code, g in df.groupby('method_code'):
        per_method.append({
            'method_code': str(code),
            'n': int(len(g)),
            'mae_global': float(g['mae_global'].mean(skipna=True)),
            'mae_oracle': float(g['mae_oracle'].mean(skipna=True)),
            'mae_routed': float(g['mae_routed'].mean(skipna=True)),
            'mae_final': float(g['mae_final'].mean(skipna=True)),
            'router_top1_acc': float(g['router_top1_correct'].mean()),
            'gate_pass_rate': float(g['router_gate_pass'].mean()),
        })
    per_method = sorted(per_method, key=lambda x: x['method_code'])

    save_csv = os.path.join(out_dir, f'{prefix}_samples.csv')
    save_json = os.path.join(out_dir, f'{prefix}_summary.json')
    df.to_csv(save_csv, index=False, encoding='utf-8-sig')
    with open(save_json, 'w', encoding='utf-8') as f:
        json.dump({'overall': overall, 'per_method': per_method}, f, ensure_ascii=False, indent=2)

    print('[EVAL][overall]', json.dumps(overall, ensure_ascii=False))
    print('[SAVE]', save_csv)
    print('[SAVE]', save_json)
    return {'overall': overall, 'per_method': per_method}


def train_family_head(selected_pairs, out_dir, global_ckpt_path):
    """复用 global ckpt 的 meta_mean/std，从 selected_pairs 派生 sample → family_id 标签训分类头。"""
    family_ids_sorted = sorted({info['family_id'] for _, info in selected_pairs})
    family_index = {fam: i for i, fam in enumerate(family_ids_sorted)}
    num_families = len(family_index)
    print(f'[family_head] num_families={num_families} families={family_ids_sorted}')

    fps_chunks, metas_chunks, smiles_chunks, label_chunks = [], [], [], []
    for ds, info in selected_pairs:
        fam_idx = family_index[info['family_id']]
        fps_chunks.append(ds.fps)
        metas_chunks.append(ds.meta)
        smiles_chunks.append(ds.smiles)
        label_chunks.append(np.full(len(ds), fam_idx, dtype=np.int64))
    fps = np.concatenate(fps_chunks, axis=0).astype(np.float32)
    metas = np.concatenate(metas_chunks, axis=0).astype(np.float32)
    smiles_all = np.concatenate(smiles_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0).astype(np.int64)
    print(f'[family_head] combined rows: {len(fps)} | fp_dim={fps.shape[1]} | meta_dim={metas.shape[1]}')

    tr_idx, va_idx = split_indices_by_group(smiles_all, val_ratio=VAL_RATIO, seed=SEED)
    print(f'[family_head] train rows: {len(tr_idx)} | val rows: {len(va_idx)}')

    # 优先复用 global ckpt 里的 meta_mean/std；若缺则用 train fold 现算
    blob = torch.load(global_ckpt_path, map_location='cpu')
    g_meta = blob.get('meta', {})
    g_mean = g_meta.get('meta_mean')
    g_std = g_meta.get('meta_std')
    if g_mean is not None and g_std is not None and len(g_mean) == metas.shape[1]:
        meta_mean = np.asarray(g_mean, dtype=np.float32)
        meta_std = np.asarray(g_std, dtype=np.float32)
        print('[family_head] reuse meta_mean/std from global ckpt')
    else:
        meta_mean, meta_std = _compute_meta_stats(metas, tr_idx)
        print('[family_head] recompute meta_mean/std from train fold')

    model = FamilyHead(
        fp_dim=fps.shape[1], meta_dim=metas.shape[1],
        hidden=FAMILY_HEAD_HIDDEN, num_families=num_families, dropout=FAMILY_HEAD_DROPOUT,
        meta_mean=meta_mean, meta_std=meta_std,
        meta_embed_dim=META_EMBED_DIM, meta_encoder_hidden=META_ENCODER_HIDDEN,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=FAMILY_HEAD_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    def batches(idx, shuffle):
        order = np.arange(len(idx))
        if shuffle:
            np.random.default_rng(SEED).shuffle(order)
        for i in range(0, len(idx), FAMILY_HEAD_BATCH_SIZE):
            sub = idx[order[i:i + FAMILY_HEAD_BATCH_SIZE]]
            yield (
                torch.from_numpy(fps[sub]).float().to(DEVICE),
                torch.from_numpy(metas[sub]).float().to(DEVICE),
                torch.from_numpy(labels[sub]).long().to(DEVICE),
            )

    best_val_loss, best_val_acc, best_epoch, bad = 1e9, 0.0, -1, 0
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'family_head.pt')

    for epoch in range(FAMILY_HEAD_EPOCHS):
        model.train()
        tr_losses, tr_recon_losses, tr_correct, tr_n = [], [], 0, 0
        for fp_b, meta_b, lab_b in batches(tr_idx, shuffle=True):
            logits, aux = model(fp_b, meta_b, return_aux=True)
            recon_loss = torch.mean((aux['meta_recon'] - aux['meta_norm']) ** 2)
            loss = loss_fn(logits, lab_b) + META_RECON_LAMBDA_FAMILY * recon_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tr_losses.append(loss.item())
            tr_recon_losses.append(recon_loss.item())
            tr_correct += int((logits.argmax(dim=1) == lab_b).sum().item())
            tr_n += int(lab_b.numel())

        model.eval()
        va_losses, va_correct, va_n = [], 0, 0
        with torch.no_grad():
            for fp_b, meta_b, lab_b in batches(va_idx, shuffle=False):
                logits, aux = model(fp_b, meta_b, return_aux=True)
                recon_loss = torch.mean((aux['meta_recon'] - aux['meta_norm']) ** 2)
                va_losses.append((loss_fn(logits, lab_b) + META_RECON_LAMBDA_FAMILY * recon_loss).item())
                va_correct += int((logits.argmax(dim=1) == lab_b).sum().item())
                va_n += int(lab_b.numel())
        va_loss = float(np.mean(va_losses)) if va_losses else 1e9
        va_acc = float(va_correct / max(va_n, 1))
        tr_acc = float(tr_correct / max(tr_n, 1))
        sched.step(va_loss)
        print(
            f'[family_head] epoch {epoch:03d} | tr_loss={np.mean(tr_losses):.4f} '
            f'tr_recon={np.mean(tr_recon_losses):.4f} tr_acc={tr_acc:.4f} '
            f'| va_loss={va_loss:.4f} va_acc={va_acc:.4f} | best_va_loss={best_val_loss:.4f}'
        )

        if va_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch, bad = va_loss, va_acc, epoch, 0
            torch.save({
                'model_state': model.state_dict(),
                'meta': {
                    'fp_dim': int(fps.shape[1]),
                    'meta_dim': int(metas.shape[1]),
                    'hidden': FAMILY_HEAD_HIDDEN,
                    'dropout': FAMILY_HEAD_DROPOUT,
                    'num_families': num_families,
                    'family_index': family_index,
                    'meta_mean': meta_mean.tolist(),
                    'meta_std': meta_std.tolist(),
                    'meta_embed_dim': META_EMBED_DIM,
                    'meta_encoder_hidden': META_ENCODER_HIDDEN,
                    'best_val_loss': float(va_loss),
                    'best_val_acc': float(va_acc),
                },
            }, ckpt_path)
        else:
            bad += 1
            if bad >= FAMILY_HEAD_PATIENCE:
                print(f'[EARLY STOP][family_head] epoch={epoch}')
                break

    return {
        'ckpt_path': ckpt_path,
        'family_index': family_index,
        'num_families': num_families,
        'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc),
        'best_epoch': int(best_epoch),
    }


def update_manifest_with_family_head(manifest_path, family_head_res):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    manifest['family_head_ckpt_path'] = family_head_res['ckpt_path']
    manifest['family_index'] = family_head_res['family_index']
    manifest['family_head_summary'] = {
        'num_families': family_head_res['num_families'],
        'best_val_loss': family_head_res['best_val_loss'],
        'best_val_acc': family_head_res['best_val_acc'],
        'best_epoch': family_head_res['best_epoch'],
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('[family_head] manifest updated:', manifest_path)


def family_head_only():
    """复用现有 global+expert ckpt，单训 family head，更新 manifest，重跑 same_set eval。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    selected, _ = collect_selected_pairs()
    if not os.path.exists(os.path.join(OUT_DIR, 'best_global.pt')):
        raise FileNotFoundError('best_global.pt not found; need to run RUN_MODE=train first')
    fh_res = train_family_head(selected, OUT_DIR, os.path.join(OUT_DIR, 'best_global.pt'))
    update_manifest_with_family_head(ROUTER_MANIFEST, fh_res)
    evaluate_router_same_set(selected, ROUTER_MANIFEST, OUT_DIR, route_with_method_hint=False, prefix='v47_same_set_eval_without_method_identity')
    evaluate_router_same_set(selected, ROUTER_MANIFEST, OUT_DIR, route_with_method_hint=True, prefix='v47_same_set_eval_with_method_identity')


def collect_selected_pairs():
    family_map = load_method_family_map(FAMILY_MAP_PATH)
    method_dirs = list_method_dirs(SEARCH_ROOT)

    candidates, skipped = [], []
    for md in method_dirs:
        ds, info = load_method_dataset(md, min_rows=MIN_ROWS, family_map=family_map)
        if ds is None:
            skipped.append(info)
            continue
        candidates.append((ds, info))

    candidates.sort(key=lambda x: x[1]['rows'], reverse=True)
    selected = candidates if MAX_EXPERTS is None else candidates[:MAX_EXPERTS]
    return selected, skipped


def train_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    selected, skipped = collect_selected_pairs()

    print('[INFO] device:', DEVICE)
    print('[INFO] selected experts:', [(info['method_code'], info['rows'], info['family_id']) for _, info in selected])

    combined_ds = build_combined_dataset(selected)
    print('[INFO] combined rows for global:', len(combined_ds))

    global_res = train_global(combined_ds, OUT_DIR)
    g_meta_mean = np.asarray(global_res.get('meta_mean', []), dtype=np.float32)
    g_meta_std = np.asarray(global_res.get('meta_std', []), dtype=np.float32)

    results = {}
    family_summary = {}
    for ds, info in selected:
        out_dir = os.path.join(OUT_DIR, f"expert_{info['method_code']}")
        res = train_one_expert_delta(
            ds,
            info,
            out_dir,
            global_res['ckpt_path'],
            meta_mean=g_meta_mean if g_meta_mean.size else None,
            meta_std=g_meta_std if g_meta_std.size else None,
        )
        results[info['method_code']] = res
        family_summary.setdefault(info['family_id'], []).append(info['method_code'])

    proto_arr = np.stack([np.asarray(res['prototype'], dtype=np.float32) for res in results.values()], axis=0)
    route_arr = np.stack([np.asarray(res['route_signature'], dtype=np.float32) for res in results.values()], axis=0)
    manifest = {
        'search_root': SEARCH_ROOT,
        'min_rows': MIN_ROWS,
        'max_experts': MAX_EXPERTS,
        'seed': SEED,
        'device': DEVICE,
        'global_ckpt_path': global_res['ckpt_path'],
        'global_summary': global_res,
        'experts': {code: res for code, res in results.items()},
        'families': family_summary,
        'router_stats': {
            'meta_mean': proto_arr.mean(axis=0).tolist(),
            'meta_std': (proto_arr.std(axis=0) + 1e-6).tolist(),
            'route_mean': route_arr.mean(axis=0).tolist(),
            'route_std': (route_arr.std(axis=0) + 1e-6).tolist(),
        },
        'router_cfg': {
            'topk': ROUTER_TOPK,
            'temperature': ROUTER_TEMPERATURE,
            'alpha_max': ROUTER_ALPHA_MAX,
            'quality_lambda': ROUTER_QUALITY_LAMBDA,
            'disagreement_scale': ROUTER_DISAGREEMENT_SCALE,
            'delta_scale': ROUTER_DELTA_SCALE,
            'min_family_top1': ROUTER_MIN_FAMILY_TOP1,
            'min_family_margin': ROUTER_MIN_FAMILY_MARGIN,
            'min_expert_margin': ROUTER_MIN_EXPERT_MARGIN,
            'min_top_expert_weight': ROUTER_MIN_TOP_EXPERT_WEIGHT,
            'min_effective_alpha': ROUTER_MIN_EFFECTIVE_ALPHA,
        },
        'learned_meta_cfg': {
            'meta_embed_dim': META_EMBED_DIM,
            'meta_encoder_hidden': META_ENCODER_HIDDEN,
            'meta_recon_lambda_global': META_RECON_LAMBDA_GLOBAL,
            'meta_recon_lambda_expert': META_RECON_LAMBDA_EXPERT,
            'meta_recon_lambda_family': META_RECON_LAMBDA_FAMILY,
        },
        'skipped_count': len(skipped),
        'family_map_path': FAMILY_MAP_PATH,
    }

    manifest_path = os.path.join(OUT_DIR, 'router_manifest_v47.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, 'skipped_methods_v47.json'), 'w', encoding='utf-8') as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)
    print('[DONE] manifest:', manifest_path)

    evaluate_router_same_set(selected, manifest_path, OUT_DIR, route_with_method_hint=False, prefix='v47_same_set_eval_without_method_identity')
    evaluate_router_same_set(selected, manifest_path, OUT_DIR, route_with_method_hint=True, prefix='v47_same_set_eval_with_method_identity')


def predict_once():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta_path = resolve_metadata_path(METHOD_METADATA_PATH, METHOD_CODE, SEARCH_ROOT)
    print('[INFO] using metadata:', meta_path)
    meta = load_yaml_or_json(meta_path)
    meta_vec = extract_method_meta_vector(meta)
    fp = smiles_to_fp(SMILES)
    if fp is None:
        raise ValueError(f'Invalid SMILES: {SMILES}')

    predictor = build_predictor(ROUTER_MANIFEST)
    out = predictor.predict(fp, meta_vec)
    out['method_metadata_path'] = meta_path
    out['smiles'] = SMILES

    print('[PREDICT] done')
    print(json.dumps(out, ensure_ascii=False, indent=2))

    save_path = os.path.join(OUT_DIR, 'v47_last_prediction.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('[SAVE]', save_path)


def eval_existing():
    os.makedirs(OUT_DIR, exist_ok=True)
    selected, _ = collect_selected_pairs()
    evaluate_router_same_set(selected, ROUTER_MANIFEST, OUT_DIR)


def main():
    if RUN_MODE == 'train':
        train_all()
    elif RUN_MODE == 'predict':
        predict_once()
    elif RUN_MODE == 'eval':
        eval_existing()
    elif RUN_MODE == 'family_head':
        family_head_only()
    else:
        raise ValueError(f'Unknown RUN_MODE: {RUN_MODE}')


if __name__ == '__main__':
    main()
