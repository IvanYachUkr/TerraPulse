#!/usr/bin/env python3
"""
Train SpectralSpatialNet V8: prior-guided + center-veto ambiguity-aware classifier.

Main V8 idea:
  - keep the stable V7.1 training system
  - keep spectral priors and ambiguity-aware supervision
  - add an explicit center-only specialist that can override neighborhood smoothing
    when the center pixel looks like a real local exception (e.g. water inside grass,
    tiny river channels, isolated built-up pixel, etc.)

This is designed specifically to address the V7.1 trade-off:
  better coherent regions and boundaries, but occasional oversmoothing of small islands.
"""

import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (  # noqa: E402
    SEED, N_CLASSES, CLASS_NAMES,
    get_train_cities, get_val_cities, get_test_cities, city_has_raw_tifs,
)
from reproduce.models.shared.data import (  # noqa: E402
    extract_pixels_for_city, compute_center_indices,
)
from reproduce.models.architectures.spectral_spatial_v8 import SpectralSpatialNetV8  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")
VAL_CACHE_DIR = os.path.join(OUT_DIR, "val_raw_cache_v8")
FIXED_SCALER_PATH = os.path.join(OUT_DIR, "ssnet_v8_fixed_scaler.pkl")
BEST_MODEL_PATH = os.path.join(OUT_DIR, "ssnet_v8.pt")
ROUND_MODEL_PATH = os.path.join(OUT_DIR, "ssnet_v8_round.pt")
CKPT_PATH = os.path.join(OUT_DIR, "ssnet_v8_training.pt")
METRICS_PATH = os.path.join(OUT_DIR, "ssnet_v8_metrics.json")
FALLBACK_V7_SCALER_PATH = os.path.join(OUT_DIR, "ssnet_v7_fixed_scaler.pkl")
FALLBACK_V5_SCALER_PATH = os.path.join(OUT_DIR, "ssnet_v5_fixed_scaler.pkl")

TRAIN_PX = 60_000
VAL_PX = 5_000
PAD = 1
N_INDICES = 145
SCALE_CHUNK = 200_000
SCALER_SAMPLE_PX = 20_000


def ts():
    return time.strftime("%H:%M:%S")


def compute_indices_chunked(x_1x1: np.ndarray, chunk_size: int = 500_000) -> np.ndarray:
    n = x_1x1.shape[0]
    result = np.empty((n, N_INDICES), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        result[start:end] = compute_center_indices(x_1x1[start:end])
    return result


def load_split(cities, max_px, label="split", use_fp16=False, round_seed=None):
    all_patches, all_indices, all_center_raw, all_y = [], [], [], []

    for i, city in enumerate(cities):
        if not city_has_raw_tifs(city):
            continue
        city_seed = (round_seed + i) if round_seed is not None else (SEED + abs(hash(city.name)) % 10000)
        rng = np.random.RandomState(city_seed)
        result = extract_pixels_for_city(city, max_pixels=max_px, pad=PAD, rng=rng)
        if result is None:
            print(f"  [{i+1}/{len(cities)}] {city.name:25s} - SKIP")
            continue

        feat_3x3 = result["feat_3x3"].astype(np.float32)
        center_raw = feat_3x3[:, 4 * 72:5 * 72].copy()
        indices = compute_indices_chunked(center_raw)

        store_dtype = np.float16 if use_fp16 else np.float32
        all_patches.append(feat_3x3.astype(store_dtype))
        all_indices.append(indices.astype(store_dtype))
        all_center_raw.append(center_raw.astype(store_dtype))
        all_y.append(result["labels"])
        print(f"  [{i+1}/{len(cities)}] {city.name:25s} - {result['n_pixels']:>7,} px")
        del result, feat_3x3, center_raw, indices
        gc.collect()

    if not all_patches:
        return None, None, None, None

    patches = np.concatenate(all_patches)
    indices = np.concatenate(all_indices)
    center_raw = np.concatenate(all_center_raw)
    y = np.concatenate(all_y).astype(np.int32)
    del all_patches, all_indices, all_center_raw, all_y
    gc.collect()

    mem = (patches.nbytes + indices.nbytes + center_raw.nbytes) / 1e9
    dt = "fp16" if use_fp16 else "fp32"
    print(f"  {label}: {patches.shape[0]:,} px [{dt}], patches {patches.nbytes/1e9:.2f} GB + "
          f"indices {indices.nbytes/1e9:.2f} GB + center_raw {center_raw.nbytes/1e9:.2f} GB = {mem:.2f} GB total")
    return patches, indices, center_raw, y


def print_class_dist(label, y):
    cls, cnt = np.unique(y, return_counts=True)
    total = cnt.sum()
    dist = " ".join(f"{CLASS_NAMES[c][:4]}={100*n/total:.1f}%" for c, n in zip(cls, cnt))
    print(f"  {label}: {dist}")


def load_or_fit_fixed_scalers(train_cities):
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FIXED_SCALER_PATH):
        print(f"\n[{ts()}] Loading fixed scalers from {FIXED_SCALER_PATH}...")
        with open(FIXED_SCALER_PATH, "rb") as f:
            sc = pickle.load(f)
        return sc["patches"], sc["indices"]

    for fallback in (FALLBACK_V7_SCALER_PATH, FALLBACK_V5_SCALER_PATH):
        if os.path.exists(fallback):
            print(f"\n[{ts()}] Reusing fixed scalers from {fallback}...")
            with open(fallback, "rb") as f:
                sc = pickle.load(f)
            with open(FIXED_SCALER_PATH, "wb") as f:
                pickle.dump(sc, f)
            return sc["patches"], sc["indices"]

    print(f"\n[{ts()}] Fitting fixed scalers once on representative train subset...")
    patches, indices, center_raw, y = load_split(
        train_cities, SCALER_SAMPLE_PX, "Scaler sample", use_fp16=False,
        round_seed=SEED + 123_457,
    )
    if patches is None:
        raise RuntimeError("No data available to fit fixed scalers.")

    patch_scaler = StandardScaler()
    patch_scaler.fit(patches)
    idx_scaler = StandardScaler()
    idx_scaler.fit(indices)

    with open(FIXED_SCALER_PATH, "wb") as f:
        pickle.dump({"patches": patch_scaler, "indices": idx_scaler}, f)

    print_class_dist("Scaler sample", y)
    print(f"  Fixed scalers saved to {FIXED_SCALER_PATH}")
    del patches, indices, center_raw, y
    gc.collect()
    return patch_scaler, idx_scaler


def apply_scaler_inplace(arr, scaler, target_dtype=np.float16):
    n = arr.shape[0]
    for s in range(0, n, SCALE_CHUNK):
        e = min(s + SCALE_CHUNK, n)
        arr[s:e] = scaler.transform(arr[s:e].astype(np.float32)).astype(target_dtype)


def ensure_val_cache(val_cities):
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)
    files = ["patches.npy", "indices.npy", "center_raw.npy", "y.npy"]
    val_cache_exists = all(os.path.exists(os.path.join(VAL_CACHE_DIR, f)) for f in files)

    if val_cache_exists:
        print(f"\n[{ts()}] Loading cached val labels...")
        val_y = np.load(os.path.join(VAL_CACHE_DIR, "y.npy"))
        return val_y

    print(f"\n[{ts()}] Loading validation data (one-time)...")
    val_patches, val_indices, val_center_raw, val_y = load_split(val_cities, VAL_PX, "Val", use_fp16=False)
    if val_patches is None:
        raise RuntimeError("No validation data.")
    np.save(os.path.join(VAL_CACHE_DIR, "patches.npy"), val_patches)
    np.save(os.path.join(VAL_CACHE_DIR, "indices.npy"), val_indices)
    np.save(os.path.join(VAL_CACHE_DIR, "center_raw.npy"), val_center_raw)
    np.save(os.path.join(VAL_CACHE_DIR, "y.npy"), val_y)
    del val_patches, val_indices, val_center_raw
    gc.collect()
    print(f"  Val raw cached to {VAL_CACHE_DIR}")
    return val_y


def scale_val_to_cpu(patch_scaler, idx_scaler):
    val_p = np.load(os.path.join(VAL_CACHE_DIR, "patches.npy"))
    val_i = np.load(os.path.join(VAL_CACHE_DIR, "indices.npy"))
    val_c = np.load(os.path.join(VAL_CACHE_DIR, "center_raw.npy"))
    val_patches = torch.from_numpy(patch_scaler.transform(val_p).astype(np.float32))
    val_indices = torch.from_numpy(idx_scaler.transform(val_i).astype(np.float32))
    val_center = torch.from_numpy(val_c.astype(np.float32))
    del val_p, val_i, val_c
    gc.collect()
    return val_patches, val_indices, val_center


def random_d4_augment_flat(patches, n_timesteps=6, n_bands=12, p=0.60):
    if torch.rand((), device=patches.device).item() >= p:
        return patches

    n = patches.shape[0]
    x = patches.reshape(n, 9, n_timesteps, n_bands)
    x = x.permute(0, 2, 3, 1).contiguous().reshape(n, n_timesteps, n_bands, 3, 3)

    k = int(torch.randint(0, 4, (1,), device=patches.device).item())
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if torch.rand((), device=patches.device).item() < 0.5:
        x = torch.flip(x, dims=(-1,))

    x = x.reshape(n, n_timesteps, n_bands, 9)
    x = x.permute(0, 3, 1, 2).contiguous().reshape(n, -1)
    return x


def confusion_from_preds(pred, target, n_classes):
    idx = target * n_classes + pred
    cm = torch.bincount(idx, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes).cpu().numpy()


def metrics_from_confusion(conf):
    total = conf.sum()
    tp = np.diag(conf).astype(np.float64)
    row_sum = conf.sum(axis=1).astype(np.float64)
    col_sum = conf.sum(axis=0).astype(np.float64)

    overall = float(tp.sum() / max(total, 1))
    valid = row_sum > 0
    recall = np.zeros_like(tp)
    precision = np.zeros_like(tp)
    f1 = np.zeros_like(tp)

    recall[valid] = tp[valid] / row_sum[valid]
    precision[col_sum > 0] = tp[col_sum > 0] / col_sum[col_sum > 0]
    denom = precision + recall
    good = denom > 0
    f1[good] = 2.0 * precision[good] * recall[good] / denom[good]

    balanced = float(recall[valid].mean()) if valid.any() else 0.0
    macro_f1 = float(f1[valid].mean()) if valid.any() else 0.0
    score = float((2.0 * overall * balanced) / max(overall + balanced, 1e-12))

    hard_idx = [i for i, n in enumerate(CLASS_NAMES) if n in ("shrubland", "bare_sparse")]
    hard_recall = float(np.mean([recall[i] for i in hard_idx])) if hard_idx else 0.0
    priority_idx = [i for i, n in enumerate(CLASS_NAMES) if n in ("shrubland", "bare_sparse", "built_up", "water")]
    priority_recall = float(np.mean([recall[i] for i in priority_idx])) if priority_idx else 0.0
    focus_score = 0.60 * score + 0.25 * hard_recall + 0.15 * priority_recall

    return {
        "overall_acc": overall,
        "balanced_acc": balanced,
        "macro_f1": macro_f1,
        "score": score,
        "hard_recall": hard_recall,
        "priority_recall": priority_recall,
        "focus_score": focus_score,
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
    }


def validate(model, val_patches, val_indices, val_center, val_y_gpu, n_val, n_classes, use_amp, device):
    model.eval()
    loss_sum = 0.0
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    vb = 16384

    with torch.no_grad():
        for vs in range(0, n_val, vb):
            ve = min(vs + vb, n_val)
            vp = val_patches[vs:ve].to(device, non_blocking=True)
            vi = val_indices[vs:ve].to(device, non_blocking=True)
            vc = val_center[vs:ve].to(device, non_blocking=True)
            vy = val_y_gpu[vs:ve]
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out = model(vp, vi, vc)
                logits = out["logits"]
                loss_sum += F.cross_entropy(logits, vy, reduction="sum").item()
            pred = logits.argmax(dim=1)
            conf += confusion_from_preds(pred, vy, n_classes)
            del vp, vi, vc, out, logits, pred

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = metrics_from_confusion(conf)
    metrics["loss"] = loss_sum / n_val
    return metrics


class PriorGuidedCenterCriterion:
    def __init__(self,
                 n_classes: int,
                 class_names,
                 label_smoothing: float = 0.02,
                 anchor_conf_thresh: float = 0.88,
                 center_conf_thresh: float = 0.72,
                 agree_alpha: float = 0.04,
                 disagree_alpha_max: float = 0.18,
                 center_agree_alpha: float = 0.05,
                 center_disagree_alpha_max: float = 0.28,
                 max_total_alpha: float = 0.32,
                 min_weight: float = 0.38,
                 boundary_downweight: float = 0.32,
                 ambiguity_downweight: float = 0.20,
                 anomaly_upweight: float = 0.16,
                 trusted_boundary_max: float = 0.55,
                 trusted_ambiguity_max: float = 0.55,
                 anomaly_trust_min: float = 0.64,
                 prior_aux_weight: float = 0.03,
                 spatial_aux_weight: float = 0.05,
                 index_aux_weight: float = 0.05,
                 center_aux_weight: float = 0.07,
                 ambiguity_aux_weight: float = 0.04,
                 gate_aux_weight: float = 0.03):
        self.n_classes = n_classes
        self.class_names = list(class_names)
        self.label_smoothing = label_smoothing
        self.anchor_conf_thresh = anchor_conf_thresh
        self.center_conf_thresh = center_conf_thresh
        self.agree_alpha = agree_alpha
        self.disagree_alpha_max = disagree_alpha_max
        self.center_agree_alpha = center_agree_alpha
        self.center_disagree_alpha_max = center_disagree_alpha_max
        self.max_total_alpha = max_total_alpha
        self.min_weight = min_weight
        self.boundary_downweight = boundary_downweight
        self.ambiguity_downweight = ambiguity_downweight
        self.anomaly_upweight = anomaly_upweight
        self.trusted_boundary_max = trusted_boundary_max
        self.trusted_ambiguity_max = trusted_ambiguity_max
        self.anomaly_trust_min = anomaly_trust_min
        self.prior_aux_weight = prior_aux_weight
        self.spatial_aux_weight = spatial_aux_weight
        self.index_aux_weight = index_aux_weight
        self.center_aux_weight = center_aux_weight
        self.ambiguity_aux_weight = ambiguity_aux_weight
        self.gate_aux_weight = gate_aux_weight

    def _smoothed_one_hot(self, y: torch.Tensor) -> torch.Tensor:
        off = self.label_smoothing / self.n_classes
        on = 1.0 - self.label_smoothing + off
        target = torch.full((y.size(0), self.n_classes), off, device=y.device, dtype=torch.float32)
        target.scatter_(1, y.unsqueeze(1), on)
        return target

    def _soft_ce(self, logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        loss = -(target * logp).sum(dim=1)
        if weight is not None:
            loss = loss * weight
            return loss.sum() / weight.sum().clamp_min(1e-6)
        return loss.mean()

    def _weighted_ce(self, logits: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.cross_entropy(logits, y, reduction="none")
        if weight is not None:
            loss = loss * weight
            return loss.sum() / weight.sum().clamp_min(1e-6)
        return loss.mean()

    def __call__(self, outputs: Dict[str, torch.Tensor], y: torch.Tensor):
        logits = outputs["logits"]
        ambiguity = outputs["ambiguity"].detach()
        boundary = outputs["boundary_score"].detach()
        anomaly = outputs["anomaly_score"].detach()

        anchor_probs = outputs["anchor_probs"].detach()
        anchor_conf = outputs["anchor_conf"].detach()
        anchor_pred = outputs["anchor_pred"].detach()

        center_probs = F.softmax(outputs["center_branch_logits"].detach(), dim=1)
        center_conf = outputs["center_conf"].detach()
        center_pred = center_probs.argmax(dim=1)

        anchor_mask = anchor_conf >= self.anchor_conf_thresh
        center_mask = center_conf >= self.center_conf_thresh
        agree_anchor = anchor_mask & anchor_pred.eq(y)
        disagree_anchor = anchor_mask & (~anchor_pred.eq(y))
        agree_center = center_mask & center_pred.eq(y)
        disagree_center = center_mask & (~center_pred.eq(y))

        hard_target = self._smoothed_one_hot(y)

        alpha_anchor = torch.zeros(anchor_conf.shape, device=anchor_conf.device, dtype=torch.float32)
        alpha_anchor[agree_anchor] = self.agree_alpha * anchor_conf[agree_anchor].float()
        alpha_anchor[disagree_anchor] = self.disagree_alpha_max * anchor_conf[disagree_anchor].float() * (0.35 + 0.65 * ambiguity[disagree_anchor].float())

        alpha_center = torch.zeros(center_conf.shape, device=center_conf.device, dtype=torch.float32)
        alpha_center[agree_center] = self.center_agree_alpha * center_conf[agree_center].float() * (0.35 + 0.65 * anomaly[agree_center].float())
        alpha_center[disagree_center] = self.center_disagree_alpha_max * center_conf[disagree_center].float() * anomaly[disagree_center].float()

        total_alpha = (alpha_anchor + alpha_center).clamp(max=self.max_total_alpha)

        mixed_probs = hard_target.clone()
        for source_alpha, source_probs in ((alpha_anchor, anchor_probs.float()), (alpha_center, center_probs.float())):
            mixed_probs = mixed_probs * (1.0 - source_alpha.unsqueeze(1)) + source_probs * source_alpha.unsqueeze(1)
        mixed_probs = mixed_probs / mixed_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

        sample_weight = (1.0 - self.boundary_downweight * boundary.float() - self.ambiguity_downweight * ambiguity.float() + self.anomaly_upweight * anomaly.float())
        sample_weight[disagree_anchor] = sample_weight[disagree_anchor] * (1.0 - 0.15 * anchor_conf[disagree_anchor].float())
        # For anomaly pixels, keep training pressure so small real exceptions are not smoothed away.
        sample_weight = sample_weight.clamp(min=self.min_weight, max=1.10)

        main = self._soft_ce(logits, mixed_probs, weight=sample_weight)

        trusted_mask = ((boundary < self.trusted_boundary_max) & (ambiguity < self.trusted_ambiguity_max)) | ((anomaly > self.anomaly_trust_min) & center_mask)
        trusted_weight = trusted_mask.float()
        spatial_aux = self._weighted_ce(outputs["spatial_logits"], y, weight=trusted_weight) if trusted_weight.sum() > 0 else logits.new_tensor(0.0)
        index_aux = self._weighted_ce(outputs["index_logits"], y, weight=trusted_weight) if trusted_weight.sum() > 0 else logits.new_tensor(0.0)
        center_aux_weight = torch.maximum(trusted_weight, ((anomaly > self.anomaly_trust_min) & center_mask).float())
        center_aux = self._weighted_ce(outputs["center_logits"], y, weight=center_aux_weight) if center_aux_weight.sum() > 0 else logits.new_tensor(0.0)

        prior_aux = logits.new_tensor(0.0)
        if anchor_mask.any():
            prior_aux = self._weighted_ce(outputs["prior_logits"].float(), anchor_pred, weight=anchor_conf.float() * anchor_mask.float())

        ambiguity_target = torch.clamp(0.55 * boundary.float() + 0.20 * anomaly.float() + 0.25 * torch.maximum(disagree_anchor.float() * anchor_conf.float(), disagree_center.float() * center_conf.float()), 0.0, 1.0)
        ambiguity_aux = F.binary_cross_entropy_with_logits(outputs["ambiguity_logit"].float(), ambiguity_target)

        prior_gate_target = torch.clamp(0.70 * (anchor_mask.float() * anchor_conf.float()) + 0.20 * boundary.float() + 0.10 * ambiguity.float(), 0.0, 1.0)
        center_gate_target = torch.clamp(0.70 * (center_mask.float() * center_conf.float()) * anomaly.float() + 0.20 * anomaly.float() + 0.10 * (1.0 - boundary.float()), 0.0, 1.0)
        gate_aux = (
            F.binary_cross_entropy_with_logits(outputs["prior_gate_logit"].float(), prior_gate_target)
            + F.binary_cross_entropy_with_logits(outputs["center_gate_logit"].float(), center_gate_target)
        ) * 0.5

        total = (
            main
            + self.prior_aux_weight * prior_aux
            + self.spatial_aux_weight * spatial_aux
            + self.index_aux_weight * index_aux
            + self.center_aux_weight * center_aux
            + self.ambiguity_aux_weight * ambiguity_aux
            + self.gate_aux_weight * gate_aux
        )
        stats = {
            "main": float(main.detach().cpu()),
            "prior_aux": float(prior_aux.detach().cpu()),
            "spatial_aux": float(spatial_aux.detach().cpu()),
            "index_aux": float(index_aux.detach().cpu()),
            "center_aux": float(center_aux.detach().cpu()),
            "ambiguity_aux": float(ambiguity_aux.detach().cpu()),
            "gate_aux": float(gate_aux.detach().cpu()),
            "alpha": float(total_alpha.mean().detach().cpu()),
            "alpha_nz": float((total_alpha > 1e-8).float().mean().detach().cpu()),
            "anchor_conf": float(anchor_conf.mean().detach().cpu()),
            "anchor_frac": float(anchor_mask.float().mean().detach().cpu()),
            "center_conf": float(center_conf.mean().detach().cpu()),
            "center_frac": float(center_mask.float().mean().detach().cpu()),
            "relabel_frac": float(torch.maximum(disagree_anchor.float(), disagree_center.float()).mean().detach().cpu()),
            "boundary": float(boundary.mean().detach().cpu()),
            "anomaly": float(anomaly.mean().detach().cpu()),
            "ambiguity": float(outputs["ambiguity"].mean().detach().cpu()),
            "weight": float(sample_weight.mean().detach().cpu()),
            "prior_gate": float(outputs["prior_gate"].mean().detach().cpu()),
            "center_gate": float(outputs["center_gate"].mean().detach().cpu()),
        }
        return total, stats


def maybe_load_initial_weights(model, init_path: Optional[str], device: str) -> int:
    if not init_path or (not os.path.exists(init_path)):
        return 0
    state = torch.load(init_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model_state = model.state_dict()
    matched = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched[k] = v
    if not matched:
        return 0
    model_state.update(matched)
    model.load_state_dict(model_state)
    return len(matched)


def evaluate_test(model, patch_scaler, idx_scaler, device):
    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    if not test_cities:
        print("  No test cities")
        return None

    all_correct, all_total = 0, 0
    class_correct = np.zeros(N_CLASSES, dtype=np.int64)
    class_total = np.zeros(N_CLASSES, dtype=np.int64)
    per_city = {}
    model.eval()

    for city in test_cities:
        rng = np.random.RandomState(SEED)
        result = extract_pixels_for_city(city, max_pixels=500_000, pad=PAD, rng=rng)
        if result is None:
            print(f"  {city.name:25s} - SKIP")
            continue

        patches = result["feat_3x3"].astype(np.float32)
        center_raw = patches[:, 4 * 72:5 * 72].copy().astype(np.float32)
        y = result["labels"]
        n = result["n_pixels"]
        del result
        gc.collect()

        indices = compute_indices_chunked(center_raw)
        patches = patch_scaler.transform(patches).astype(np.float32)
        indices = idx_scaler.transform(indices).astype(np.float32)

        batch = 16384
        preds = []
        with torch.no_grad():
            for s in range(0, n, batch):
                xp = torch.from_numpy(patches[s:s+batch]).to(device)
                xi = torch.from_numpy(indices[s:s+batch]).to(device)
                xc = torch.from_numpy(center_raw[s:s+batch]).to(device)
                pb = model.predict(xp, xi, xc).cpu().numpy()
                preds.append(pb.argmax(1))
        pred_classes = np.concatenate(preds)

        correct = (pred_classes == y).sum()
        city_acc = correct / n
        per_city[city.name] = {"accuracy": float(city_acc), "n_pixels": int(n)}
        all_correct += correct
        all_total += n
        for ci in range(N_CLASSES):
            mask = y == ci
            class_total[ci] += mask.sum()
            class_correct[ci] += (pred_classes[mask] == ci).sum()
        del patches, indices, center_raw, preds, pred_classes, y
        gc.collect()
        print(f"  {city.name:25s} - {n:>9,} px, acc={city_acc:.4f}")

    overall = all_correct / max(all_total, 1)
    per_class = {}
    print(f"\n  Test Overall: {overall:.4f} ({overall*100:.2f}%)")
    for ci in range(N_CLASSES):
        if class_total[ci] > 0:
            acc = class_correct[ci] / class_total[ci]
            per_class[CLASS_NAMES[ci]] = float(acc)
            print(f"    {CLASS_NAMES[ci]:>15}: {acc:.4f} ({class_total[ci]:,} px)")

    return {"overall_accuracy": float(overall), "per_class": per_class, "per_city": per_city}


def train(device, n_rounds, max_epochs_per_round, batch_size, lr,
          weight_decay, inner_patience, resume=False,
          ckpt_metric="score", init_path: Optional[str] = None):
    train_cities = get_train_cities()
    val_cities = get_val_cities()

    print(f"\n{'='*88}")
    print("  SpectralSpatialNet V8 — prior-guided + center-veto ambiguity-aware training")
    print(f"  Rounds: {n_rounds}, Max epochs/round: {max_epochs_per_round}")
    print(f"  Train: {TRAIN_PX:,} px/city, Val: {VAL_PX:,} px/city")
    print(f"  Inner patience: {inner_patience}, Batch: {batch_size}")
    print(f"  Device: {device}, CKPT metric: {ckpt_metric}")
    print(f"{'='*88}")

    patch_scaler, idx_scaler = load_or_fit_fixed_scalers(train_cities)
    val_y = ensure_val_cache(val_cities)
    n_val = len(val_y)
    val_y_gpu = torch.from_numpy(val_y.astype(np.int64)).to(device)
    print_class_dist("Val", val_y)
    del val_y
    gc.collect()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    model = SpectralSpatialNetV8(
        n_bands=12,
        n_timesteps=6,
        n_indices=N_INDICES,
        spatial_dims=(32, 64, 128),
        expand_ratio=4,
        temporal_dim=128,
        n_attn_layers=2,
        n_heads=4,
        n_classes=N_CLASSES,
        dropout=0.12,
        prior_hidden=96,
    ).to(device)
    use_amp = device == "cuda"
    n_params = model.n_params()
    print(f"\n  Model: {n_params:,} parameters")

    start_round = 0
    best_metric = 0.0
    best_score = 0.0
    best_val_acc = 0.0
    best_bal_acc = 0.0

    if resume and os.path.exists(CKPT_PATH):
        print(f"  Resuming from {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_round = ckpt.get("round_idx", 0) + 1
        best_metric = ckpt.get("best_metric", 0.0)
        best_score = ckpt.get("best_score", 0.0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_bal_acc = ckpt.get("best_bal_acc", 0.0)
        print(f"  Resuming from round {start_round}, best_metric={best_metric:.4f}")
        del ckpt
        gc.collect()
    elif resume and os.path.exists(BEST_MODEL_PATH):
        print(f"  Loading model weights from {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    elif init_path:
        matched = maybe_load_initial_weights(model, init_path, device)
        if matched:
            print(f"  Warm-started from {init_path} with {matched} matching tensors")
        else:
            print("  Init path provided but no matching tensors loaded.")
    else:
        print("  Training from scratch (no warm start).")

    criterion = PriorGuidedCenterCriterion(
        n_classes=N_CLASSES,
        class_names=CLASS_NAMES,
        label_smoothing=0.02,
        anchor_conf_thresh=0.88,
        center_conf_thresh=0.72,
        agree_alpha=0.04,
        disagree_alpha_max=0.18,
        center_agree_alpha=0.05,
        center_disagree_alpha_max=0.28,
        max_total_alpha=0.32,
        min_weight=0.38,
        boundary_downweight=0.32,
        ambiguity_downweight=0.20,
        anomaly_upweight=0.16,
        trusted_boundary_max=0.55,
        trusted_ambiguity_max=0.55,
        anomaly_trust_min=0.64,
        prior_aux_weight=0.03,
        spatial_aux_weight=0.05,
        index_aux_weight=0.05,
        center_aux_weight=0.07,
        ambiguity_aux_weight=0.04,
        gate_aux_weight=0.03,
    )

    total_epochs_trained = 0
    val_patches_cpu, val_indices_cpu, val_center_cpu = scale_val_to_cpu(patch_scaler, idx_scaler)
    print(f"  Val on CPU: {val_patches_cpu.shape}, center_raw {val_center_cpu.shape} (streamed to GPU during validation)")

    for round_idx in range(start_round, n_rounds):
        print(f"\n{'='*88}")
        print(f"  ROUND {round_idx+1}/{n_rounds}")
        print(f"{'='*88}")

        round_seed = SEED + round_idx * 7919
        print(f"\n[{ts()}] Loading training data (round seed={round_seed})...")
        train_patches, train_indices, train_center_raw, train_y = load_split(
            train_cities, TRAIN_PX, f"Train R{round_idx+1}", use_fp16=True, round_seed=round_seed,
        )
        if train_patches is None:
            print("ERROR: No train data!")
            continue

        n_train = len(train_y)
        print_class_dist(f"Train R{round_idx+1}", train_y)

        print(f"\n[{ts()}] Applying fixed scalers...")
        apply_scaler_inplace(train_patches, patch_scaler, target_dtype=np.float16)
        apply_scaler_inplace(train_indices, idx_scaler, target_dtype=np.float16)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        grad_scaler = torch.amp.GradScaler(enabled=use_amp)

        steps_per_epoch = (n_train + batch_size - 1) // batch_size
        round_total_steps = max_epochs_per_round * steps_per_epoch
        warmup_steps = steps_per_epoch
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.20, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(round_total_steps - warmup_steps, 1)),
            ],
            milestones=[warmup_steps],
        )

        round_best_metric = 0.0
        round_best_metrics = None
        wait = 0
        min_epochs = 4

        train_center_raw_f32 = train_center_raw.astype(np.float32)
        del train_center_raw
        gc.collect()

        print(f"\n[{ts()}] Training round {round_idx+1} (max {max_epochs_per_round} epochs, patience={inner_patience})...\n")

        for epoch in range(max_epochs_per_round):
            epoch_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model.train()
            perm = np.random.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            running = {k: 0.0 for k in [
                "main", "prior_aux", "spatial_aux", "index_aux", "center_aux", "ambiguity_aux", "gate_aux",
                "alpha", "alpha_nz", "anchor_conf", "anchor_frac", "center_conf", "center_frac", "relabel_frac",
                "boundary", "anomaly", "ambiguity", "weight", "prior_gate", "center_gate",
            ]}

            batch_iter = tqdm(
                range(0, n_train, batch_size),
                desc=f"  R{round_idx+1} Ep{epoch:02d}",
                total=steps_per_epoch,
                ncols=150,
                leave=False,
                file=sys.stderr,
                mininterval=10,
            )

            for start in batch_iter:
                idx = perm[start:start + batch_size]
                xp = torch.from_numpy(train_patches[idx]).to(device, non_blocking=True)
                xi = torch.from_numpy(train_indices[idx]).to(device, non_blocking=True)
                xc = torch.from_numpy(train_center_raw_f32[idx]).to(device, non_blocking=True)
                yb = torch.from_numpy(train_y[idx].astype(np.int64)).to(device, non_blocking=True)
                if xp.size(0) < 2:
                    continue

                xp_student = random_d4_augment_flat(xp.float().clone(), n_timesteps=6, n_bands=12, p=0.60)
                xi_student = xi.float()

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    outputs = model(xp_student, xi_student, xc)
                    loss, loss_stats = criterion(outputs, yb)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1
                for k in running:
                    running[k] += loss_stats[k]
                del xp, xi, xc, yb, xp_student, xi_student, outputs, loss, loss_stats

            epoch_end_time = time.time()
            epoch_wall = epoch_end_time - epoch_start_time

            val_metrics = validate(
                model, val_patches_cpu, val_indices_cpu, val_center_cpu, val_y_gpu,
                n_val, N_CLASSES, use_amp, device,
            )
            total_epochs_trained += 1

            if torch.cuda.is_available():
                mem_alloc = torch.cuda.max_memory_allocated() / 1e9
                mem_reserved = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()
                try:
                    smi = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5,
                    )
                    if smi.returncode == 0:
                        parts = smi.stdout.strip().split(',')
                        smi_used = float(parts[0].strip()) / 1024
                        smi_total = float(parts[1].strip()) / 1024
                        smi_info = f" smi={smi_used:.1f}/{smi_total:.1f}G"
                    else:
                        smi_info = ""
                except Exception:
                    smi_info = ""
            else:
                mem_alloc = mem_reserved = 0.0
                smi_info = ""

            this_metric = val_metrics[ckpt_metric]
            improved_round = this_metric > round_best_metric
            improved_global = this_metric > best_metric

            if improved_round:
                round_best_metric = this_metric
                round_best_metrics = dict(val_metrics)
                torch.save(model.state_dict(), ROUND_MODEL_PATH)
                wait = 0
            else:
                wait += 1

            if improved_global:
                best_metric = this_metric
                best_score = val_metrics["score"]
                best_val_acc = val_metrics["overall_acc"]
                best_bal_acc = val_metrics["balanced_acc"]
                os.makedirs(OUT_DIR, exist_ok=True)
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                torch.save({
                    "model": model.state_dict(),
                    "round_idx": round_idx,
                    "best_metric": best_metric,
                    "best_score": best_score,
                    "best_val_acc": best_val_acc,
                    "best_bal_acc": best_bal_acc,
                }, CKPT_PATH)

            avg = epoch_loss / max(n_batches, 1)
            marker = " ** GLOBAL" if improved_global else (" *" if improved_round else "")
            print(
                f"  R{round_idx+1} Ep{epoch:02d}: "
                f"loss={avg:.5f} "
                f"main={running['main']/max(n_batches,1):.4f} "
                f"prior={running['prior_aux']/max(n_batches,1):.4f} "
                f"sp={running['spatial_aux']/max(n_batches,1):.4f} "
                f"idx={running['index_aux']/max(n_batches,1):.4f} "
                f"ctr={running['center_aux']/max(n_batches,1):.4f} "
                f"amb_aux={running['ambiguity_aux']/max(n_batches,1):.4f} "
                f"alpha={running['alpha']/max(n_batches,1):.3f} "
                f"alpha_nz={running['alpha_nz']/max(n_batches,1):.3f} "
                f"anc={running['anchor_conf']/max(n_batches,1):.3f} "
                f"anc_frac={running['anchor_frac']/max(n_batches,1):.3f} "
                f"ctr_conf={running['center_conf']/max(n_batches,1):.3f} "
                f"ctr_frac={running['center_frac']/max(n_batches,1):.3f} "
                f"relbl={running['relabel_frac']/max(n_batches,1):.3f} "
                f"bnd={running['boundary']/max(n_batches,1):.3f} "
                f"anom={running['anomaly']/max(n_batches,1):.3f} "
                f"amb={running['ambiguity']/max(n_batches,1):.3f} "
                f"w={running['weight']/max(n_batches,1):.3f} "
                f"pg={running['prior_gate']/max(n_batches,1):.3f} "
                f"cg={running['center_gate']/max(n_batches,1):.3f} | "
                f"val_loss={val_metrics['loss']:.5f} "
                f"acc={val_metrics['overall_acc']:.4f} "
                f"bal={val_metrics['balanced_acc']:.4f} "
                f"mf1={val_metrics['macro_f1']:.4f} "
                f"hard={val_metrics['hard_recall']:.4f} "
                f"prio={val_metrics['priority_recall']:.4f} "
                f"focus={val_metrics['focus_score']:.4f} "
                f"score={val_metrics['score']:.4f} "
                f"(best_round={round_best_metric:.4f} best_global={best_metric:.4f}) "
                f"wait={wait} | {epoch_wall:.0f}s alloc={mem_alloc:.2f}G rsv={mem_reserved:.2f}G{smi_info}{marker}"
            )

            if epoch >= min_epochs and wait >= inner_patience:
                print(f"  Early stop round {round_idx+1} at epoch {epoch}")
                break

        if os.path.exists(ROUND_MODEL_PATH):
            best_round_state = torch.load(ROUND_MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(best_round_state)

        del train_patches, train_indices, train_center_raw_f32, train_y
        del optimizer, grad_scaler, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if round_best_metrics is None:
            round_best_metrics = validate(model, val_patches_cpu, val_indices_cpu, val_center_cpu, val_y_gpu, n_val, N_CLASSES, use_amp, device)

        print(f"\n  Round {round_idx+1} done: "
              f"acc={round_best_metrics['overall_acc']:.4f}, "
              f"bal={round_best_metrics['balanced_acc']:.4f}, "
              f"hard={round_best_metrics['hard_recall']:.4f}, "
              f"score={round_best_metrics['score']:.4f}, "
              f"focus={round_best_metrics['focus_score']:.4f}, "
              f"best_global={best_score:.4f}, total_epochs={total_epochs_trained}")

    print(f"\n{'='*88}")
    print(f"  All {n_rounds} rounds complete.")
    print(f"  Best metric ({ckpt_metric}): {best_metric:.4f}")
    print(f"  Best score: {best_score:.4f}")
    print(f"  Total epochs trained: {total_epochs_trained}")
    print(f"{'='*88}")

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    val_metrics = validate(model, val_patches_cpu, val_indices_cpu, val_center_cpu, val_y_gpu, n_val, N_CLASSES, use_amp, device)

    result = {
        "model": "ssnet_v8_prior_guided_center_veto",
        "n_params": n_params,
        "n_rounds": n_rounds,
        "total_epochs": total_epochs_trained,
        "val_accuracy": float(val_metrics["overall_acc"]),
        "val_balanced_acc": float(val_metrics["balanced_acc"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_score": float(val_metrics["score"]),
        "val_focus_score": float(val_metrics["focus_score"]),
        "best_metric": float(best_metric),
        "best_score": float(best_score),
    }
    print(f"\n  Val accuracy (best model): {val_metrics['overall_acc']:.4f} ({val_metrics['overall_acc']*100:.2f}%)")
    for ci in range(N_CLASSES):
        result[f"recall_{CLASS_NAMES[ci]}"] = float(val_metrics["per_class_recall"][ci])
        result[f"precision_{CLASS_NAMES[ci]}"] = float(val_metrics["per_class_precision"][ci])
        result[f"f1_{CLASS_NAMES[ci]}"] = float(val_metrics["per_class_f1"][ci])
        print(f"    {CLASS_NAMES[ci]:>15}: recall={val_metrics['per_class_recall'][ci]:.4f} "
              f"precision={val_metrics['per_class_precision'][ci]:.4f} f1={val_metrics['per_class_f1'][ci]:.4f}")

    print(f"\n[{ts()}] Evaluating on test cities...")
    test_result = evaluate_test(model, patch_scaler, idx_scaler, device)
    if test_result:
        result["test_accuracy"] = test_result["overall_accuracy"]
        result["test_per_class"] = test_result["per_class"]
        result["test_per_city"] = test_result["per_city"]

    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="Train SpectralSpatialNet V8")
    parser.add_argument("--n-rounds", type=int, default=8, help="Number of data resampling rounds")
    parser.add_argument("--max-epochs", type=int, default=20, help="Max epochs per round")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--inner-patience", type=int, default=4, help="Early stop patience within each round")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--ckpt-metric", type=str, default="score",
                        choices=["score", "focus_score", "overall_acc", "balanced_acc", "macro_f1"],
                        help="Validation metric used for checkpoint selection")
    parser.add_argument("--init-path", type=str, default=None,
                        help="Optional checkpoint to warm-start shared weights; default is None (train from scratch)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    result = train(
        device,
        args.n_rounds,
        args.max_epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.inner_patience,
        resume=args.resume,
        ckpt_metric=args.ckpt_metric,
        init_path=args.init_path,
    )

    if result:
        print(f"\n{'='*88}")
        print("  DONE: SpectralSpatialNet V8")
        print(f"  Val:  {result['val_accuracy']:.4f}")
        if "test_accuracy" in result:
            print(f"  Test: {result['test_accuracy']:.4f}")
        print(f"{'='*88}")


if __name__ == "__main__":
    main()
