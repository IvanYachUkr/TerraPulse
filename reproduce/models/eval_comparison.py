#!/usr/bin/env python3
"""
Head-to-head evaluation: SSNet V3 vs V5 vs CatBoost pixel_v5.

Evaluates ALL models on the SAME test cities with the SAME WorldCover labels.
Uses the 6 test cities from the SSNet pipeline: nuremberg, ankara_test,
sofia_test, riga_test, edinburgh_test, palermo_test.

For each city, samples 500K pixels (seed=42), builds features for each model
type, and computes accuracy on the shared labels.
"""

import gc
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    SEED, N_CLASSES, CLASS_NAMES, get_test_cities, city_has_raw_tifs,
)
from reproduce.models.shared.data import (
    extract_pixels_for_city, compute_center_indices,
)

# CatBoost imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "mlp"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "pixel"))
from importlib import import_module

CKPT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")
CB_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_pixel_v5")
OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "prediction_maps")

MAX_PX = 500_000


def ts():
    return time.strftime("%H:%M:%S")


# ── SSNet helpers ────────────────────────────────────────────────────────────

def load_ssnet_v3(device):
    from reproduce.models.architectures.spectral_spatial import SpectralSpatialNetV2
    model = SpectralSpatialNetV2(
        spatial_dims=(48, 96, 192), expand_ratio=4,
        temporal_dim=192, n_attn_layers=3,
    ).to(device)
    state = torch.load(os.path.join(CKPT_DIR, "ssnet_v3_ep3_backup.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(os.path.join(CKPT_DIR, "ssnet_scaler_v3_backup.pkl"), "rb") as f:
        sc = pickle.load(f)
    return model, sc["patches"], sc["indices"]


def load_ssnet_v5(device):
    from reproduce.models.architectures.spectral_spatial_v5 import SpectralSpatialNetV5
    model = SpectralSpatialNetV5(
        n_bands=12, n_timesteps=6, n_indices=145,
        spatial_dims=(32, 64, 128), expand_ratio=4,
        temporal_dim=128, n_attn_layers=2, n_heads=4,
        n_classes=7, dropout=0.15,
        spatial_branch_drop=0.10, index_branch_drop=0.25,
    ).to(device)
    state = torch.load(os.path.join(CKPT_DIR, "ssnet_v5.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(os.path.join(CKPT_DIR, "ssnet_v5_fixed_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    return model, sc["patches"], sc["indices"]


def predict_ssnet(model, patch_scaler, idx_scaler, patches, device, is_v5=False):
    """Run SSNet on pre-extracted patches. Returns class predictions."""
    centers = patches[:, 4 * 72:5 * 72].copy()
    indices = compute_center_indices(centers)
    del centers

    patches_s = patch_scaler.transform(patches).astype(np.float32)
    indices_s = idx_scaler.transform(indices).astype(np.float32)
    del patches, indices

    n = len(patches_s)
    preds = np.empty(n, dtype=np.int64)
    BATCH = 8192

    with torch.no_grad():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            xp = torch.from_numpy(patches_s[s:e]).to(device)
            xi = torch.from_numpy(indices_s[s:e]).to(device)
            out = model(xp, xi)
            # V5 returns dict {"logits": ...}, V3 returns log_softmax tensor
            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out
            preds[s:e] = logits.argmax(dim=1).cpu().numpy()

    del patches_s, indices_s
    return preds


# ── CatBoost helpers ─────────────────────────────────────────────────────────

def load_catboost():
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(os.path.join(CB_DIR, "catboost_pixel_v5_deep_unweighted.cbm"))
    return model


def build_catboost_features_for_pixels(city, pixel_rows, pixel_cols):
    """Build CatBoost pixel_v5 features for specific pixel coords.

    Uses the same feature engineering as reproduce/pixel/02_train_catboost.py.
    """
    step2 = import_module("02_train_catboost")

    raw_dir = step2._raw_dir(city)
    H_max = pixel_rows.max() + 1
    W_max = pixel_cols.max() + 1

    YEARS = [2020, 2021]
    SEASONS = ["spring", "summer", "autumn"]
    INDEX_NAMES = step2.INDEX_NAMES

    all_bands, band_names = [], []
    indices_by_tag, sar_by_tag = {}, {}

    for year in YEARS:
        for season in SEASONS:
            tag = f"{year}_{season}"
            s2_path = os.path.join(raw_dir, f"sentinel2_{city.name}_{year}_{season}.tif")
            s1_path = os.path.join(raw_dir, f"sentinel1_{city.name}_{year}_{season}.tif")

            s2 = step2._load_tif(s2_path, step2.SENTINEL_NODATA) if os.path.exists(s2_path) else None
            if s2 is not None and s2.shape[0] >= 10:
                H_tif, W_tif = s2.shape[1], s2.shape[2]
                for bi, bn in enumerate(step2.SENTINEL_BANDS):
                    all_bands.append(s2[bi])
                    band_names.append(f"{bn}_{tag}")
                idx = step2._compute_indices(s2[:10])
                indices_by_tag[tag] = idx
                for n in INDEX_NAMES:
                    all_bands.append(idx[n])
                    band_names.append(f"{n}_{tag}")
            else:
                H_tif = int(H_max)
                W_tif = int(W_max)
                for bn in step2.SENTINEL_BANDS:
                    all_bands.append(np.full((H_tif, W_tif), np.nan, np.float32))
                    band_names.append(f"{bn}_{tag}")
                for n in INDEX_NAMES:
                    all_bands.append(np.full((H_tif, W_tif), np.nan, np.float32))
                    band_names.append(f"{n}_{tag}")

            s1 = step2._load_tif(s1_path, step2.SAR_NODATA) if os.path.exists(s1_path) else None
            if s1 is not None:
                for bi, sn in enumerate(["vv", "vh"]):
                    all_bands.append(s1[bi])
                    band_names.append(f"SAR_{sn.upper()}_{tag}")
                vvvh = np.where(np.abs(s1[1]) > 1e-10, s1[0] / s1[1], np.nan).astype(np.float32)
                all_bands.append(vvvh)
                band_names.append(f"SAR_VVVH_{tag}")
                sar_by_tag[tag] = {"vv": s1[0], "vh": s1[1]}
            else:
                for sn in ["vv", "vh"]:
                    all_bands.append(np.full((H_tif, W_tif), np.nan, np.float32))
                    band_names.append(f"SAR_{sn.upper()}_{tag}")
                all_bands.append(np.full((H_tif, W_tif), np.nan, np.float32))
                band_names.append(f"SAR_VVVH_{tag}")

    # Temporal diffs (same as 02_train_catboost.py)
    for year in YEARS:
        for sf, st in [("spring", "summer"), ("summer", "autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in indices_by_tag and tt in indices_by_tag:
                for n in INDEX_NAMES:
                    all_bands.append((indices_by_tag[tt][n] - indices_by_tag[tf][n]).astype(np.float32))
                    band_names.append(f"{n}_diff_{st}_{sf}_{year}")

    for season in SEASONS:
        t0, t1 = f"{YEARS[0]}_{season}", f"{YEARS[1]}_{season}"
        if t0 in indices_by_tag and t1 in indices_by_tag:
            for n in INDEX_NAMES:
                all_bands.append((indices_by_tag[t1][n] - indices_by_tag[t0][n]).astype(np.float32))
                band_names.append(f"{n}_interannual_{season}")

    for year in YEARS:
        ts_s, ts_a = f"{year}_spring", f"{year}_autumn"
        if ts_s in indices_by_tag and ts_a in indices_by_tag:
            for n in ["NDVI", "NDWI", "EVI2", "BSI"]:
                all_bands.append((indices_by_tag[ts_a][n] - indices_by_tag[ts_s][n]).astype(np.float32))
                band_names.append(f"{n}_range_{year}")

    for year in YEARS:
        for sf, st in [("spring", "summer"), ("summer", "autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in sar_by_tag and tt in sar_by_tag:
                for b in ["vv", "vh"]:
                    all_bands.append((sar_by_tag[tt][b] - sar_by_tag[tf][b]).astype(np.float32))
                    band_names.append(f"SAR_{b.upper()}_diff_{st}_{sf}_{year}")

    for season in SEASONS:
        t0, t1 = f"{YEARS[0]}_{season}", f"{YEARS[1]}_{season}"
        if t0 in sar_by_tag and t1 in sar_by_tag:
            for b in ["vv", "vh"]:
                all_bands.append((sar_by_tag[t1][b] - sar_by_tag[t0][b]).astype(np.float32))
                band_names.append(f"SAR_{b.upper()}_interannual_{season}")

    # Extract features only at sampled pixel positions
    n_pix = len(pixel_rows)
    n_feat = len(all_bands)
    features = np.empty((n_pix, n_feat), dtype=np.float32)
    for fi, band in enumerate(all_bands):
        features[:, fi] = band[pixel_rows, pixel_cols]

    np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    del all_bands, indices_by_tag, sar_by_tag
    gc.collect()

    return features, band_names


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*78}")
    print(f"  Head-to-Head Model Comparison on Test Cities")
    print(f"  Device: {device}")
    print(f"{'='*78}\n")

    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    print(f"  Test cities: {[c.name for c in test_cities]}")

    # Load models
    print(f"\n[{ts()}] Loading models...")
    v3_model, v3_ps, v3_is = load_ssnet_v3(device)
    print(f"  SSNet V3: {v3_model.n_params():,} params")

    v5_model, v5_ps, v5_is = load_ssnet_v5(device)
    print(f"  SSNet V5: {v5_model.n_params():,} params")

    cb_model = load_catboost()
    print(f"  CatBoost: {cb_model.tree_count_} trees")

    # Results storage
    all_y = []
    all_v3 = []
    all_v5 = []
    all_cb = []
    per_city = {}

    for city in test_cities:
        print(f"\n[{ts()}] === {city.name} ===")

        # Extract SSNet-style pixels (3x3 patches + labels + positions)
        rng = np.random.RandomState(SEED)
        result = extract_pixels_for_city(city, max_pixels=MAX_PX, pad=1, rng=rng)
        if result is None:
            print(f"  SKIP: no data")
            continue

        patches = result["feat_3x3"].astype(np.float32)
        y = result["labels"]
        n = result["n_pixels"]
        rows = result.get("rows")
        cols = result.get("cols")
        print(f"  Pixels: {n:,}")

        # SSNet V3
        print(f"  [{ts()}] SSNet V3 predicting...")
        v3_pred = predict_ssnet(v3_model, v3_ps, v3_is, patches.copy(), device, is_v5=False)

        # SSNet V5
        print(f"  [{ts()}] SSNet V5 predicting...")
        v5_pred = predict_ssnet(v5_model, v5_ps, v5_is, patches.copy(), device, is_v5=True)

        del patches
        gc.collect()

        # CatBoost - need pixel positions
        if rows is not None and cols is not None:
            print(f"  [{ts()}] CatBoost building features ({n:,} pixels)...")
            cb_features, _ = build_catboost_features_for_pixels(city, rows, cols)
            print(f"  [{ts()}] CatBoost predicting ({cb_features.shape[1]} features)...")
            cb_pred = cb_model.predict(cb_features).flatten().astype(np.int64)
            del cb_features
        else:
            print(f"  WARNING: No pixel positions - can't evaluate CatBoost")
            cb_pred = np.full(n, -1, dtype=np.int64)

        gc.collect()

        # City-level metrics
        v3_acc = accuracy_score(y, v3_pred)
        v5_acc = accuracy_score(y, v5_pred)
        cb_acc = accuracy_score(y, cb_pred) if cb_pred[0] >= 0 else float("nan")

        v3_bal = balanced_accuracy_score(y, v3_pred)
        v5_bal = balanced_accuracy_score(y, v5_pred)
        cb_bal = balanced_accuracy_score(y, cb_pred) if cb_pred[0] >= 0 else float("nan")

        per_city[city.name] = {
            "n_pixels": int(n),
            "v3_acc": float(v3_acc), "v3_bal": float(v3_bal),
            "v5_acc": float(v5_acc), "v5_bal": float(v5_bal),
            "cb_acc": float(cb_acc), "cb_bal": float(cb_bal),
        }

        print(f"  Results:  V3={v3_acc:.4f}/{v3_bal:.4f}  "
              f"V5={v5_acc:.4f}/{v5_bal:.4f}  "
              f"CB={cb_acc:.4f}/{cb_bal:.4f}  (acc/bal)")

        all_y.append(y)
        all_v3.append(v3_pred)
        all_v5.append(v5_pred)
        all_cb.append(cb_pred)

        del y, v3_pred, v5_pred, cb_pred
        gc.collect()

    # ── Aggregate ────────────────────────────────────────────────────────
    Y = np.concatenate(all_y)
    V3 = np.concatenate(all_v3)
    V5 = np.concatenate(all_v5)
    CB = np.concatenate(all_cb)

    print(f"\n{'='*78}")
    print(f"  OVERALL RESULTS ({len(Y):,} test pixels)")
    print(f"{'='*78}")

    for name, pred in [("SSNet V3 (2.8M)", V3), ("SSNet V5 (1.2M)", V5),
                       ("CatBoost pixel_v5", CB)]:
        acc = accuracy_score(Y, pred)
        bal = balanced_accuracy_score(Y, pred)
        mf1 = f1_score(Y, pred, average="macro")
        print(f"\n  {name}:")
        print(f"    Accuracy:          {acc:.4f}")
        print(f"    Balanced accuracy: {bal:.4f}")
        print(f"    Macro F1:          {mf1:.4f}")
        print(f"    Per-class recall:")
        for ci in range(N_CLASSES):
            mask = Y == ci
            if mask.sum() > 0:
                recall = (pred[mask] == ci).sum() / mask.sum()
                print(f"      {CLASS_NAMES[ci]:15s}: {recall:.4f}  ({mask.sum():>8,} px)")

    # Per-city table
    print(f"\n{'='*78}")
    print(f"  PER-CITY ACCURACY (overall / balanced)")
    print(f"{'='*78}")
    print(f"  {'City':20s}  {'SSNet V3':>16s}  {'SSNet V5':>16s}  {'CatBoost':>16s}")
    print(f"  {'-'*20}  {'-'*16}  {'-'*16}  {'-'*16}")
    for cname, m in per_city.items():
        print(f"  {cname:20s}  {m['v3_acc']:.4f}/{m['v3_bal']:.4f}  "
              f"{m['v5_acc']:.4f}/{m['v5_bal']:.4f}  "
              f"{m['cb_acc']:.4f}/{m['cb_bal']:.4f}")

    # Save results
    results = {
        "test_pixels": int(len(Y)),
        "models": {},
        "per_city": per_city,
    }
    for name, pred in [("ssnet_v3", V3), ("ssnet_v5", V5), ("catboost_v5", CB)]:
        results["models"][name] = {
            "accuracy": float(accuracy_score(Y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(Y, pred)),
            "macro_f1": float(f1_score(Y, pred, average="macro")),
            "per_class": {
                CLASS_NAMES[ci]: float((pred[Y == ci] == ci).sum() / max((Y == ci).sum(), 1))
                for ci in range(N_CLASSES)
            },
        }

    out_path = os.path.join(OUT_DIR, "model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
