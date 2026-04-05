#!/usr/bin/env python3
"""
3-way Ensemble: CatBoost V5 + MLP 3x3 + TempCNN on SAME aligned pixels.

Tests probability blending weights across all 3 models on the
intersection of valid pixels for each test city.
"""
import sys, os, gc, time, pickle, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
PROJECT = r"c:\Users\vanya\Documents\ML_1_sem\final"
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "reproduce", "models"))
sys.stdout.reconfigure(line_buffering=True)

from catboost import CatBoostClassifier
from reproduce.models.architectures.mlp import build_model as build_mlp
from reproduce.models.architectures.tempcnn import build_tempcnn
from reproduce.models.shared.config import (
    get_test_cities, city_has_raw_tifs, N_CLASSES, CLASS_NAMES, SEED
)
from reproduce.models.shared.data import load_raw_feature_cube, load_pixel_labels

N_RAW = 72
PAD = 1
MAX_PX = 200_000

CKPT = os.path.join(PROJECT, "reproduce", "models", "checkpoints")
CATBOOST_PATH = os.path.join(PROJECT, "data", "cities", "models_pixel_v5",
                             "catboost_pixel_v5_deep_unweighted.cbm")
MLP_CKPT = os.path.join(CKPT, "mlp_3x3.pt")
MLP_SCALER = os.path.join(CKPT, "mlp_3x3_scaler.pkl")
TCNN_CKPT = os.path.join(CKPT, "tempcnn_1x1.pt")
TCNN_SCALER = os.path.join(CKPT, "tempcnn_1x1_scaler.pkl")

# Import CatBoost feature builder
sys.path.insert(0, os.path.join(PROJECT, "reproduce", "pixel"))
sys.path.insert(0, os.path.join(PROJECT, "reproduce", "mlp"))
from importlib import import_module
cb_mod = import_module("02_train_catboost")
step1 = import_module("01_download_data")
CITY_MAP = step1.CITY_MAP

N_BANDS = 12
N_SLOTS = 6


def ts():
    return time.strftime("%H:%M:%S")


def build_cb_spatial_cube(city):
    """Build CatBoost features as (H, W, F) cube."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _city = CITY_MAP.get(city.name)
    if _city is None:
        return None, None, None

    labels = cb_mod.load_worldcover_pixels(_city)
    if labels is None:
        return None, None, None
    H, W = labels.shape
    raw = cb_mod._raw_dir(_city)

    tasks = {}
    for year in cb_mod.YEARS:
        for season in cb_mod.SEASONS:
            tag = f"{year}_{season}"
            tasks[f"s2_{tag}"] = (os.path.join(raw, f"sentinel2_{_city.name}_{tag}.tif"), cb_mod.SENTINEL_NODATA)
            tasks[f"s1_{tag}"] = (os.path.join(raw, f"sentinel1_{_city.name}_{tag}.tif"), cb_mod.SAR_NODATA)

    tifs = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(cb_mod._load_tif, p, nd): k for k, (p, nd) in tasks.items()}
        for f in as_completed(futures):
            tifs[futures[f]] = f.result()

    all_bands, indices_by_tag, sar_by_tag = [], {}, {}
    has_any = False

    for year in cb_mod.YEARS:
        for season in cb_mod.SEASONS:
            tag = f"{year}_{season}"
            s2 = tifs.get(f"s2_{tag}")
            if s2 is not None and s2.shape[0] >= 11:
                has_any = True
                for bi in range(10):
                    all_bands.append(s2[bi])
                idx = cb_mod._compute_indices(s2[:10])
                indices_by_tag[tag] = idx
                for n in cb_mod.INDEX_NAMES:
                    all_bands.append(idx[n])
            else:
                for _ in range(10 + 9):
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))

            s1 = tifs.get(f"s1_{tag}")
            if s1 is not None:
                for bi in range(2):
                    all_bands.append(s1[bi])
                vvvh = np.where(np.abs(s1[1]) > 1e-10, s1[0]/s1[1], np.nan).astype(np.float32)
                all_bands.append(vvvh)
                sar_by_tag[tag] = {"vv": s1[0], "vh": s1[1]}
            else:
                for _ in range(3):
                    all_bands.append(np.full((H, W), np.nan, dtype=np.float32))

    if not has_any:
        return None, None, None
    del tifs; gc.collect()

    # Temporal diffs
    for year in cb_mod.YEARS:
        for sf, st in [("spring","summer"), ("summer","autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in indices_by_tag and tt in indices_by_tag:
                for n in cb_mod.INDEX_NAMES:
                    all_bands.append((indices_by_tag[tt][n] - indices_by_tag[tf][n]).astype(np.float32))
    for season in cb_mod.SEASONS:
        t0, t1 = f"2020_{season}", f"2021_{season}"
        if t0 in indices_by_tag and t1 in indices_by_tag:
            for n in cb_mod.INDEX_NAMES:
                all_bands.append((indices_by_tag[t1][n] - indices_by_tag[t0][n]).astype(np.float32))
    for year in cb_mod.YEARS:
        ts_s, ts_a = f"{year}_spring", f"{year}_autumn"
        if ts_s in indices_by_tag and ts_a in indices_by_tag:
            for n in ["NDVI", "NDWI", "EVI2", "BSI"]:
                all_bands.append((indices_by_tag[ts_a][n] - indices_by_tag[ts_s][n]).astype(np.float32))
    for year in cb_mod.YEARS:
        for sf, st in [("spring","summer"), ("summer","autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in sar_by_tag and tt in sar_by_tag:
                for b in ["vv","vh"]:
                    all_bands.append((sar_by_tag[tt][b] - sar_by_tag[tf][b]).astype(np.float32))
    for season in cb_mod.SEASONS:
        t0, t1 = f"2020_{season}", f"2021_{season}"
        if t0 in sar_by_tag and t1 in sar_by_tag:
            for b in ["vv","vh"]:
                all_bands.append((sar_by_tag[t1][b] - sar_by_tag[t0][b]).astype(np.float32))

    cube = np.stack(all_bands, axis=-1)
    del all_bands, indices_by_tag, sar_by_tag; gc.collect()
    return cube, labels, cube.shape[-1]


def reshape_to_temporal(X_flat):
    """(N, 72) -> (N, 6, 12) for TempCNN."""
    return X_flat.reshape(X_flat.shape[0], N_SLOTS, N_BANDS)


def main():
    print(f"\n{'='*70}")
    print(f"  3-Way Ensemble: CatBoost V5 + MLP 3x3 + TempCNN 1x1")
    print(f"{'='*70}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load CatBoost ──
    print(f"[{ts()}] Loading CatBoost V5...")
    cb = CatBoostClassifier()
    cb.load_model(CATBOOST_PATH)
    print(f"  CatBoost: {cb.tree_count_} trees")

    # ── Load MLP 3x3 ──
    print(f"[{ts()}] Loading MLP 3x3...")
    state = torch.load(MLP_CKPT, map_location=device, weights_only=False)
    mlp, _ = build_mlp("mlp_3x3", n_classes=N_CLASSES, device=device)
    mlp.load_state_dict(state)
    mlp.eval()
    with open(MLP_SCALER, "rb") as f:
        mlp_scaler = pickle.load(f)
    print(f"  MLP: {mlp.n_params():,} params on {device}")

    # ── Load TempCNN 1x1 ──
    print(f"[{ts()}] Loading TempCNN 1x1...")
    tcnn_state = torch.load(TCNN_CKPT, map_location=device, weights_only=False)
    tcnn, _ = build_tempcnn("tempcnn_1x1", n_classes=N_CLASSES, device=device)
    tcnn.load_state_dict(tcnn_state)
    tcnn.eval()
    with open(TCNN_SCALER, "rb") as f:
        tcnn_scaler = pickle.load(f)
    print(f"  TempCNN: {tcnn.n_params():,} params on {device}")

    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    print(f"  Cities: {[c.name for c in test_cities]}\n")

    # Storage for global metrics
    G_y, G_cb, G_mlp, G_tcnn = [], [], [], []
    G_cb_prob, G_mlp_prob, G_tcnn_prob = [], [], []

    for city in test_cities:
        print(f"[{ts()}] {city.name}...")

        # ── MLP/TempCNN raw cube ──
        raw_cube, Hm, Wm = load_raw_feature_cube(city)
        if raw_cube is None:
            print(f"  SKIP (no raw)"); continue
        raw_labels = load_pixel_labels(city)
        if raw_labels is None:
            del raw_cube; gc.collect()
            print(f"  SKIP (no labels)"); continue

        # ── CatBoost spatial cube ──
        cb_cube, cb_labels, n_cb_f = build_cb_spatial_cube(city)
        if cb_cube is None:
            del raw_cube, raw_labels; gc.collect()
            print(f"  SKIP (no CB features)"); continue

        # Align shapes
        minH = min(Hm, cb_cube.shape[0], raw_labels.shape[0], cb_labels.shape[0])
        minW = min(Wm, cb_cube.shape[1], raw_labels.shape[1], cb_labels.shape[1])

        valid = np.ones((minH, minW), dtype=bool)
        valid[:PAD, :] = False; valid[-PAD:, :] = False
        valid[:, :PAD] = False; valid[:, -PAD:] = False
        valid &= (raw_labels[:minH, :minW] < N_CLASSES)
        valid &= (cb_labels[:minH, :minW] < N_CLASSES)

        raw_nan = np.isnan(raw_cube[:minH, :minW, :]).sum(axis=-1)
        cb_nan = np.isnan(cb_cube[:minH, :minW, :]).sum(axis=-1)
        valid &= (raw_nan < 36)
        valid &= (cb_nan < n_cb_f * 0.5)

        coords = np.argwhere(valid)
        n_valid = len(coords)
        if n_valid == 0:
            del raw_cube, cb_cube, raw_labels, cb_labels; gc.collect()
            print(f"  SKIP (no valid)"); continue

        rng = np.random.RandomState(SEED)
        n_sample = min(MAX_PX, n_valid)
        chosen = rng.choice(n_valid, n_sample, replace=False)
        pix = coords[chosen]
        y = raw_labels[pix[:, 0], pix[:, 1]].astype(np.int32)

        # ── Extract features ──
        np.nan_to_num(raw_cube, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.nan_to_num(cb_cube, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # MLP 3x3 features
        feat_mlp = np.empty((n_sample, N_RAW * 9), dtype=np.float32)
        for i, (r, c) in enumerate(pix):
            feat_mlp[i] = raw_cube[r-1:r+2, c-1:c+2, :].reshape(-1)

        # TempCNN 1x1 features (center pixel)
        feat_tcnn = raw_cube[pix[:, 0], pix[:, 1], :]  # (N, 72)

        # CatBoost features
        feat_cb = cb_cube[pix[:, 0], pix[:, 1], :]
        del raw_cube, cb_cube, raw_labels, cb_labels; gc.collect()

        # ── MLP predictions ──
        X_mlp = mlp_scaler.transform(feat_mlp).astype(np.float32)
        del feat_mlp
        mlp_probs = np.zeros((n_sample, N_CLASSES), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_sample, 8192):
                xb = torch.from_numpy(X_mlp[i:i+8192]).to(device)
                mlp_probs[i:i+8192] = mlp(xb).exp().cpu().numpy()
        del X_mlp

        # ── TempCNN predictions ──
        X_tcnn = tcnn_scaler.transform(feat_tcnn).astype(np.float32)
        X_tcnn_t = reshape_to_temporal(X_tcnn)
        del feat_tcnn, X_tcnn
        tcnn_probs = np.zeros((n_sample, N_CLASSES), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_sample, 8192):
                xb = torch.from_numpy(X_tcnn_t[i:i+8192]).to(device)
                tcnn_probs[i:i+8192] = tcnn.predict(xb).cpu().numpy()
        del X_tcnn_t

        # ── CatBoost predictions ──
        cb_probs = cb.predict_proba(feat_cb)
        del feat_cb; gc.collect()

        # ── Per-city results ──
        cb_acc = (cb_probs.argmax(1) == y).mean()
        mlp_acc = (mlp_probs.argmax(1) == y).mean()
        tcnn_acc = (tcnn_probs.argmax(1) == y).mean()
        print(f"  CB: {cb_acc:.4f}  MLP: {mlp_acc:.4f}  TempCNN: {tcnn_acc:.4f}  ({n_sample:,} px)")

        # Test a few key blends
        for w_cb, w_mlp, w_tcnn in [
            (0.7, 0.3, 0.0),   # original best 2-way
            (0.7, 0.15, 0.15), # equal NN split
            (0.6, 0.2, 0.2),   # more NN weight
            (0.5, 0.25, 0.25), # even more NN
            (0.6, 0.15, 0.25), # TempCNN-heavy NN
            (0.5, 0.2, 0.3),   # TempCNN-dominant NN
        ]:
            ens = (w_cb * cb_probs + w_mlp * mlp_probs + w_tcnn * tcnn_probs).argmax(1)
            acc = (ens == y).mean()
            print(f"    CB={w_cb:.1f} MLP={w_mlp:.2f} T={w_tcnn:.2f}: {acc:.4f}")

        G_y.append(y)
        G_cb.append(cb_probs.argmax(1))
        G_mlp.append(mlp_probs.argmax(1))
        G_tcnn.append(tcnn_probs.argmax(1))
        G_cb_prob.append(cb_probs)
        G_mlp_prob.append(mlp_probs)
        G_tcnn_prob.append(tcnn_probs)
        del cb_probs, mlp_probs, tcnn_probs; gc.collect()

    # ── GLOBAL RESULTS ──
    all_y = np.concatenate(G_y)
    all_cb = np.concatenate(G_cb)
    all_mlp = np.concatenate(G_mlp)
    all_tcnn = np.concatenate(G_tcnn)
    all_cb_p = np.concatenate(G_cb_prob)
    all_mlp_p = np.concatenate(G_mlp_prob)
    all_tcnn_p = np.concatenate(G_tcnn_prob)
    N = len(all_y)

    print(f"\n{'='*70}")
    print(f"  GLOBAL ({N:,} aligned pixels across {len(G_y)} cities)")
    print(f"{'='*70}")
    print(f"\n  Individual models:")
    print(f"    CatBoost V5: {(all_cb==all_y).mean():.4f}")
    print(f"    MLP 3x3:     {(all_mlp==all_y).mean():.4f}")
    print(f"    TempCNN 1x1: {(all_tcnn==all_y).mean():.4f}")

    # ── Sweep weights ──
    print(f"\n  Ensemble sweep:")
    print(f"  {'CB':>4s} {'MLP':>5s} {'TCNN':>5s}  {'Acc':>7s}")
    print(f"  {'-'*28}")

    best_acc, best_w = 0, (0.7, 0.3, 0.0)
    for w_cb in np.arange(0.4, 0.85, 0.05):
        for w_mlp in np.arange(0.0, 1.0 - w_cb + 0.01, 0.05):
            w_tcnn = round(1.0 - w_cb - w_mlp, 2)
            if w_tcnn < -0.01:
                continue
            w_tcnn = max(w_tcnn, 0.0)
            ens = (w_cb * all_cb_p + w_mlp * all_mlp_p + w_tcnn * all_tcnn_p).argmax(1)
            acc = (ens == all_y).mean()
            if acc > best_acc:
                best_acc = acc
                best_w = (w_cb, w_mlp, w_tcnn)

    # Print top results around best
    print(f"\n  Best: CB={best_w[0]:.2f} MLP={best_w[1]:.2f} TCNN={best_w[2]:.2f} -> {best_acc:.4f}")

    # Also show 2-way baselines
    for w in [0.6, 0.7, 0.8]:
        acc2 = ((w * all_cb_p + (1-w) * all_mlp_p).argmax(1) == all_y).mean()
        print(f"  2-way CB={w:.1f}+MLP={1-w:.1f}: {acc2:.4f}")
        acc2t = ((w * all_cb_p + (1-w) * all_tcnn_p).argmax(1) == all_y).mean()
        print(f"  2-way CB={w:.1f}+TCNN={1-w:.1f}: {acc2t:.4f}")

    # Per-class for best 3-way
    ens_best = (best_w[0] * all_cb_p + best_w[1] * all_mlp_p + best_w[2] * all_tcnn_p).argmax(1)
    print(f"\n  Per-class breakdown (best 3-way):")
    print(f"  {'Class':15s}  {'CatBoost':>9s} {'MLP':>9s} {'TempCNN':>9s} {'3-way':>9s}  {'Count':>8s}")
    for ci in range(N_CLASSES):
        m = all_y == ci
        n = m.sum()
        if n == 0: continue
        print(f"  {CLASS_NAMES[ci]:15s}  {(all_cb[m]==ci).mean():9.4f} "
              f"{(all_mlp[m]==ci).mean():9.4f} {(all_tcnn[m]==ci).mean():9.4f} "
              f"{(ens_best[m]==ci).mean():9.4f}  {n:8,}")

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
