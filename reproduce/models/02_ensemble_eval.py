"""
Ensemble: CatBoost V5 + MLP 3x3 on SAME pixels.

Strategy: For each test city, build the full spatial grid for both models,
find the intersection of valid pixel coordinates, and blend predictions
on those shared pixels.
"""
import sys, os, gc, time, pickle, warnings, math
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
PROJECT = r"c:\Users\vanya\Documents\ML_1_sem\final"
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "reproduce", "models"))
sys.stdout.reconfigure(line_buffering=True)

from catboost import CatBoostClassifier
from reproduce.models.architectures.mlp import build_model
from reproduce.models.shared.config import (
    get_test_cities, city_has_raw_tifs, N_CLASSES, CLASS_NAMES, SEED
)
from reproduce.models.shared.data import load_raw_feature_cube, load_pixel_labels

N_RAW = 72
PAD = 1
MAX_PX = 200_000

CATBOOST_PATH = os.path.join(PROJECT, "data", "cities", "models_pixel_v5",
                             "catboost_pixel_v5_deep_unweighted.cbm")
MLP_CKPT = os.path.join(PROJECT, "reproduce", "models", "checkpoints", "mlp_3x3.pt")
MLP_SCALER = os.path.join(PROJECT, "reproduce", "models", "checkpoints", "mlp_3x3_scaler.pkl")

# ── Import CatBoost feature builder ──
sys.path.insert(0, os.path.join(PROJECT, "reproduce", "pixel"))
sys.path.insert(0, os.path.join(PROJECT, "reproduce", "mlp"))
from importlib import import_module
cb_mod = import_module("02_train_catboost")
step1 = import_module("01_download_data")
CITY_MAP = step1.CITY_MAP


def ts():
    return time.strftime("%H:%M:%S")


def build_cb_spatial_cube(city):
    """Build CatBoost features as (H, W, F) cube instead of flattened."""
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

    cube = np.stack(all_bands, axis=-1)  # (H, W, F)
    del all_bands, indices_by_tag, sar_by_tag; gc.collect()

    return cube, labels, cube.shape[-1]


def main():
    print(f"\n{'='*70}")
    print(f"  Ensemble: CatBoost V5 + MLP 3x3 on SAME pixels")
    print(f"{'='*70}\n")

    # Load CatBoost
    print(f"[{ts()}] Loading CatBoost V5...")
    cb = CatBoostClassifier()
    cb.load_model(CATBOOST_PATH)
    print(f"  CatBoost: {cb.tree_count_} trees")

    # Load MLP
    print(f"[{ts()}] Loading MLP 3x3...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(MLP_CKPT, map_location=device, weights_only=False)
    mlp, _ = build_model("mlp_3x3", n_classes=N_CLASSES, device=device)
    mlp.load_state_dict(state)
    mlp.eval()
    with open(MLP_SCALER, "rb") as f:
        scaler = pickle.load(f)
    print(f"  MLP: {mlp.n_params():,} params on {device}")

    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    print(f"  Cities: {[c.name for c in test_cities]}\n")

    G_y, G_cb, G_mlp = [], [], []
    G_ens = {w: [] for w in [0.3, 0.4, 0.5, 0.6, 0.7]}

    for city in test_cities:
        print(f"[{ts()}] {city.name}...")

        # ── MLP raw cube ──
        mlp_cube, Hm, Wm = load_raw_feature_cube(city)
        if mlp_cube is None:
            print(f"  SKIP (no raw)"); continue
        mlp_labels = load_pixel_labels(city)
        if mlp_labels is None:
            del mlp_cube; gc.collect()
            print(f"  SKIP (no labels)"); continue

        # ── CatBoost spatial cube ──
        cb_cube, cb_labels, n_cb_f = build_cb_spatial_cube(city)
        if cb_cube is None:
            del mlp_cube, mlp_labels; gc.collect()
            print(f"  SKIP (no CB features)"); continue

        # Align shapes
        minH = min(Hm, cb_cube.shape[0], mlp_labels.shape[0], cb_labels.shape[0])
        minW = min(Wm, cb_cube.shape[1], mlp_labels.shape[1], cb_labels.shape[1])

        valid = np.ones((minH, minW), dtype=bool)
        valid[:PAD, :] = False; valid[-PAD:, :] = False
        valid[:, :PAD] = False; valid[:, -PAD:] = False
        valid &= (mlp_labels[:minH, :minW] < N_CLASSES)
        valid &= (cb_labels[:minH, :minW] < N_CLASSES)

        # NaN mask for both cubes
        mlp_nan = np.isnan(mlp_cube[:minH, :minW, :]).sum(axis=-1)
        cb_nan = np.isnan(cb_cube[:minH, :minW, :]).sum(axis=-1)
        valid &= (mlp_nan < 36)  # <50% NaN
        valid &= (cb_nan < n_cb_f * 0.5)

        coords = np.argwhere(valid)
        n_valid = len(coords)
        if n_valid == 0:
            del mlp_cube, cb_cube, mlp_labels, cb_labels; gc.collect()
            print(f"  SKIP (no valid)"); continue

        rng = np.random.RandomState(SEED)
        n_sample = min(MAX_PX, n_valid)
        chosen = rng.choice(n_valid, n_sample, replace=False)
        pix = coords[chosen]

        # Labels
        y = mlp_labels[pix[:, 0], pix[:, 1]].astype(np.int32)

        # ── Extract MLP 3x3 features ──
        np.nan_to_num(mlp_cube, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        feat_mlp = np.empty((n_sample, N_RAW * 9), dtype=np.float32)
        for i, (r, c) in enumerate(pix):
            feat_mlp[i] = mlp_cube[r-1:r+2, c-1:c+2, :].reshape(-1)
        del mlp_cube, mlp_labels; gc.collect()

        # ── Extract CatBoost features ──
        np.nan_to_num(cb_cube, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        feat_cb = cb_cube[pix[:, 0], pix[:, 1], :]  # (N, F)
        del cb_cube, cb_labels; gc.collect()

        # ── MLP predictions ──
        X_mlp = scaler.transform(feat_mlp).astype(np.float32)
        del feat_mlp
        mlp_probs = np.zeros((n_sample, N_CLASSES), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n_sample, 8192):
                xb = torch.from_numpy(X_mlp[i:i+8192]).to(device)
                mlp_probs[i:i+8192] = mlp(xb).exp().cpu().numpy()
        del X_mlp

        # ── CatBoost predictions ──
        cb_probs = cb.predict_proba(feat_cb)
        del feat_cb; gc.collect()

        # ── Results ──
        cb_pred = cb_probs.argmax(1)
        mlp_pred = mlp_probs.argmax(1)
        cb_acc = (cb_pred == y).mean()
        mlp_acc = (mlp_pred == y).mean()
        print(f"  CatBoost: {cb_acc:.4f}  MLP: {mlp_acc:.4f}  ({n_sample:,} px)")

        for w in G_ens:
            ens = (w * cb_probs + (1-w) * mlp_probs).argmax(1)
            ens_acc = (ens == y).mean()
            print(f"    Ens(CB={w:.1f}): {ens_acc:.4f}")
            G_ens[w].append((ens.copy(), y.copy()))

        G_y.append(y); G_cb.append(cb_pred); G_mlp.append(mlp_pred)
        del cb_probs, mlp_probs, cb_pred, mlp_pred, y; gc.collect()

    # ── GLOBAL ──
    all_y = np.concatenate(G_y)
    all_cb = np.concatenate(G_cb)
    all_mlp = np.concatenate(G_mlp)
    N = len(all_y)

    print(f"\n{'='*70}")
    print(f"  GLOBAL ({N:,} aligned pixels across {len(G_y)} cities)")
    print(f"{'='*70}")
    print(f"\n  CatBoost V5: {(all_cb==all_y).mean():.4f}")
    print(f"  MLP 3x3:     {(all_mlp==all_y).mean():.4f}")

    best_w, best_acc = 0, 0
    for w in sorted(G_ens):
        ep = np.concatenate([p for p, _ in G_ens[w]])
        ey = np.concatenate([y for _, y in G_ens[w]])
        acc = (ep == ey).mean()
        print(f"  Ens(CB={w:.1f}): {acc:.4f}")
        if acc > best_acc:
            best_w, best_acc = w, acc

    print(f"\n  ** Best ensemble: CB={best_w:.1f}, acc={best_acc:.4f} **")

    # Per-class for best ensemble
    ep_best = np.concatenate([p for p, _ in G_ens[best_w]])
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':15s}  {'CatBoost':>9s} {'MLP':>9s} {'Ensemble':>9s}  {'Count':>8s}")
    for ci in range(N_CLASSES):
        m = all_y == ci
        n = m.sum()
        if n == 0: continue
        print(f"  {CLASS_NAMES[ci]:15s}  {(all_cb[m]==ci).mean():9.4f} "
              f"{(all_mlp[m]==ci).mean():9.4f} {(ep_best[m]==ci).mean():9.4f}  {n:8,}")

    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
