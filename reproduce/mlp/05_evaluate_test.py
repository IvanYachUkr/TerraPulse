#!/usr/bin/env python3
"""
Step 5: Evaluate top models on 6 held-out test cities.

Test cities (NEVER used in training or validation):
  nuremberg, ankara_test, sofia_test, riga_test, edinburgh_test, palermo_test

Expected result: Model #7 ranks #1 at 5% and 10% thresholds.

Usage:
    python 05_evaluate_test.py
"""

import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CITIES_DIR = os.path.join(PROJECT_ROOT, "data", "cities")
V10_DIR = os.path.join(CITIES_DIR, "models_v10_bohb")

SEED = 42
N_CLASSES = 7
CLASS_NAMES = ["tree_cover", "shrubland", "grassland", "cropland",
               "built_up", "bare_sparse", "water"]
TEST_CITIES = ["nuremberg", "ankara_test", "sofia_test",
               "riga_test", "edinburgh_test", "palermo_test"]
FIXED_THRESHOLDS = [0.0, 0.05, 0.10]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def _make_norm(norm_type, dim):
    if norm_type == "batchnorm": return nn.BatchNorm1d(dim)
    elif norm_type == "layernorm": return nn.LayerNorm(dim)
    return nn.Identity()


class PlainBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.15, activation="gelu",
                 norm_type="batchnorm"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = _make_norm(norm_type, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = {
            "gelu": lambda x: F.gelu(x, approximate="tanh"),
            "silu": F.silu, "relu": F.relu, "mish": F.mish,
        }[activation]

    def forward(self, x):
        return self.dropout(self.norm(self.act_fn(self.linear(x))))


class TaperedMLP(nn.Module):
    def __init__(self, in_f, n_c, widths, do=0.15, act="gelu", ido=0.05):
        super().__init__()
        self.input_drop = nn.Dropout(ido) if ido > 0 else nn.Identity()
        layers, prev = [], in_f
        for w in widths:
            layers.append(PlainBlock(prev, w, do, act, "batchnorm"))
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_c)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------
def compute_r2(preds, y, threshold):
    r2s = []
    for ci in range(N_CLASSES):
        mk = y[:, ci] >= threshold if threshold > 0 else np.ones(len(y), dtype=bool)
        if mk.sum() < 50: continue
        yt = y[mk, ci]
        if np.var(yt) < 1e-8: continue
        r2s.append(r2_score(yt, preds[mk, ci]))
    return np.mean(r2s) if r2s else 0


def compute_topk_accuracy(preds, y, k, threshold=0.01):
    n_above = (y >= threshold).sum(axis=1)
    mask = n_above >= k
    if mask.sum() < 50: return float("nan")
    true_topk = np.argsort(-y[mask], axis=1)[:, :k]
    pred_topk = np.argsort(-preds[mask], axis=1)[:, :k]
    return np.mean([set(t) == set(p) for t, p in zip(true_topk, pred_topk)])



# ===== MODEL #7 — hardcoded config (from BOHB trial 77) =====
MODEL7_CONFIG = {
    "trial": 77,
    "arch": "T_1024_512_256_64",
    "widths": [1024, 512, 256, 64],
    "activation": "gelu",
    "dropout": 0.3255,
    "input_dropout": 0.0031,
    "label_threshold": 0.021,
}


def load_models():
    """Load Model #7 (hardcoded). Optionally also loads extra BOHB models
    from trial_log.jsonl if it exists."""
    scaler_path = os.path.join(V10_DIR, "scaler.pkl")
    cols_path = os.path.join(V10_DIR, "mlp_cols.json")
    model7_path = os.path.join(V10_DIR,
        f"trial_{MODEL7_CONFIG['trial']}_{MODEL7_CONFIG['arch']}.pt")

    if not os.path.exists(model7_path):
        print(f"ERROR: Model #7 not found: {model7_path}")
        print("Run 04_train_model7.py (or 03_train_bohb_sweep.py) first.")
        sys.exit(1)

    # Always include Model #7
    models = [{
        "rank": 1,
        "name": f"# 1 {MODEL7_CONFIG['arch']} {MODEL7_CONFIG['activation']} *",
        "path": model7_path,
        "widths": MODEL7_CONFIG["widths"],
        "act": MODEL7_CONFIG["activation"],
        "do": MODEL7_CONFIG["dropout"],
        "ido": MODEL7_CONFIG["input_dropout"],
        "thresh": max(MODEL7_CONFIG["label_threshold"], 0.01),
        "scaler": scaler_path,
        "cols": cols_path,
        "val_combined": None,
        "trial": MODEL7_CONFIG["trial"],
    }]

    # Optionally load more models from sweep log
    log_path = os.path.join(V10_DIR, "trial_log.jsonl")
    if os.path.exists(log_path):
        trials = [json.loads(l) for l in open(log_path) if l.strip()]
        if len(trials) > 20:
            trials = trials[20:]
        ranked = sorted(trials, key=lambda t: t["combined"], reverse=True)
        seen_trials = {MODEL7_CONFIG["trial"]}
        for t in ranked:
            if t["trial"] in seen_trials:
                continue
            seen_trials.add(t["trial"])
            pt = os.path.join(V10_DIR, f"trial_{t['trial']}_{t['arch']}.pt")
            if not os.path.exists(pt):
                continue
            rank = len(models) + 1
            models.append({
                "rank": rank,
                "name": f"#{rank:>2d} {t['arch']} {t['config']['activation']}",
                "path": pt,
                "widths": t["widths"],
                "act": t["config"]["activation"],
                "do": t["config"]["dropout"],
                "ido": t["config"]["input_dropout"],
                "thresh": max(t["config"]["label_threshold"], 0.01),
                "scaler": scaler_path,
                "cols": cols_path,
                "val_combined": t["combined"],
                "trial": t["trial"],
            })
            if len(models) >= 10:
                break
        print(f"  (Also loaded {len(models)-1} extra models from sweep log)")

    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    sep = "=" * 90

    print(f"\n{sep}")
    print("  V10 BOHB MODEL EVALUATION — 6 Held-Out Test Cities")
    print(f"{sep}")

    models = load_models()
    print(f"\n  Models: {len(models)}")
    for m in models:
        dep = " <-- DEPLOYED" if m["trial"] == MODEL7_CONFIG["trial"] else ""
        print(f"    {m['name']:30s} trial={m['trial']}{dep}")

    # Precompute predictions
    all_preds, all_labels, all_top1 = {}, {}, {}

    for city in TEST_CITIES:
        feat_path = os.path.join(CITIES_DIR, city, "features_v7",
                                  "features_rust_2020_2021.parquet")
        label_path = os.path.join(CITIES_DIR, city, "labels_2021.parquet")
        if not os.path.exists(feat_path) or not os.path.exists(label_path):
            print(f"    {city}: SKIP (missing data)")
            continue

        df_lab = pd.read_parquet(label_path)
        y = df_lab[CLASS_NAMES].values.astype(np.float32)
        rs = y.sum(axis=1)
        valid = rs > 0
        y = y[valid]
        y = y / y.sum(axis=1, keepdims=True)
        all_labels[city] = y

        for m in models:
            with open(m["cols"]) as f:
                mlp_cols = json.load(f)
            with open(m["scaler"], "rb") as f:
                scaler = pickle.load(f)

            df_feat = pd.read_parquet(feat_path)
            for col in mlp_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            X = scaler.transform(df_feat[mlp_cols].values.astype(np.float32)[valid])
            del df_feat

            n_f = len(mlp_cols)
            net = TaperedMLP(n_f, N_CLASSES, m["widths"], m["do"], m["act"], m["ido"]).to(device)
            net.load_state_dict(torch.load(m["path"], map_location=device, weights_only=True))
            net.eval()

            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            preds = net.predict(X_t).cpu().numpy()
            del X_t, net, X
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            all_preds[(m["name"], city)] = preds
            all_top1[(m["name"], city)] = float((y.argmax(1) == preds.argmax(1)).mean())

        print(f"    {city} ({valid.sum():,} px) — done")

    model_names = [m["name"] for m in models]

    # Top-1 per city
    print(f"\n{sep}")
    print(f"  TOP-1 ACCURACY PER CITY")
    print(f"{sep}")
    W = 12
    header = f"{'Model':30s}" + "".join(f"{c:>{W}s}" for c in TEST_CITIES) + f"{'MEAN':>{W}s}"
    print(header)
    print("-" * len(header))
    for mn in model_names:
        row = f"{mn:30s}"
        vals = []
        for city in TEST_CITIES:
            if city not in all_labels:
                row += f"{'--':>{W}s}"; continue
            v = all_top1[(mn, city)]
            row += f"{v:>{W}.4f}"; vals.append(v)
        row += f"{np.mean(vals):>{W}.4f}"
        print(row)

    # Rankings at each threshold
    for thresh in FIXED_THRESHOLDS:
        tl = f"{int(thresh*100)}%" if thresh > 0 else "0%"
        print(f"\n{sep}")
        print(f"  RANKING @ THRESHOLD = {tl}")
        print(f"{sep}")
        print(f"{'Rank':<5s} {'Model':30s} {'Combined':>10s} {'Top-1':>10s} {'R2':>10s}")
        print("-" * 65)
        ranks = []
        for mn in model_names:
            t1_vals, r2_vals = [], []
            for city in TEST_CITIES:
                if city not in all_labels: continue
                y = all_labels[city]; p = all_preds[(mn, city)]
                t1_vals.append(all_top1[(mn, city)])
                r2_vals.append(compute_r2(p, y, thresh))
            mean_t1 = np.mean(t1_vals); mean_r2 = np.mean(r2_vals)
            ranks.append((mn, 0.5*mean_t1 + 0.5*max(0, mean_r2), mean_t1, mean_r2))
        ranks.sort(key=lambda x: x[1], reverse=True)
        for i, (mn, c, t1, r2) in enumerate(ranks, 1):
            print(f"{i:<5d} {mn:30s} {c:>10.4f} {t1:>10.4f} {r2:>10.4f}")

    print(f"\n  Evaluation complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
