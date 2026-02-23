#!/usr/bin/env python3
"""
Run inference with top-2 V8 models on Nuremberg,
save predictions, compute ranking accuracy, and create visualizations.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    TEST_CITIES, CLASS_NAMES, N_CLASSES,
    city_features_dir, city_labels_path,
)
from scripts.run_mlp_overnight_v4 import PlainBlock

SWEEP_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v6_sweep", "20260222_211645")

# ── Model definitions (from sweep_mlp_v6.py) ────────────────────────

class ConstantMLP(nn.Module):
    def __init__(self, in_features, n_classes, n_layers, width,
                 dropout=0.15, activation="silu", input_dropout=0.05,
                 norm_type="batchnorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = [PlainBlock(in_features, width, dropout, activation, norm_type)]
        for _ in range(n_layers - 1):
            layers.append(PlainBlock(width, width, dropout, activation, norm_type))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


class TaperedMLP(nn.Module):
    def __init__(self, in_features, n_classes, widths,
                 dropout=0.15, activation="silu", input_dropout=0.05,
                 norm_type="batchnorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        layers = []
        prev_dim = in_features
        for w in widths:
            layers.append(PlainBlock(prev_dim, w, dropout, activation, norm_type))
            prev_dim = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# ── Config for models to compare ────────────────────────────────────

MODELS = {
    "C_5x2048_full": {
        "type": "C", "shape": (5, 2048), "dropout": 0.30, "file": "C_5x2048_full.pt",
    },
    "T_512_256_128_64_mixup": {
        "type": "T", "shape": [512, 256, 128, 64], "dropout": 0.10, "file": "T_512_256_128_64_mixup.pt",
    },
}

# ── Data loading ────────────────────────────────────────────────────

def load_nuremberg_data(mlp_cols):
    """Load features and labels for Nuremberg."""
    import rasterio
    from scripts.run_multi_city_pipeline_v5 import city_dir, city_anchor_path
    city = [c for c in TEST_CITIES if c.name == "nuremberg"][0]
    # V8 sweep used features_v7/ directory
    feat_dir = os.path.join(city_dir(city), "features_v7")
    feat_path = os.path.join(feat_dir, "features_rust_2020_2021.parquet")
    df = pd.read_parquet(feat_path)
    
    # Select only the MLP columns (in order)
    available = [c for c in mlp_cols if c in df.columns]
    if len(available) != len(mlp_cols):
        print(f"  WARNING: {len(mlp_cols) - len(available)} columns missing")
    X = df[available].values.astype(np.float32)
    
    # Get cell_id
    cell_ids = df["cell_id"].values if "cell_id" in df.columns else np.arange(len(df))
    del df
    
    # Load labels
    label_path = city_labels_path(city, 2021)
    labels = pd.read_parquet(label_path)
    y = labels[CLASS_NAMES].values.astype(np.float32)
    label_cell_ids = labels["cell_id"].values if "cell_id" in labels.columns else np.arange(len(labels))
    
    # Normalize labels
    row_sums = y.sum(axis=1, keepdims=True)
    valid = (row_sums.ravel() > 0)
    if not valid.all():
        X = X[valid]
        y = y[valid]
        cell_ids = cell_ids[valid]
        row_sums = row_sums[valid]
    y = y / np.maximum(row_sums, 1e-8)
    del labels
    
    # Derive row/col from cell_id and anchor grid dimensions
    anchor_path = city_anchor_path(city)
    with rasterio.open(anchor_path) as src:
        n_cols_grid = src.width // 10  # 10m pixels -> 100m cells
    
    meta = {
        "row": (cell_ids // n_cols_grid).astype(np.int32),
        "col": (cell_ids % n_cols_grid).astype(np.int32),
    }
    
    return X, y, meta


def build_model(name, n_features, config):
    """Build model from config."""
    if config["type"] == "C":
        n_layers, width = config["shape"]
        model = ConstantMLP(n_features, N_CLASSES, n_layers, width,
                           dropout=config["dropout"])
    else:
        model = TaperedMLP(n_features, N_CLASSES, config["shape"],
                          dropout=config["dropout"])
    
    # Load weights
    pt_path = os.path.join(SWEEP_DIR, config["file"])
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, X, scaler, batch_size=4096):
    """Run inference and return predictions as numpy array."""
    # Scale features
    X_scaled = (X - scaler.mean_.astype(np.float32)) / scaler.scale_.astype(np.float32)
    
    preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.from_numpy(X_scaled[i:i+batch_size]).to(device)
            p = model.predict(batch).cpu().numpy()
            preds.append(p)
    
    model = model.cpu()
    torch.cuda.empty_cache()
    return np.concatenate(preds, axis=0)


# ── Ranking accuracy ────────────────────────────────────────────────

def ranking_accuracy(pred, truth, k):
    """
    Top-k ranking accuracy: for each cell, check if the top-k predicted
    classes match the top-k ground truth classes (order-insensitive).
    Returns fraction of cells where predicted top-k SET equals truth top-k SET.
    """
    pred_topk = np.argsort(-pred, axis=1)[:, :k]
    truth_topk = np.argsort(-truth, axis=1)[:, :k]
    
    matches = 0
    for i in range(len(pred)):
        if set(pred_topk[i]) == set(truth_topk[i]):
            matches += 1
    return matches / len(pred)


def ordered_ranking_accuracy(pred, truth, k):
    """
    Top-k ordered ranking accuracy: top-k classes must match in exact order.
    """
    pred_topk = np.argsort(-pred, axis=1)[:, :k]
    truth_topk = np.argsort(-truth, axis=1)[:, :k]
    return (pred_topk == truth_topk).all(axis=1).mean()


# ── Visualization helpers ───────────────────────────────────────────

CLASS_COLORS = {
    "tree_cover":   "#228B22",
    "shrubland":    "#DAA520",
    "grassland":    "#90EE90",
    "cropland":     "#FFD700",
    "built_up":     "#DC143C",
    "bare_sparse":  "#D2B48C",
    "water":        "#4169E1",
}

def make_rgb_map(probs, meta, title, filename):
    """Create an RGB land cover map from probability predictions."""
    if "row" not in meta or "col" not in meta:
        print(f"  Cannot create map for {title} — no row/col metadata")
        return
    
    rows, cols = meta["row"], meta["col"]
    n_rows = int(rows.max()) + 1
    n_cols = int(cols.max()) + 1
    
    # Create RGB image
    rgb = np.ones((n_rows, n_cols, 3), dtype=np.float32) * 0.9  # light gray background
    
    color_array = np.array([
        mcolors.to_rgb(CLASS_COLORS[cn]) for cn in CLASS_NAMES
    ], dtype=np.float32)  # (7, 3)
    
    # Weighted blend of class colors by probability
    cell_colors = probs @ color_array  # (N, 3)
    rgb[rows, cols] = cell_colors
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CLASS_COLORS[cn], label=cn) for cn in CLASS_NAMES]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def make_diff_map(pred1, pred2, meta, title, filename):
    """Create a map showing where two models disagree on dominant class."""
    if "row" not in meta or "col" not in meta:
        return
    
    rows, cols = meta["row"], meta["col"]
    n_rows = rows.max() + 1
    n_cols = cols.max() + 1
    
    dom1 = np.argmax(pred1, axis=1)
    dom2 = np.argmax(pred2, axis=1)
    agree = (dom1 == dom2)
    
    rgb = np.ones((n_rows, n_cols, 3), dtype=np.float32) * 0.9
    # Green where they agree, red where they disagree
    rgb[rows[agree], cols[agree]] = [0.7, 0.9, 0.7]
    rgb[rows[~agree], cols[~agree]] = [0.9, 0.2, 0.2]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"{title}\nAgree: {agree.mean()*100:.1f}% ({agree.sum():,}/{len(agree):,})", fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def make_class_diff_map(pred1, pred2, meta, class_idx, class_name, title, filename):
    """Create a map showing per-class probability difference."""
    if "row" not in meta or "col" not in meta:
        return
    
    rows, cols = meta["row"], meta["col"]
    n_rows = rows.max() + 1
    n_cols = cols.max() + 1
    
    diff = pred1[:, class_idx] - pred2[:, class_idx]
    
    img = np.full((n_rows, n_cols), np.nan)
    img[rows, cols] = diff
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    vmax = max(0.1, np.percentile(np.abs(diff), 99))
    im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title}: {class_name} probability diff (Model1 - Model2)", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Prob difference")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    OUT_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_v6_sweep",
                           "20260222_211645", "inference_nuremberg")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load scaler and column list
    print("Loading scaler and feature columns...")
    with open(os.path.join(SWEEP_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(SWEEP_DIR, "mlp_cols.json")) as f:
        mlp_cols = json.load(f)
    
    n_features = len(mlp_cols)
    print(f"  {n_features} features")
    
    # Load Nuremberg data
    print("Loading Nuremberg data...")
    X, y_true, meta = load_nuremberg_data(mlp_cols)
    print(f"  {len(X):,} cells, {X.shape[1]} features")
    
    # Run inference for each model
    predictions = {}
    for name, config in MODELS.items():
        print(f"\nBuilding and running {name}...")
        model = build_model(name, n_features, config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {n_params:,} parameters")
        preds = run_inference(model, X, scaler)
        predictions[name] = preds
        
        # Save predictions
        pred_df = pd.DataFrame(preds, columns=CLASS_NAMES)
        if "row" in meta:
            pred_df["row"] = meta["row"]
            pred_df["col"] = meta["col"]
        pred_df.to_parquet(os.path.join(OUT_DIR, f"preds_{name}.parquet"), index=False)
        print(f"  Saved predictions: preds_{name}.parquet")
        
        del model
        torch.cuda.empty_cache()
    
    # ── Ranking accuracy ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RANKING ACCURACY — Nuremberg")
    print("=" * 60)
    
    for name, preds in predictions.items():
        print(f"\n{name}:")
        for k in [1, 2, 3]:
            set_acc = ranking_accuracy(preds, y_true, k)
            ord_acc = ordered_ranking_accuracy(preds, y_true, k)
            print(f"  Top-{k} set accuracy:     {set_acc*100:.2f}%")
            print(f"  Top-{k} ordered accuracy: {ord_acc*100:.2f}%")
    
    # Direct comparison
    names_list = list(predictions.keys())
    p1, p2 = predictions[names_list[0]], predictions[names_list[1]]
    
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD: {names_list[0]} vs {names_list[1]}")
    print(f"{'='*60}")
    
    # Dominant class agreement
    dom_true = np.argmax(y_true, axis=1)
    dom1 = np.argmax(p1, axis=1)
    dom2 = np.argmax(p2, axis=1)
    
    print(f"\n  Dominant class accuracy:")
    print(f"    {names_list[0]}: {(dom1 == dom_true).mean()*100:.2f}%")
    print(f"    {names_list[1]}: {(dom2 == dom_true).mean()*100:.2f}%")
    print(f"    Both agree with truth: {((dom1 == dom_true) & (dom2 == dom_true)).mean()*100:.2f}%")
    print(f"    Models agree with each other: {(dom1 == dom2).mean()*100:.2f}%")
    
    # Where do they disagree?
    disagree = (dom1 != dom2)
    print(f"\n  Disagreement cells: {disagree.sum():,} ({disagree.mean()*100:.2f}%)")
    if disagree.sum() > 0:
        # Who is right when they disagree?
        m1_right = ((dom1[disagree] == dom_true[disagree])).sum()
        m2_right = ((dom2[disagree] == dom_true[disagree])).sum()
        neither = disagree.sum() - m1_right - m2_right
        print(f"    {names_list[0]} correct: {m1_right} ({m1_right/disagree.sum()*100:.1f}%)")
        print(f"    {names_list[1]} correct: {m2_right} ({m2_right/disagree.sum()*100:.1f}%)")
        print(f"    Neither correct:         {neither} ({neither/disagree.sum()*100:.1f}%)")
        
        # Per-class disagreement
        print(f"\n  Confusion on disagreement cells (ground truth class):")
        for i, cn in enumerate(CLASS_NAMES):
            mask_cls = (dom_true[disagree] == i)
            if mask_cls.sum() > 0:
                print(f"    {cn:15s}: {mask_cls.sum():>4d} cells disagree")
    
    # Per-class R2 and MAE
    from sklearn.metrics import r2_score, mean_absolute_error
    print(f"\n  Per-class R² comparison:")
    print(f"    {'Class':15s} {'Model1':>10s} {'Model2':>10s} {'Delta':>10s}")
    print(f"    {'-'*47}")
    for i, cn in enumerate(CLASS_NAMES):
        r2_1 = r2_score(y_true[:, i], p1[:, i])
        r2_2 = r2_score(y_true[:, i], p2[:, i])
        print(f"    {cn:15s} {r2_1:>10.4f} {r2_2:>10.4f} {r2_1-r2_2:>+10.4f}")
    
    # ── Visualizations ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # 1. Land cover maps
    make_rgb_map(y_true, meta, "Nuremberg — Ground Truth (WorldCover 2021)",
                 os.path.join(OUT_DIR, "map_truth.png"))
    
    for name, preds in predictions.items():
        make_rgb_map(preds, meta, f"Nuremberg — {name}",
                     os.path.join(OUT_DIR, f"map_{name}.png"))
    
    # 2. Agreement/disagreement map
    make_diff_map(p1, p2, meta,
                  f"Dominant Class Agreement: {names_list[0]} vs {names_list[1]}",
                  os.path.join(OUT_DIR, "map_agreement.png"))
    
    # 3. Per-class probability difference maps for interesting classes
    for cls_name in ["bare_sparse", "cropland", "grassland"]:
        idx = CLASS_NAMES.index(cls_name)
        make_class_diff_map(p1, p2, meta, idx, cls_name,
                           f"{names_list[0]} vs {names_list[1]}",
                           os.path.join(OUT_DIR, f"map_diff_{cls_name}.png"))
    
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
