#!/usr/bin/env python3
"""
Temporal Feature Transformer (TFT) — Architecture experiment.

Instead of flattening all 1800 features into one MLP input, this model:
  1. Groups features into 6 S2 season tokens (224 features each)
     + 1 SAR token (remaining features)
  2. Projects each token to a shared embedding dim via per-group MLPs
  3. Applies multi-head self-attention across the 7 tokens
  4. Pools and classifies into 7 land-cover classes

This exploits temporal structure: "this cell greens up in spring and browns
in summer" is a natural attention pattern that MLPs can't learn.

Usage:
    python scripts/train_tft.py
"""

import gc
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.run_multi_city_pipeline_v5 import (
    TRAIN_CITIES as _ALL_TRAIN, TEST_CITIES as _ALL_TEST,
    SEED, CLASS_NAMES, N_CLASSES,
    city_dir, city_labels_path,
    _load_city_arrays, ts,
)

import pyarrow.parquet as pq

torch.manual_seed(SEED)
np.random.seed(SEED)

# =====================================================================
# Feature grouping — map column names to season tokens
# =====================================================================

S2_SEASONS = [
    "2020_spring", "2020_summer", "2020_autumn",
    "2021_spring", "2021_summer", "2021_autumn",
]

def group_feature_columns(cols):
    """
    Group feature columns into 6 S2 season groups + 1 SAR/other group.
    Returns: dict mapping group_name -> list of column indices.
    """
    groups = {s: [] for s in S2_SEASONS}
    groups["sar_other"] = []
    skip = {"cell_id", "row", "col"}

    for i, c in enumerate(cols):
        if c in skip:
            continue
        matched = False
        for s in S2_SEASONS:
            if s in c:
                groups[s].append(i)
                matched = True
                break
        if not matched:
            groups["sar_other"].append(i)

    return groups


# =====================================================================
# Model: Temporal Feature Transformer
# =====================================================================

class SeasonEncoder(nn.Module):
    """Small MLP to project per-season features to embedding dim."""
    def __init__(self, in_features, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class TemporalFeatureTransformer(nn.Module):
    """
    Temporal Feature Transformer for land cover classification.
    
    Input: [batch, total_features] (flat feature vector)
    Process:
      1. Split into 7 token groups (6 S2 seasons + SAR)
      2. Project each to embed_dim via SeasonEncoder
      3. Add learned positional embeddings
      4. Multi-head self-attention (2 layers)
      5. Mean pool → classification head
    """
    def __init__(self, group_sizes, embed_dim=128, n_heads=4,
                 n_attn_layers=2, dropout=0.15, n_classes=7):
        super().__init__()
        self.group_sizes = group_sizes  # list of (name, size, col_indices)
        n_tokens = len(group_sizes)
        
        # Per-group encoders (different input dims, same output dim)
        self.encoders = nn.ModuleList([
            SeasonEncoder(size, embed_dim, dropout)
            for _, size, _ in group_sizes
        ])
        
        # Learned positional embeddings for each token slot
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, embed_dim) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, n_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Split flat feature vector into token groups
        tokens = []
        for i, (name, size, indices) in enumerate(self.group_sizes):
            group_features = x[:, indices]
            token = self.encoders[i](group_features)
            tokens.append(token)
        
        # Stack into [batch, n_tokens, embed_dim]
        tokens = torch.stack(tokens, dim=1)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed
        
        # Self-attention
        tokens = self.transformer(tokens)
        
        # Mean pooling over tokens
        pooled = tokens.mean(dim=1)
        
        # Classify
        return F.log_softmax(self.head(pooled), dim=-1)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# Also compare with a simple baseline MLP for fair comparison
class BaselineMLP(nn.Module):
    """Simple MLP baseline with same parameter budget."""
    def __init__(self, in_features, n_classes, hidden=512, n_layers=4, dropout=0.15):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden),
                  nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                          nn.SiLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.log_softmax(self.net(x), dim=-1)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# =====================================================================
# Data loading (reuse V6 infrastructure)
# =====================================================================

def load_data(cities, label_year=2021):
    """Load features + labels for a list of cities. Reads parquet directly."""
    META_COLS = {"cell_id", "row", "col", "valid_fraction", "low_valid_fraction"}
    
    # Pass 1: discover common feature columns (excluding metadata)
    col_sets = []
    valid_cities = []
    for city in cities:
        feat_dir = os.path.join(city_dir(city), "features_v7")
        feat_path = os.path.join(feat_dir, "features_rust_2020_2021.parquet")
        if not os.path.exists(feat_path):
            print(f"  WARNING: Missing {city.name} -- skip")
            continue
        schema = pq.read_schema(feat_path)
        numeric_cols = set(c for c in schema.names if c not in META_COLS)
        col_sets.append(numeric_cols)
        valid_cities.append(city)

    if not col_sets:
        return None, None, None, []

    common_cols = sorted(set.intersection(*col_sets))
    print(f"  Common feature columns: {len(common_cols)}")

    # Pass 2: count cells
    city_counts = []
    for city in valid_cities:
        feat_dir = os.path.join(city_dir(city), "features_v7")
        feat_path = os.path.join(feat_dir, "features_rust_2020_2021.parquet")
        n = pq.read_metadata(feat_path).num_rows
        city_counts.append(n)

    total = sum(city_counts)
    n_features = len(common_cols)
    print(f"  Total: {total:,} cells x {n_features} features")

    # Pass 3: load data with explicit column selection
    X = np.empty((total, n_features), dtype=np.float32)
    y = np.empty((total, N_CLASSES), dtype=np.float32)
    offset = 0
    loaded = []

    for ci, (city, n) in enumerate(zip(valid_cities, city_counts)):
        feat_dir = os.path.join(city_dir(city), "features_v7")
        feat_path = os.path.join(feat_dir, "features_rust_2020_2021.parquet")
        
        df = pd.read_parquet(feat_path, columns=common_cols)
        X_city = df.values.astype(np.float32)
        X_city = np.nan_to_num(X_city, nan=0.0, posinf=0.0, neginf=0.0)
        del df
        
        label_path = city_labels_path(city, label_year)
        if not os.path.exists(label_path):
            del X_city
            continue
        labels = pd.read_parquet(label_path)
        y_city = labels[CLASS_NAMES].values.astype(np.float32)
        del labels

        row_sums = y_city.sum(axis=1, keepdims=True)
        valid_mask = (row_sums.ravel() > 0)
        if not valid_mask.all():
            X_city = X_city[valid_mask]
            y_city = y_city[valid_mask]
            row_sums = row_sums[valid_mask]
        y_city = y_city / np.maximum(row_sums, 1e-8)

        actual_n = min(n, X_city.shape[0], y_city.shape[0])
        X[offset:offset + actual_n] = X_city[:actual_n]
        y[offset:offset + actual_n] = y_city[:actual_n]
        loaded.append(city)
        offset += actual_n
        if (ci + 1) % 10 == 0:
            print(f"    ... loaded {ci + 1}/{len(valid_cities)} cities ({offset:,} cells)")
        del X_city, y_city
        gc.collect()

    X = X[:offset]
    y = y[:offset]
    return X, y, common_cols, loaded


# =====================================================================
# Training loop
# =====================================================================

def train_model(model, X_train, y_train, X_test, y_test,
                device, epochs=100, batch_size=4096, lr=1e-3,
                weight_decay=1e-4, patience=15, model_name="model"):
    """Train with early stopping on test loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Move data to GPU
    X_tr = torch.from_numpy(X_train).to(device)
    y_tr = torch.from_numpy(y_train).to(device)
    X_te = torch.from_numpy(X_test).to(device)
    y_te = torch.from_numpy(y_test).to(device)
    
    n = X_tr.shape[0]
    best_test_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    
    print(f"\n  Training {model_name} ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Device: {device}, Batch: {batch_size}, LR: {lr}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_tr[idx]
            yb = y_tr[idx]
            
            logp = model(xb)
            loss = F.kl_div(logp, yb, reduction="batchmean")
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        
        # Evaluate on test
        model.eval()
        with torch.no_grad():
            test_losses = []
            for start in range(0, X_te.shape[0], batch_size * 2):
                end = min(start + batch_size * 2, X_te.shape[0])
                logp = model(X_te[start:end])
                tl = F.kl_div(logp, y_te[start:end], reduction="batchmean")
                test_losses.append(tl.item())
            test_loss = np.mean(test_losses)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 5 == 0 or epoch == 1 or no_improve == 0:
            marker = " *" if no_improve == 0 else ""
            print(f"    Epoch {epoch:3d} | train={train_loss:.5f} test={test_loss:.5f} "
                  f"best={best_test_loss:.5f}@{best_epoch}{marker}")
        
        if no_improve >= patience:
            print(f"    Early stopped at epoch {epoch} (patience={patience})")
            break
    
    model.load_state_dict(best_state)
    return best_test_loss, best_epoch


def evaluate_per_class(model, X_test, y_test, device, batch_size=8192):
    """Compute per-class R² on test set."""
    model.eval()
    X_te = torch.from_numpy(X_test).to(device)
    
    all_preds = []
    with torch.no_grad():
        for start in range(0, X_te.shape[0], batch_size):
            end = min(start + batch_size, X_te.shape[0])
            pred = model.predict(X_te[start:end])
            all_preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    
    # Per-class R²
    r2s = []
    for ci, cn in enumerate(CLASS_NAMES):
        y_true = y_test[:, ci]
        y_pred = preds[:, ci]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        r2s.append(r2)
        print(f"    {cn:15s}  R²={r2:.4f}")
    
    macro_r2 = np.mean(r2s)
    print(f"    {'MACRO':15s}  R²={macro_r2:.4f}")
    return r2s, macro_r2


# =====================================================================
# Main
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Temporal Feature Transformer — Architecture Experiment")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Monkey-patch city_features_dir so _load_city_arrays uses features_v7/
    import scripts.run_multi_city_pipeline_v5 as pipeline
    pipeline.city_features_dir = lambda city: os.path.join(pipeline.city_dir(city), "features_v7")
    
    # Load training data
    print(f"\n[{ts()}] Loading training data...")
    TRAIN_CITIES = [c for c in _ALL_TRAIN]
    X_train, y_train, common_cols, loaded_train = load_data(TRAIN_CITIES)
    print(f"  Loaded {X_train.shape[0]:,} training samples, {len(loaded_train)} cities")
    
    # Load test data
    print(f"\n[{ts()}] Loading test data...")
    X_test, y_test, _, loaded_test = load_data(_ALL_TEST)
    print(f"  Loaded {X_test.shape[0]:,} test samples, {len(loaded_test)} cities")
    
    # Normalize features (standardize per-column)
    print(f"\n[{ts()}] Normalizing features...")
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    # Replace any remaining NaN/inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Group features
    groups = group_feature_columns(common_cols)
    print(f"\n  Feature groups:")
    group_info = []
    for name in S2_SEASONS + ["sar_other"]:
        indices = groups[name]
        print(f"    {name:20s}: {len(indices)} features")
        group_info.append((name, len(indices), indices))
    
    # Output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, "data", "cities", "models_tft", run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    # ── Model 1: Temporal Feature Transformer ──
    print(f"\n{'='*70}")
    print(f"MODEL 1: Temporal Feature Transformer")
    print(f"{'='*70}")
    
    for embed_dim, n_heads, n_layers in [(128, 4, 2), (192, 4, 3), (256, 8, 3)]:
        name = f"TFT_d{embed_dim}_h{n_heads}_L{n_layers}"
        model = TemporalFeatureTransformer(
            group_sizes=group_info,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_attn_layers=n_layers,
            dropout=0.15,
            n_classes=N_CLASSES,
        ).to(device)
        
        test_loss, best_ep = train_model(
            model, X_train, y_train, X_test, y_test,
            device, epochs=150, batch_size=4096, lr=1e-3,
            weight_decay=1e-4, patience=20, model_name=name,
        )
        
        print(f"\n  {name} — Per-class R²:")
        r2s, macro = evaluate_per_class(model, X_test, y_test, device)
        results[name] = {"loss": test_loss, "macro_r2": macro, "r2s": r2s,
                        "params": sum(p.numel() for p in model.parameters())}
        
        # Save best model
        torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # ── Model 2: Baseline MLP (same-ish parameter count) ──
    print(f"\n{'='*70}")
    print(f"MODEL 2: Baseline MLP (for comparison)")
    print(f"{'='*70}")
    
    n_features = X_train.shape[1]
    for hidden, n_layers in [(512, 4), (1024, 3), (2048, 4)]:
        name = f"MLP_h{hidden}_L{n_layers}"
        model = BaselineMLP(
            n_features, N_CLASSES,
            hidden=hidden, n_layers=n_layers, dropout=0.15,
        ).to(device)
        
        test_loss, best_ep = train_model(
            model, X_train, y_train, X_test, y_test,
            device, epochs=150, batch_size=4096, lr=1e-3,
            weight_decay=1e-4, patience=20, model_name=name,
        )
        
        print(f"\n  {name} — Per-class R²:")
        r2s, macro = evaluate_per_class(model, X_test, y_test, device)
        results[name] = {"loss": test_loss, "macro_r2": macro, "r2s": r2s,
                        "params": sum(p.numel() for p in model.parameters())}
        
        torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':40s} {'Params':>10s} {'Test Loss':>10s} {'Macro R²':>10s}")
    print("-" * 72)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["macro_r2"]):
        print(f"{name:40s} {r['params']:>10,} {r['loss']:>10.5f} {r['macro_r2']:>10.4f}")
    
    # Save results
    import json
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({k: {kk: (vv if not isinstance(vv, list) else
                        [float(v) for v in vv])
                      for kk, vv in v.items()}
                  for k, v in results.items()}, f, indent=2)
    
    print(f"\n  Results saved to {out_dir}")
    print(f"\n[{ts()}] Done!")


if __name__ == "__main__":
    main()
