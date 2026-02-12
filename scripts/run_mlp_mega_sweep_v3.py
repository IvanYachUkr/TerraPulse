"""
MLP Mega Sweep V3: Comprehensive tabular MLP sweep + all GPT fixes.

Key improvements vs V2:
FIX:  BatchNorm → LayerNorm everywhere (fair activation comparison)
FIX:  topK_by_variance on TRAIN ONLY (no test leakage)
FIX:  Scale once per feature set (huge speedup)
FIX:  full_no_deltas enforced via col.startswith("delta")
FIX:  SWA returns .module (has .predict()), val loss re-evaluated on SWA weights
FIX:  MixUp stays on GPU via torch.distributions.Beta
NEW:  GPU pre-loading — all data on device, index-batch (zero CPU→GPU transfer)
NEW:  ResMLPBlock — pre-norm residual blocks for stable deep networks (4-12 layers)
NEW:  Cosine annealing with linear warmup
NEW:  MixUp regularization (interpolates samples + targets)
NEW:  SWA (Stochastic Weight Averaging) for final generalization boost
NEW:  Multi-fidelity screening: 15-epoch quick pass, full train for survivors

Architecture families (~65 configs × 6 feature sets ≈ 390 configs):
  - plain:    V2-style blocks (LayerNorm fixed), 2-5 layers, 5 activations
  - residual: pre-norm residual blocks, 4-12 layers, stable gradients
  - residual+regularization: MixUp, SWA, input dropout, dropout sweeps
  - residual+scaling: wide (d512), narrow-deep (d128), expansion=4

Usage:
    .venv\\Scripts\\python.exe scripts/run_mlp_mega_sweep_v3.py
    .venv\\Scripts\\python.exe scripts/run_mlp_mega_sweep_v3.py --skip-screening
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CFG, PROCESSED_V2_DIR, PROJECT_ROOT
from src.models.evaluation import evaluate_model
from src.splitting import get_fold_indices

CLASS_NAMES = CFG["worldcover"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
SPLIT_CFG = CFG["split"]
SEED = SPLIT_CFG["seed"]

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
OUT_CSV = os.path.join(OUT_DIR, "mlp_mega_sweep_v3.csv")

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}


# =====================================================================
# Building blocks
# =====================================================================

class PlainBlock(nn.Module):
    """Standard block: Linear → Act → LayerNorm → Dropout.
    All activations use LayerNorm for fair comparison (V2 had BatchNorm for ReLU).
    """

    def __init__(self, in_dim, out_dim, dropout=0.15, activation="gelu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = {
            "gelu": F.gelu, "silu": F.silu, "relu": F.relu, "mish": F.mish,
        }[activation]

    def forward(self, x):
        return self.dropout(self.norm(self.act_fn(self.linear(x))))


class GeGLUBlock(nn.Module):
    """Gated Linear Unit with GELU activation."""

    def __init__(self, in_dim, out_dim, dropout=0.15, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear(x)
        gate, value = h.chunk(2, dim=-1)
        return self.dropout(self.norm(F.gelu(gate) * value))


class ResMLPBlock(nn.Module):
    """Pre-norm residual block (Gorishniy et al., 2021 style).

    LayerNorm → Linear → Act → Dropout → Linear → Dropout + residual.
    Enables stable training of 6-12 layer networks.
    """

    def __init__(self, d_model, d_hidden=None, dropout=0.15, activation="gelu"):
        super().__init__()
        d_hidden = d_hidden or d_model * 2
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act_fn = {
            "gelu": F.gelu, "silu": F.silu, "relu": F.relu, "mish": F.mish,
        }[activation]

    def forward(self, x):
        h = self.norm(x)
        h = self.act_fn(self.fc1(h))
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


# =====================================================================
# Network architectures
# =====================================================================

class PlainMLP(nn.Module):
    """Standard stacked MLP (like V2, but all LayerNorm)."""

    def __init__(self, input_dim, n_classes, hidden_dim=256,
                 n_layers=3, dropout=0.15, activation="gelu"):
        super().__init__()
        layers = []
        cur = input_dim
        for _ in range(n_layers):
            if activation == "geglu":
                layers.append(GeGLUBlock(cur, hidden_dim, dropout))
            else:
                layers.append(PlainBlock(cur, hidden_dim, dropout, activation))
            cur = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(x)), dim=-1)

    def predict(self, x):
        return F.softmax(self.head(self.backbone(x)), dim=-1)


class ResMLP(nn.Module):
    """Residual MLP: projection → N residual blocks → head.

    The projection maps input to d_model, then residual blocks maintain
    the same dimensionality, allowing much deeper networks (6-12 blocks).
    """

    def __init__(self, input_dim, n_classes, d_model=256,
                 n_blocks=6, expansion=2, dropout=0.15, activation="gelu",
                 input_dropout=0.0):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.proj = nn.Linear(input_dim, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        # Use same activation for projection as for blocks (no GELU hardcode)
        self.act_fn = {
            "gelu": F.gelu, "silu": F.silu, "relu": F.relu, "mish": F.mish,
        }[activation]
        d_hidden = d_model * expansion
        self.blocks = nn.ModuleList([
            ResMLPBlock(d_model, d_hidden, dropout, activation)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def _encode(self, x):
        x = self.input_drop(x)
        h = self.act_fn(self.proj_norm(self.proj(x)))
        for block in self.blocks:
            h = block(h)
        return self.final_norm(h)

    def forward(self, x):
        return F.log_softmax(self.head(self._encode(x)), dim=-1)

    def predict(self, x):
        return F.softmax(self.head(self._encode(x)), dim=-1)


# =====================================================================
# Training utilities
# =====================================================================

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    """Cosine annealing with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / optimizer.defaults["lr"],
                   0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_batch(x, y, alpha=0.2):
    """MixUp: interpolate samples and targets within a batch.
    Uses torch.distributions.Beta to stay fully on GPU.
    """
    if alpha <= 0:
        return x, y
    dist = torch.distributions.Beta(alpha, alpha)
    lam = dist.sample().item()
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix


def normalize_targets(y, n_classes):
    """Ensure non-negative rows summing to 1."""
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, 0.0, None)
    s = y.sum(axis=1, keepdims=True)
    uniform = np.full_like(y, 1.0 / n_classes)
    y = np.where(s > 0, y / s, uniform)
    return y


def soft_cross_entropy(log_probs, targets):
    """Soft cross-entropy: -sum(target * log_pred). No log(target) needed."""
    return -(targets * log_probs).sum(dim=-1).mean()


def train_model(net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                lr=1e-3, weight_decay=1e-4,
                batch_size=1024, max_epochs=300, patience=20,
                mixup_alpha=0.0, use_swa=False, use_cosine=True,
                max_steps=None):
    """Train with soft CE loss, optional MixUp, cosine schedule, SWA.

    All tensor inputs (X_trn_t, y_trn_t, X_val_t, y_val_t) must already
    be on the correct device (GPU or CPU). No DataLoader is used —
    index-batching directly on GPU avoids CPU→GPU transfer overhead.
    """
    n_trn = X_trn_t.size(0)
    device = X_trn_t.device

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=weight_decay,
    )

    steps_per_epoch = (n_trn + batch_size - 1) // batch_size
    total_steps = max_epochs * steps_per_epoch

    if use_cosine:
        warmup_steps = int(0.05 * total_steps)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)
    else:
        scheduler = None

    # SWA setup — start at 55% so it actually triggers before early stopping
    swa_model = None
    swa_start_epoch = int(max_epochs * 0.55)
    effective_patience = patience if not use_swa else max_epochs  # no early stop for SWA
    if use_swa:
        swa_model = AveragedModel(net)
        swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1, anneal_epochs=5)

    best_val = float("inf")
    no_improve = 0
    best_state = None
    n_epochs_done = 0
    global_step = 0

    for epoch in range(max_epochs):
        net.train()
        epoch_loss = 0.0
        n_b = 0

        # Index-batch on device — no DataLoader, no CPU→GPU copies
        perm = torch.randperm(n_trn, device=device)
        for i in range(0, n_trn, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_trn_t[idx]
            yb = y_trn_t[idx]

            if mixup_alpha > 0:
                xb, yb = mixup_batch(xb, yb, alpha=mixup_alpha)

            log_pred = net(xb)
            loss = soft_cross_entropy(log_pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None and not (use_swa and epoch >= swa_start_epoch):
                scheduler.step()

            epoch_loss += loss.item()
            n_b += 1
            global_step += 1

            if max_steps and global_step >= max_steps:
                break

        n_epochs_done = epoch + 1

        # SWA update
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(net)
            swa_scheduler.step()

        # Validation
        net.eval()
        with torch.no_grad():
            log_val = net(X_val_t)
            val_loss = soft_cross_entropy(log_val, y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= effective_patience:
                break

        if max_steps and global_step >= max_steps:
            break

    # Load best checkpoint
    if best_state is not None:
        net.load_state_dict(best_state)

    # Return SWA model if it was active, otherwise best early-stop checkpoint
    final_model = net
    if use_swa and swa_model is not None and n_epochs_done > swa_start_epoch:
        # .module exposes the actual network class (with .predict())
        # AveragedModel wrapper doesn't have custom methods like .predict()
        final_model = swa_model.module
        # Re-evaluate val loss on SWA weights so CSV logs are consistent
        final_model.eval()
        with torch.no_grad():
            log_val = final_model(X_val_t)
            best_val = soft_cross_entropy(log_val, y_val_t).item()

    return n_epochs_done, best_val, final_model


# =====================================================================
# Feature grouping
# =====================================================================

def partition_features(core_cols, full_cols):
    """Split columns into semantic groups. full_no_deltas enforced properly."""
    band_prefixes = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
    index_prefixes = {
        "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
        "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
    }

    bands_idx, indices_idx = [], []
    hog_idx, gabor_idx, lbp_idx, glcm_idx, mp_idx, sv_idx = [], [], [], [], [], []
    other_idx = []

    for i, col in enumerate(full_cols):
        prefix = col.split("_")[0]
        if col.startswith("delta"):
            continue  # skip deltas entirely from group assignment
        elif prefix in band_prefixes:
            bands_idx.append(i)
        elif prefix in index_prefixes:
            indices_idx.append(i)
        elif prefix == "HOG":
            hog_idx.append(i)
        elif prefix == "Gabor":
            gabor_idx.append(i)
        elif prefix == "LBP":
            lbp_idx.append(i)
        elif prefix == "GLCM":
            glcm_idx.append(i)
        elif prefix == "MP":
            mp_idx.append(i)
        elif prefix == "SV":
            sv_idx.append(i)
        else:
            other_idx.append(i)

    texture_all = hog_idx + gabor_idx + lbp_idx + glcm_idx + mp_idx + sv_idx
    bands_indices = bands_idx + indices_idx

    # full_no_deltas: everything EXCEPT delta features
    full_no_deltas = [i for i, c in enumerate(full_cols) if not c.startswith("delta")]

    return {
        "bands_indices": bands_indices,
        "bands_indices_texture": bands_indices + texture_all,
        "bands_indices_hog": bands_indices + hog_idx,
        "bands_indices_glcm_lbp": bands_indices + glcm_idx + lbp_idx,
        "texture_all": texture_all,
        "full_no_deltas": full_no_deltas,
        "all_full": list(range(len(full_cols))),
    }


def top_k_by_variance(X_train_only, k):
    """Select top-K features by variance — computed on TRAIN ONLY."""
    var = np.var(X_train_only, axis=0)
    return np.argsort(var)[-k:][::-1].tolist()


# =====================================================================
# Sweep configuration
# =====================================================================

# Feature sets to test (focused on V2 winners + full)
FEATURE_SETS = [
    "bands_indices",         # V2 winner (798 features)
    "bands_indices_texture", # core + all texture (~1368)
    "bands_indices_hog",     # core + HOG (~1002)
    "full_no_deltas",        # everything minus deltas
    "top500_full",           # top-500 by train variance
    "all_full",              # everything (~3535)
]


def _cfg(run_id, feat_set, arch, act, nl, dm, **kw):
    """Shorthand config builder with sensible defaults."""
    return {
        "run_id": run_id, "feature_set": feat_set,
        "arch": arch, "activation": act,
        "n_layers": nl, "d_model": dm,
        "expansion": kw.get("expansion", 0 if arch == "plain" else 2),
        "dropout": kw.get("dropout", 0.15),
        "input_dropout": kw.get("input_dropout", 0.0),
        "lr": kw.get("lr", 1e-3),
        "weight_decay": kw.get("weight_decay", 1e-4),
        "mixup_alpha": kw.get("mixup_alpha", 0.0),
        "use_swa": kw.get("use_swa", False),
        "use_cosine": kw.get("use_cosine", True),
    }


def generate_configs():
    """
    Comprehensive sweep: ~65 configs per feature set × 6 feature sets ≈ 390.

    Axes explored:
      Plain MLP:  5 activations × 4 architectures + dropout/LR/WD variations
      ResMLP:     3 activations × 5 depths (4-12) + dropout/expansion/width/LR
      Regularization: MixUp (0.2, 0.4), SWA, input dropout, combined
      Scaling:    d_model ∈ {128, 256, 512}, expansion ∈ {2, 4}
    """
    configs = []
    rid = 0

    for fs in FEATURE_SETS:

        # =============================================================
        # A) Plain MLPs — direct V2 comparison (all LayerNorm now)
        # =============================================================

        # Core grid: 4 activations × 4 architectures = 16
        for act in ["gelu", "silu", "relu", "mish"]:
            for nl, hd in [(3, 256), (5, 256), (3, 512), (3, 128)]:
                configs.append(_cfg(rid, fs, "plain", act, nl, hd))
                rid += 1

        # GeGLU (gated, separate block) = 2
        for nl, hd in [(3, 256), (5, 256)]:
            configs.append(_cfg(rid, fs, "plain", "geglu", nl, hd))
            rid += 1

        # Plain variations on relu (V2's best activation) = 3
        configs.append(_cfg(rid, fs, "plain", "relu", 5, 256, dropout=0.10))
        rid += 1
        configs.append(_cfg(rid, fs, "plain", "relu", 3, 256, lr=5e-4))
        rid += 1
        configs.append(_cfg(rid, fs, "plain", "relu", 3, 256, weight_decay=5e-4))
        rid += 1

        # Plain subtotal: 21

        # =============================================================
        # B) ResMLP — depth × activation sweep
        # =============================================================

        # Core grid: 3 activations × 5 depths = 15
        for act in ["gelu", "silu", "relu"]:
            for nb in [4, 6, 8, 10, 12]:
                configs.append(_cfg(rid, fs, "residual", act, nb, 256))
                rid += 1

        # Dropout variations (gelu, key depths) = 3
        for nb in [6, 8]:
            configs.append(_cfg(rid, fs, "residual", "gelu", nb, 256,
                                dropout=0.10))
            rid += 1
        configs.append(_cfg(rid, fs, "residual", "gelu", 8, 256,
                            dropout=0.25))
        rid += 1

        # =============================================================
        # C) ResMLP + MixUp
        # =============================================================

        # 3 activations × 2 depths, alpha=0.2 = 6
        for act in ["gelu", "silu", "relu"]:
            for nb in [6, 8]:
                configs.append(_cfg(rid, fs, "residual", act, nb, 256,
                                    mixup_alpha=0.2))
                rid += 1

        # Higher MixUp alpha = 1
        configs.append(_cfg(rid, fs, "residual", "gelu", 8, 256,
                            mixup_alpha=0.4))
        rid += 1

        # =============================================================
        # D) ResMLP + SWA
        # =============================================================

        # 2 activations × 2 depths = 4
        for act in ["gelu", "silu"]:
            for nb in [6, 8]:
                configs.append(_cfg(rid, fs, "residual", act, nb, 256,
                                    use_swa=True))
                rid += 1

        # =============================================================
        # E) ResMLP + MixUp + SWA ("kitchen sink")
        # =============================================================

        # 2 activations × 1 depth = 2
        for act in ["gelu", "silu"]:
            configs.append(_cfg(rid, fs, "residual", act, 8, 256,
                                mixup_alpha=0.2, use_swa=True))
            rid += 1

        # + input dropout variant = 1
        configs.append(_cfg(rid, fs, "residual", "gelu", 8, 256,
                            mixup_alpha=0.2, use_swa=True, input_dropout=0.05))
        rid += 1

        # =============================================================
        # F) ResMLP scaling variants
        # =============================================================

        # Wide d512: 2 activations × 2 depths = 4
        for act in ["gelu", "silu"]:
            for nb in [4, 6]:
                configs.append(_cfg(rid, fs, "residual", act, nb, 512,
                                    lr=5e-4))
                rid += 1

        # Wide d512 + MixUp = 1
        configs.append(_cfg(rid, fs, "residual", "gelu", 6, 512,
                            lr=5e-4, mixup_alpha=0.2))
        rid += 1

        # Narrow-deep d128 = 2
        for nb in [8, 12]:
            configs.append(_cfg(rid, fs, "residual", "gelu", nb, 128))
            rid += 1

        # Expansion=4 (wider FFN inside residual blocks) = 2
        for nb in [6, 8]:
            configs.append(_cfg(rid, fs, "residual", "gelu", nb, 256,
                                expansion=4))
            rid += 1

        # =============================================================
        # G) ResMLP LR + weight decay variations
        # =============================================================

        # Deep (10, 12) + lower LR = 2
        for nb in [10, 12]:
            configs.append(_cfg(rid, fs, "residual", "gelu", nb, 256,
                                lr=5e-4))
            rid += 1

        # 8-block + higher weight decay = 1
        configs.append(_cfg(rid, fs, "residual", "gelu", 8, 256,
                            weight_decay=5e-4))
        rid += 1

        # Per feature set: 21 + 15 + 3 + 7 + 4 + 3 + 9 + 3 = 65

    print(f"Generated {len(configs)} configurations across {len(FEATURE_SETS)} feature sets")
    return configs


def make_name(cfg):
    """Human-readable config name."""
    name = f"{cfg['feature_set']}_{cfg['arch']}_{cfg['activation']}_L{cfg['n_layers']}_d{cfg['d_model']}"
    if cfg.get("mixup_alpha", 0) > 0:
        name += "_mixup"
    if cfg.get("use_swa", False):
        name += "_swa"
    if cfg.get("input_dropout", 0) > 0:
        name += f"_idrop{cfg['input_dropout']}"
    if cfg["d_model"] != 256:
        pass  # already in name
    if cfg["lr"] != 1e-3:
        name += f"_lr{cfg['lr']}"
    return name


def build_model(cfg, input_dim, device):
    """Build network from config."""
    if cfg["arch"] == "plain":
        net = PlainMLP(
            input_dim=input_dim, n_classes=N_CLASSES,
            hidden_dim=cfg["d_model"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], activation=cfg["activation"],
        )
    elif cfg["arch"] == "residual":
        net = ResMLP(
            input_dim=input_dim, n_classes=N_CLASSES,
            d_model=cfg["d_model"], n_blocks=cfg["n_layers"],
            expansion=cfg["expansion"], dropout=cfg["dropout"],
            activation=cfg["activation"],
            input_dropout=cfg.get("input_dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown arch: {cfg['arch']}")
    return net.to(device)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--skip-screening", action="store_true",
                        help="Skip multi-fidelity screening, run everything fully")
    args = parser.parse_args()

    # ---- Device + seeds ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        # Skip cudnn deterministic flags — we use Linear+LayerNorm, not conv layers
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ---- Load data ----
    print("Loading data...")
    feat_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_full.parquet"))
    labels_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "labels_2021.parquet"))
    split_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "split_spatial.parquet"))

    core_df = pd.read_parquet(os.path.join(PROCESSED_V2_DIR, "features_merged_core.parquet"))
    from pandas.api.types import is_numeric_dtype
    core_cols = [c for c in core_df.columns
                 if c not in CONTROL_COLS and is_numeric_dtype(core_df[c])]
    del core_df

    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    y = labels_df[CLASS_NAMES].values.astype(np.float32)
    del feat_df

    print(f"Full data: X={X_all.shape}, y={y.shape}")

    # ---- Train/val/test split ----
    folds = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        m = json.load(f)

    train_idx, test_idx = get_fold_indices(
        tiles, folds, 0, m["tile_cols"], m["tile_rows"], buffer_tiles=1,
    )
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_idx))
    n_val = max(int(len(train_idx) * 0.15), 100)
    val_idx = train_idx[perm[:n_val]]
    trn_idx = train_idx[perm[n_val:]]
    print(f"Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # ---- Feature groups (topK on TRAIN ONLY) ----
    feat_groups = partition_features(core_cols, full_feature_cols)
    feat_groups["top500_full"] = top_k_by_variance(X_all[trn_idx], 500)

    print("Feature groups: " + ", ".join(f"{k}={len(v)}" for k, v in feat_groups.items()))

    # ---- Generate configs ----
    configs = generate_configs()

    # ---- Resume ----
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    completed = set()
    if os.path.exists(OUT_CSV):
        df_old = pd.read_csv(OUT_CSV)
        results = df_old.to_dict("records")
        completed = set(df_old["name"].values)
        print(f"Found {len(completed)} completed configs, resuming...")

    # ---- Pre-scale and move to GPU (all data lives on device) ----
    print("Pre-scaling feature sets and moving to GPU...")
    scaled_cache = {}  # {feat_set: (X_trn_t, X_val_t, X_test_t, n_features)}
    for feat_set in FEATURE_SETS:
        feat_idx = feat_groups.get(feat_set)
        if feat_idx is None or len(feat_idx) == 0:
            continue
        X = X_all[:, feat_idx]
        scaler = StandardScaler()
        X_trn_s = scaler.fit_transform(X[trn_idx]).astype(np.float32)
        X_val_s = scaler.transform(X[val_idx]).astype(np.float32)
        X_test_s = scaler.transform(X[test_idx]).astype(np.float32)
        # Move to GPU once — stays there for ALL configs using this feature set
        scaled_cache[feat_set] = (
            torch.tensor(X_trn_s, dtype=torch.float32).to(device),
            torch.tensor(X_val_s, dtype=torch.float32).to(device),
            torch.tensor(X_test_s, dtype=torch.float32).to(device),
            len(feat_idx),
        )
        print(f"  {feat_set}: {len(feat_idx)} features scaled → {device}")
    del X_all  # free ~500MB

    # Pre-normalize and move targets to GPU
    y_trn_t = torch.tensor(normalize_targets(y[trn_idx], N_CLASSES)).to(device)
    y_val_t = torch.tensor(normalize_targets(y[val_idx], N_CLASSES)).to(device)

    if device == "cuda":
        mem_mb = torch.cuda.memory_allocated() / 1e6
        print(f"  GPU memory used: {mem_mb:.0f} MB")

    # ---- Multi-fidelity screening (optional) ----
    # Stage 1: 15-epoch quick pass on bands_indices to eliminate bad archs
    screening_survivors = set()
    if not args.skip_screening:
        print("\n=== STAGE 1: Screening (15 epochs on bands_indices) ===")
        screen_feat = "bands_indices"
        if screen_feat in scaled_cache:
            X_trn_t, X_val_t, X_test_t, n_feat = scaled_cache[screen_feat]
            screen_results = []

            # Consistent arch key for screening — includes all training knobs
            def _arch_key(c):
                return (c["arch"], c["activation"], c["n_layers"],
                        c["d_model"], c.get("expansion", 0),
                        c.get("mixup_alpha", 0.0), c.get("use_swa", False),
                        c.get("input_dropout", 0.0))

            seen_archs = set()
            screen_configs = []
            for cfg in configs:
                if cfg["feature_set"] != screen_feat:
                    continue
                ak = _arch_key(cfg)
                if ak not in seen_archs:
                    seen_archs.add(ak)
                    screen_configs.append(cfg)

            for cfg in screen_configs:
                name = make_name(cfg)
                torch.manual_seed(SEED)
                net = build_model(cfg, n_feat, device)
                n_params = sum(p.numel() for p in net.parameters())

                t0 = time.time()
                epochs, val_loss, _ = train_model(
                    net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                    lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                    batch_size=1024, max_epochs=15, patience=15,
                    mixup_alpha=cfg.get("mixup_alpha", 0),
                    use_cosine=False,  # too few epochs for cosine
                )
                elapsed = time.time() - t0

                ak = _arch_key(cfg)
                screen_results.append({
                    "name": name, "val_loss": val_loss,
                    "arch_key": ak,
                    "n_params": n_params, "elapsed": elapsed,
                })
                print(f"  SCREEN {name:55s} val_loss={val_loss:.5f}  {elapsed:.0f}s")

            # Keep top 60% of architectures
            screen_results.sort(key=lambda x: x["val_loss"])
            n_keep = max(int(len(screen_results) * 0.6), 5)
            for r in screen_results[:n_keep]:
                screening_survivors.add(r["arch_key"])

            print(f"\nScreening: {len(screen_results)} → {n_keep} architectures survive")
            for r in screen_results[:n_keep]:
                print(f"  ✓ {r['name']:55s} val={r['val_loss']:.5f}")
            print()

    # ---- STAGE 2: Full training ----
    print("=== STAGE 2: Full training ===")
    t0_total = time.time()

    for cfg in configs:
        if cfg["run_id"] < args.start_from:
            continue

        name = make_name(cfg)
        if name in completed:
            continue

        feat_set = cfg["feature_set"]
        if feat_set not in scaled_cache:
            print(f"[{cfg['run_id']:3d}] SKIP {name} (unknown feature set)")
            continue

        # Check screening filter (same key as screening stage)
        if screening_survivors:
            ak = (cfg["arch"], cfg["activation"], cfg["n_layers"],
                  cfg["d_model"], cfg.get("expansion", 0),
                  cfg.get("mixup_alpha", 0.0), cfg.get("use_swa", False),
                  cfg.get("input_dropout", 0.0))
            if ak not in screening_survivors:
                continue

        X_trn_t, X_val_t, X_test_t, n_features = scaled_cache[feat_set]

        torch.manual_seed(SEED)
        net = build_model(cfg, n_features, device)
        n_params = sum(p.numel() for p in net.parameters())

        t0 = time.time()
        epochs, best_val_loss, final_model = train_model(
            net, X_trn_t, y_trn_t, X_val_t, y_val_t,
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            batch_size=1024, max_epochs=300, patience=20,
            mixup_alpha=cfg.get("mixup_alpha", 0),
            use_swa=cfg.get("use_swa", False),
            use_cosine=cfg.get("use_cosine", True),
        )

        # Evaluate — X_test_t is already on GPU
        final_model.eval()
        with torch.no_grad():
            y_pred = final_model.predict(X_test_t).cpu().numpy()
        elapsed = time.time() - t0

        summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
        summary.update({
            "name": name,
            "run_id": cfg["run_id"],
            "feature_set": feat_set,
            "arch": cfg["arch"],
            "activation": cfg["activation"],
            "n_layers": cfg["n_layers"],
            "d_model": cfg["d_model"],
            "expansion": cfg.get("expansion", 0),
            "dropout": cfg["dropout"],
            "input_dropout": cfg.get("input_dropout", 0),
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "mixup_alpha": cfg.get("mixup_alpha", 0),
            "use_swa": cfg.get("use_swa", False),
            "use_cosine": cfg.get("use_cosine", True),
            "n_features": n_features,
            "n_params": n_params,
            "epochs": epochs,
            "best_val_loss": round(best_val_loss, 6),
            "elapsed_s": round(elapsed, 1),
        })

        results.append(summary)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

        r2 = summary["r2_uniform"]
        mae = summary["mae_mean_pp"]
        ait = summary.get("aitchison_mean", float("nan"))
        print(f"[{cfg['run_id']:3d}] {name:55s}  "
              f"R2={r2:.4f}  MAE={mae:.2f}pp  Ait={ait:.4f}  "
              f"feat={n_features:4d}  ep={epochs:3d}  par={n_params/1e6:.1f}M  {elapsed:.0f}s")

    # ---- Final summary ----
    total_time = time.time() - t0_total
    df = pd.DataFrame(results)

    print(f"\n{'='*100}")
    print(f"SWEEP V3 COMPLETE: {len(df)} configs in {total_time/3600:.1f} hours (device: {device})")
    print(f"{'='*100}")

    if len(df) > 0:
        df_sorted = df.sort_values("r2_uniform", ascending=False)
        print("\nTOP 15 by R2:")
        cols = ["name", "r2_uniform", "mae_mean_pp", "aitchison_mean",
                "n_features", "n_params", "epochs", "elapsed_s"]
        cols = [c for c in cols if c in df_sorted.columns]
        print(df_sorted[cols].head(15).to_string(index=False))

        print("\nBEST PER FEATURE SET:")
        for fs in sorted(df["feature_set"].unique()):
            best = df[df["feature_set"] == fs].sort_values("r2_uniform", ascending=False).iloc[0]
            print(f"  {fs:30s}: R2={best['r2_uniform']:.4f}  "
                  f"arch={best['arch']}_{best['activation']}_L{best['n_layers']}_d{best['d_model']}  "
                  f"feat={best['n_features']}")

        print("\nBEST PER ARCHITECTURE:")
        for arch in sorted(df["arch"].unique()):
            best = df[df["arch"] == arch].sort_values("r2_uniform", ascending=False).iloc[0]
            print(f"  {arch:12s}: R2={best['r2_uniform']:.4f}  name={best['name']}")


if __name__ == "__main__":
    main()
