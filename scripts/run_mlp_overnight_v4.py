"""
MLP Overnight Sweep V4: exhaustive 5-fold spatial CV.

Two-stage approach:
  Stage A (--stage search):  all configs on fold 0 only -> find top architectures
  Stage B (--stage cv):      top-N finalists x all 5 folds -> robust mean +/- std
  Default (--stage full):    all configs x all folds (overnight)

Config space:
  - Norms:       LayerNorm, BatchNorm, None
  - Activations: gelu, silu, relu, mish, geglu
  - Archs:       plain (L3, L5), residual (L4, L6, L8, L10, L12), ResGeGLU
  - Widths:      128, 256, 512
  - Regularise:  dropout, MixUp, SWA, input_dropout, LR/WD tuning
  - Features:    all 8 groups incl. GLCM/LBP, texture_all, top500
  - Batch size:  2048 (good GPU utilisation without starving optimizer)

Training fixes over V3:
  - Correct ceil steps-per-epoch (not floor)
  - Step-based early stopping with min_steps=1500 (big nets need warmup)
  - SWA + BatchNorm: BN running stats refreshed after weight averaging
  - No torch.cuda.empty_cache() in inner loop (only between folds)

Output:
    reports/phase8/tables/mlp_overnight_v4_search.csv     -- search stage results
    reports/phase8/tables/mlp_overnight_v4.csv            -- cv/full per-fold results
    reports/phase8/tables/mlp_overnight_v4_summary_*.csv  -- mean +/- std per stage

Usage:
    .venv\\Scripts\\python.exe scripts/run_mlp_overnight_v4.py --stage search
    .venv\\Scripts\\python.exe scripts/run_mlp_overnight_v4.py --stage cv --top-n 30
    .venv\\Scripts\\python.exe scripts/run_mlp_overnight_v4.py --stage full
    .venv\\Scripts\\python.exe scripts/run_mlp_overnight_v4.py --folds 0 1 2
"""

import argparse
import json
import math
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
N_FOLDS = 5

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8", "tables")
OUT_CSV = os.path.join(OUT_DIR, "mlp_overnight_v4.csv")
SEARCH_CSV = os.path.join(OUT_DIR, "mlp_overnight_v4_search.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "mlp_overnight_v4_summary.csv")

CONTROL_COLS = {
    "cell_id", "valid_fraction", "low_valid_fraction",
    "reflectance_scale", "full_features_computed",
}


# =====================================================================
# Building blocks  (norm-configurable)
# =====================================================================

def _make_norm(norm_type, dim):
    """Create normalisation layer: layernorm | batchnorm | none."""
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "batchnorm":
        return nn.BatchNorm1d(dim)
    else:
        return nn.Identity()


class PlainBlock(nn.Module):
    """Linear -> Act -> Norm -> Dropout."""

    def __init__(self, in_dim, out_dim, dropout=0.15, activation="gelu",
                 norm_type="layernorm"):
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


class GeGLUBlock(nn.Module):
    """Gated Linear Unit with GELU activation."""

    def __init__(self, in_dim, out_dim, dropout=0.15, activation=None,
                 norm_type="layernorm"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.norm = _make_norm(norm_type, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear(x)
        gate, value = h.chunk(2, dim=-1)
        return self.dropout(self.norm(F.gelu(gate) * value))


class ResMLPBlock(nn.Module):
    """Pre-norm residual: Norm -> Linear -> Act -> Drop -> Linear -> Drop + skip."""

    def __init__(self, d_model, d_hidden=None, dropout=0.15, activation="gelu",
                 norm_type="layernorm"):
        super().__init__()
        d_hidden = d_hidden or d_model * 2
        self.norm = _make_norm(norm_type, d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act_fn = {
            "gelu": lambda x: F.gelu(x, approximate="tanh"),
            "silu": F.silu, "relu": F.relu, "mish": F.mish,
        }[activation]

    def forward(self, x):
        h = self.norm(x)
        h = self.drop1(self.act_fn(self.fc1(h)))
        h = self.drop2(self.fc2(h))
        return x + h


class ResGeGLUBlock(nn.Module):
    """Residual block with GeGLU gating inside."""

    def __init__(self, d_model, d_hidden=None, dropout=0.15, activation=None,
                 norm_type="layernorm"):
        super().__init__()
        d_hidden = d_hidden or d_model * 2
        self.norm = _make_norm(norm_type, d_model)
        self.linear = nn.Linear(d_model, d_hidden * 2)
        self.fc_out = nn.Linear(d_hidden, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.linear(h)
        gate, value = h.chunk(2, dim=-1)
        h = self.drop1(F.gelu(gate) * value)
        h = self.drop2(self.fc_out(h))
        return x + h


# =====================================================================
# Networks
# =====================================================================

class PlainMLP(nn.Module):
    def __init__(self, in_features, n_classes, hidden=256, n_layers=3,
                 dropout=0.15, activation="gelu", input_dropout=0.0,
                 norm_type="layernorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        block_cls = GeGLUBlock if activation == "geglu" else PlainBlock
        layers = [block_cls(in_features, hidden, dropout, activation, norm_type)]
        for _ in range(n_layers - 1):
            layers.append(block_cls(hidden, hidden, dropout, activation, norm_type))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


class ResMLP(nn.Module):
    def __init__(self, in_features, n_classes, d_model=256, n_blocks=6,
                 expansion=2, dropout=0.15, activation="gelu",
                 input_dropout=0.0, norm_type="layernorm"):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.stem = nn.Linear(in_features, d_model)

        if activation == "geglu":
            block_cls = ResGeGLUBlock
        else:
            block_cls = ResMLPBlock
        self.blocks = nn.ModuleList([
            block_cls(d_model, d_model * expansion, dropout, activation, norm_type)
            for _ in range(n_blocks)
        ])
        self.final_norm = _make_norm(norm_type, d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.input_drop(x)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return F.log_softmax(self.head(x), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


# =====================================================================
# Training utilities (fixed: ceil steps, step-based patience, BN refresh)
# =====================================================================

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.001, 0.5 * (1 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def normalize_targets(y):
    y = np.clip(y, 0, None).astype(np.float32)
    row_sums = y.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    y = y / row_sums
    y = y + 1e-7
    y = y / y.sum(axis=1, keepdims=True)
    return y.astype(np.float32)


def soft_cross_entropy(log_pred, target):
    return -(target * log_pred).sum(dim=-1).mean()


def refresh_bn_stats(model, X, batch_size=2048):
    """Re-compute BatchNorm running stats for SWA-averaged weights (properly).
    Resets running stats, uses cumulative moving average, keeps dropout OFF.
    """
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not bn_layers:
        return
    # Eval mode so dropout etc. is OFF, then set only BN layers to train
    model.eval()
    for bn in bn_layers:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average
        bn.train()
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            _ = model(X[i:i + batch_size])
    model.eval()


def _predict_batched(model, X_cpu, device, batch_size=65536):
    """Batched inference: handles large test/val sets without OOM."""
    model.eval()
    n = X_cpu.size(0)
    parts = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X_cpu[i:i + batch_size].to(device, non_blocking=True)
            parts.append(model.predict(xb).cpu())
            del xb
    return torch.cat(parts, dim=0).numpy()




def train_model(net, X_trn, y_trn, X_val, y_val, *,
                lr=1e-3, weight_decay=1e-4, batch_size=2048,
                max_epochs=300, patience_steps=2000, min_steps=1500,
                mixup_alpha=0, use_swa=False, use_cosine=True):
    """Train with AMP, step-based early stopping, and correct cosine schedule."""
    device = X_trn.device
    use_amp = X_trn.is_cuda

    # Fused AdamW (PyTorch 2.x CUDA) — faster kernel
    try:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay,
            fused=use_amp,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n = X_trn.size(0)
    # CEIL not floor — actual number of batches per epoch
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = max_epochs * steps_per_epoch
    scheduler = cosine_warmup_scheduler(optimizer, steps_per_epoch * 3, total_steps) if use_cosine else None

    # Convert step-based patience to epochs (respects batch size)
    patience_epochs = max(math.ceil(patience_steps / steps_per_epoch), 5)
    min_epochs = max(math.ceil(min_steps / steps_per_epoch), 3)

    swa_model = None
    swa_scheduler = None
    swa_start_epoch = int(max_epochs * 0.75)
    if use_swa:
        swa_model = AveragedModel(net)
        swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.5)

    best_val = float("inf")
    best_state = None
    wait = 0
    n_epochs_done = 0

    # Detect BN for batch-size-1 guard
    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())

    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=X_trn.device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_trn[idx], y_trn[idx]
            # Guard: BN crashes on batch size 1
            if has_bn and xb.size(0) < 2:
                continue
            if mixup_alpha > 0:
                xb, yb = mixup_batch(xb, yb, mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logp = net(xb)
                loss = soft_cross_entropy(logp, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler and not (use_swa and epoch >= swa_start_epoch):
                scheduler.step()
            epoch_loss += loss.item()

        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(net)
            swa_scheduler.step()

        net.eval()
        with torch.no_grad():
            val_loss = soft_cross_entropy(net(X_val), y_val).item()

        n_epochs_done = epoch + 1
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            # Only stop after min_epochs, using step-aware patience
            if (epoch + 1 >= min_epochs
                    and wait >= patience_epochs
                    and not use_swa):
                break

    if best_state is not None and not use_swa:
        net.load_state_dict(best_state)

    final_model = net
    if use_swa and swa_model is not None and n_epochs_done > swa_start_epoch:
        final_model = swa_model.module
        # Refresh BatchNorm running stats for SWA-averaged weights
        refresh_bn_stats(final_model, X_trn, batch_size=batch_size)
        final_model.eval()
        with torch.no_grad():
            log_val = final_model(X_val)
            best_val = soft_cross_entropy(log_val, y_val).item()

    return n_epochs_done, best_val, final_model


# =====================================================================
# Feature grouping (reuse V3 logic)
# =====================================================================

def partition_features(full_cols):
    band_prefixes = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
    index_prefixes = {
        "NDVI", "NDWI", "NDBI", "NDMI", "NBR", "SAVI", "BSI",
        "NDRE1", "NDRE2", "EVI", "MSAVI", "CRI1", "CRI2", "MCARI", "MNDWI", "TC",
    }

    bands_idx, indices_idx = [], []
    hog_idx, gabor_idx, lbp_idx, glcm_idx, mp_idx, sv_idx = [], [], [], [], [], []

    for i, col in enumerate(full_cols):
        prefix = col.split("_")[0]
        if col.startswith("delta"):
            continue
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

    texture_all = hog_idx + gabor_idx + lbp_idx + glcm_idx + mp_idx + sv_idx
    bands_indices = bands_idx + indices_idx
    full_no_deltas = [i for i, c in enumerate(full_cols) if not c.startswith("delta")]

    return {
        "bands_indices": bands_indices,
        "bands_indices_texture": bands_indices + texture_all,
        "bands_indices_hog": bands_indices + hog_idx,
        "bands_indices_glcm_lbp": bands_indices + glcm_idx + lbp_idx,
        "texture_all": texture_all,
        "full_no_deltas": full_no_deltas,
        "all_full": list(range(len(full_cols))),
        # top500_full computed at runtime per fold
    }


def top_k_by_variance(X, k=500):
    var = np.var(X, axis=0)
    return np.argsort(var)[-k:].tolist()


# =====================================================================
# All feature sets
# =====================================================================

FEATURE_SETS = [
    "bands_indices",
    "bands_indices_texture",
    "bands_indices_hog",
    "bands_indices_glcm_lbp",
    "texture_all",
    "full_no_deltas",
    "all_full",
    "top500_full",
]

# Feature-set names sorted longest first for correct prefix parsing
FS_BY_LEN = sorted(FEATURE_SETS, key=len, reverse=True)


# =====================================================================
# Config generator — EVERYTHING
# =====================================================================

def _cfg(rid, fs, arch, act, n_layers, d_model, norm_type="layernorm", **kw):
    c = dict(
        id=rid, feature_set=fs, arch=arch, activation=act,
        n_layers=n_layers, d_model=d_model, norm_type=norm_type,
        dropout=kw.get("dropout", 0.15),
        input_dropout=kw.get("input_dropout", 0.0),
        expansion=kw.get("expansion", 2),
        lr=kw.get("lr", 1e-3),
        weight_decay=kw.get("weight_decay", 1e-4),
        mixup_alpha=kw.get("mixup_alpha", 0),
        use_swa=kw.get("use_swa", False),
        use_cosine=kw.get("use_cosine", True),
    )
    return c


def generate_configs():
    configs = []
    rid = 0

    for fs in FEATURE_SETS:
        # ---- Plain MLPs ----
        for norm in ["layernorm", "batchnorm", "none"]:
            for act in ["gelu", "silu", "relu", "mish"]:
                for nl in [3, 5]:
                    for d in [256, 512]:
                        configs.append(_cfg(rid, fs, "plain", act, nl, d, norm))
                        rid += 1

            # GeGLU plain (only works with layernorm/none, not batchnorm — but we still test it)
            for nl in [3, 5]:
                for d in [256, 512]:
                    configs.append(_cfg(rid, fs, "plain", "geglu", nl, d, norm))
                    rid += 1

        # ---- Residual MLPs ----
        for norm in ["layernorm", "batchnorm", "none"]:
            for act in ["gelu", "silu", "relu", "mish", "geglu"]:
                # Core depth sweep
                for nb in [4, 6, 8, 10, 12]:
                    configs.append(_cfg(rid, fs, "residual", act, nb, 256, norm))
                    rid += 1

                # d128 compact
                for nb in [6, 12]:
                    configs.append(_cfg(rid, fs, "residual", act, nb, 128, norm))
                    rid += 1

            # Wide d512 (gelu, silu only — most promising)
            for act in ["gelu", "silu"]:
                for nb in [4, 6]:
                    configs.append(_cfg(rid, fs, "residual", act, nb, 512, norm,
                                        lr=5e-4))
                    rid += 1

        # ---- Regularisation variants (best norm=layernorm only, to keep count sane) ----
        # Dropout variants
        for act in ["gelu", "relu"]:
            for do in [0.10, 0.25]:
                configs.append(_cfg(rid, fs, "residual", act, 8, 256, "layernorm",
                                    dropout=do))
                rid += 1

        # MixUp
        for act in ["gelu", "silu"]:
            for alpha in [0.2, 0.4]:
                configs.append(_cfg(rid, fs, "residual", act, 8, 256, "layernorm",
                                    mixup_alpha=alpha))
                rid += 1

        # SWA
        for act in ["gelu", "silu"]:
            configs.append(_cfg(rid, fs, "residual", act, 6, 256, "layernorm",
                                use_swa=True))
            configs.append(_cfg(rid + 1, fs, "residual", act, 8, 256, "layernorm",
                                use_swa=True))
            rid += 2

        # Expansion=4
        for act in ["gelu", "silu"]:
            for nb in [6, 8]:
                configs.append(_cfg(rid, fs, "residual", act, nb, 256, "layernorm",
                                    expansion=4))
                rid += 1

        # LR variants
        for act in ["gelu", "relu"]:
            configs.append(_cfg(rid, fs, "residual", act, 10, 256, "layernorm",
                                lr=5e-4))
            rid += 1
            configs.append(_cfg(rid, fs, "residual", act, 10, 256, "layernorm",
                                lr=2e-3, weight_decay=5e-4))
            rid += 1

        # Input dropout
        for act in ["gelu"]:
            configs.append(_cfg(rid, fs, "residual", act, 8, 256, "layernorm",
                                input_dropout=0.1))
            rid += 1

    return configs


def make_name(cfg):
    """Human-readable name encoding all varying knobs."""
    name = (f"{cfg['feature_set']}_{cfg['arch']}_{cfg['activation']}"
            f"_L{cfg['n_layers']}_d{cfg['d_model']}")
    # Norm
    nt = cfg.get("norm_type", "layernorm")
    if nt == "batchnorm":
        name += "_bn"
    elif nt == "none":
        name += "_nonorm"
    # else layernorm is default, no suffix
    # Expansion
    if cfg.get("expansion", 2) != 2 and cfg["arch"] == "residual":
        name += f"_exp{cfg['expansion']}"
    if cfg.get("dropout", 0.15) != 0.15:
        name += f"_do{cfg['dropout']}"
    if cfg.get("mixup_alpha", 0) > 0:
        name += f"_mix{cfg['mixup_alpha']}"
    if cfg.get("use_swa", False):
        name += "_swa"
    if cfg.get("input_dropout", 0) > 0:
        name += f"_idrop{cfg['input_dropout']}"
    if cfg["lr"] != 1e-3:
        name += f"_lr{cfg['lr']}"
    if cfg.get("weight_decay", 1e-4) != 1e-4:
        name += f"_wd{cfg['weight_decay']}"
    return name


def build_model(cfg, n_features, device):
    norm_type = cfg.get("norm_type", "layernorm")
    if cfg["arch"] == "plain":
        net = PlainMLP(
            n_features, N_CLASSES,
            hidden=cfg["d_model"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], activation=cfg["activation"],
            input_dropout=cfg.get("input_dropout", 0),
            norm_type=norm_type,
        )
    else:
        net = ResMLP(
            n_features, N_CLASSES,
            d_model=cfg["d_model"], n_blocks=cfg["n_layers"],
            expansion=cfg.get("expansion", 2),
            dropout=cfg["dropout"], activation=cfg["activation"],
            input_dropout=cfg.get("input_dropout", 0),
            norm_type=norm_type,
        )
    return net.to(device)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["search", "cv", "full"], default="full",
                        help="search=fold0 only, cv=top-N x 5 folds, full=everything")
    parser.add_argument("--top-n", type=int, default=30,
                        help="For --stage cv: how many top configs to CV")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Override which folds to run")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Max epochs (default: 120 for search, 300 for cv/full)")
    parser.add_argument("--patience-steps", type=int, default=None,
                        help="Early stopping patience in steps (default: 700 search, 2000 cv/full)")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="Min steps before early stopping (default: 300 search, 1500 cv/full)")
    args = parser.parse_args()

    # Stage-aware epoch budget: quick screen vs full training
    if args.max_epochs is None:
        args.max_epochs = 120 if args.stage == "search" else 300
    if args.patience_steps is None:
        args.patience_steps = 700 if args.stage == "search" else 2000
    if args.min_steps is None:
        args.min_steps = 300 if args.stage == "search" else 1500

    # Stage-based fold selection
    if args.folds is not None:
        folds_to_run = args.folds
    elif args.stage == "search":
        folds_to_run = [0]
    else:
        folds_to_run = list(range(N_FOLDS))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
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

    from pandas.api.types import is_numeric_dtype

    full_feature_cols = [c for c in feat_df.columns
                         if c not in CONTROL_COLS and is_numeric_dtype(feat_df[c])]

    X_all = feat_df[full_feature_cols].values.astype(np.float32)
    y = labels_df[CLASS_NAMES].values.astype(np.float32)
    del feat_df

    print(f"Full data: X={X_all.shape}, y={y.shape}")

    folds_arr = split_df["fold_region_growing"].values
    tiles = split_df["tile_group"].values
    with open(os.path.join(PROCESSED_V2_DIR, "split_spatial_meta.json")) as f:
        meta = json.load(f)

    feat_groups = partition_features(full_feature_cols)

    # ---- Generate configs ----
    configs = generate_configs()

    # ---- Stage-specific filtering ----
    if args.stage == "search":
        # Skip SWA configs in search stage — they run full 300 epochs
        n_before = len(configs)
        configs = [c for c in configs if not c.get("use_swa", False)]
        print(f"Search stage: skipped SWA configs ({n_before} -> {len(configs)})")

    elif args.stage == "cv":
        search_csv = SEARCH_CSV  # separate file written by search stage
        if os.path.exists(search_csv):
            search_df = pd.read_csv(search_csv)
            search_df = search_df.dropna(subset=["r2_uniform"])
            # Only use fold-0 results to pick winners
            if "fold" in search_df.columns:
                search_df = search_df[search_df["fold"] == 0]

            # Parse arch keys and rank by best R2 per arch
            def _arch_key_from_name(name):
                for fs in FS_BY_LEN:
                    prefix = fs + "_"
                    if name.startswith(prefix):
                        return name[len(prefix):]
                return None

            search_df["arch_key"] = search_df["name"].map(_arch_key_from_name)
            search_df = search_df.dropna(subset=["arch_key"])

            best_by_key = (search_df.groupby("arch_key")["r2_uniform"]
                           .max()
                           .sort_values(ascending=False))
            top_keys = set(best_by_key.head(args.top_n).index)

            configs = [c for c in configs
                       if _arch_key_from_name(make_name(c)) in top_keys]
            print(f"Stage CV: top {len(top_keys)} arch keys (fold-0 ranking) "
                  f"-> {len(configs)} configs")
        else:
            print(f"WARNING: {search_csv} not found, running all configs")

    total_configs = len(configs)
    total_runs = total_configs * len(folds_to_run)
    print(f"\nPlan [{args.stage}]: {total_configs} configs x {len(folds_to_run)} folds "
          f"= {total_runs} total runs")

    # Check name uniqueness
    names = [make_name(c) for c in configs]
    unique_n = len(set(names))
    if unique_n != len(names):
        dupes = [n for n in names if names.count(n) > 1]
        print(f"WARNING: {len(names) - unique_n} name collisions! Dupes: {set(dupes)}")
    else:
        print(f"Names OK: {unique_n} unique configs")

    # Determine which CSV to write to
    active_csv = SEARCH_CSV if args.stage == "search" else OUT_CSV

    n_trn_est = 20000  # approximate for display only
    spe = (n_trn_est + args.batch_size - 1) // args.batch_size
    print(f"Batch size: {args.batch_size}, ~{spe} steps/epoch, "
          f"patience ~{math.ceil(args.patience_steps / spe)} epochs, "
          f"min ~{math.ceil(args.min_steps / spe)} epochs")
    est_seconds = total_runs * 10
    print(f"Estimated time: {est_seconds / 3600:.1f} hours")

    # ---- Resume (filtered to current plan + run tag) ----
    os.makedirs(OUT_DIR, exist_ok=True)
    run_tag = (f"st{args.stage}_ep{args.max_epochs}_bs{args.batch_size}"
               f"_pat{args.patience_steps}_min{args.min_steps}_seed{SEED}")
    print(f"Run tag: {run_tag}")

    plan_names = {make_name(c) for c in configs}
    plan_keys = {(n, f) for n in plan_names for f in folds_to_run}

    results = []
    completed = set()
    if os.path.exists(active_csv):
        df_old = pd.read_csv(active_csv)
        df_old["fold"] = df_old["fold"].astype(int)
        old_records = df_old.to_dict("records")
        # Keep only rows that belong to this plan AND same run tag
        results = [r for r in old_records
                   if (r.get("name"), int(r.get("fold", -1))) in plan_keys
                   and r.get("run_tag") == run_tag]
        completed = {(r["name"], int(r["fold"])) for r in results}
        skipped = len(old_records) - len(results)
        remaining = total_runs - len(completed)
        print(f"Resume: {len(completed)} in-plan completed, "
              f"{skipped} out-of-plan/old-tag rows skipped, {remaining} remaining")

    # ---- Cross-validation loop ----
    t0_total = time.time()
    n_done = len(completed)
    n_errors = 0
    elapsed_sum = 0.0  # for running average ETA

    for fold_id in folds_to_run:
        print(f"\n{'='*80}")
        print(f"FOLD {fold_id}/{N_FOLDS-1}")
        print(f"{'='*80}")

        train_idx, test_idx = get_fold_indices(
            tiles, folds_arr, fold_id, meta["tile_cols"], meta["tile_rows"],
            buffer_tiles=1,
        )
        rng = np.random.RandomState(SEED + fold_id)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.15), 100)
        val_idx = train_idx[perm[:n_val]]
        trn_idx = train_idx[perm[n_val:]]
        print(f"  Train: {len(trn_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        feat_groups["top500_full"] = top_k_by_variance(X_all[trn_idx], 500)

        print("  Pre-scaling feature sets -> GPU...")
        scaled_cache = {}
        for feat_set in FEATURE_SETS:
            feat_idx = feat_groups.get(feat_set)
            if feat_idx is None or len(feat_idx) == 0:
                continue
            X = X_all[:, feat_idx]
            scaler = StandardScaler()
            X_trn_s = scaler.fit_transform(X[trn_idx]).astype(np.float32)
            X_val_s = scaler.transform(X[val_idx]).astype(np.float32)
            X_test_s = scaler.transform(X[test_idx]).astype(np.float32)
            scaled_cache[feat_set] = (
                torch.tensor(X_trn_s, dtype=torch.float32).to(device),
                torch.tensor(X_val_s, dtype=torch.float32).to(device),
                torch.tensor(X_test_s, dtype=torch.float32),  # stays on CPU
                len(feat_idx),
            )
            print(f"    {feat_set}: {len(feat_idx)} features (test on CPU)")

        y_trn_t = torch.tensor(normalize_targets(y[trn_idx])).to(device)
        y_val_t = torch.tensor(normalize_targets(y[val_idx])).to(device)

        if device == "cuda":
            mem_mb = torch.cuda.memory_allocated() / 1e6
            print(f"  GPU memory: {mem_mb:.0f} MB")

        for ci, cfg in enumerate(configs):
            name = make_name(cfg)
            if (name, fold_id) in completed:
                continue

            feat_set = cfg["feature_set"]
            if feat_set not in scaled_cache:
                continue

            X_trn_t, X_val_t, X_test_t, n_features = scaled_cache[feat_set]

            try:
                torch.manual_seed(SEED + fold_id)
                net = build_model(cfg, n_features, device)
                n_params = sum(p.numel() for p in net.parameters())

                t0 = time.time()
                epochs, best_val_loss, final_model = train_model(
                    net, X_trn_t, y_trn_t, X_val_t, y_val_t,
                    lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                    batch_size=args.batch_size, max_epochs=args.max_epochs,
                    patience_steps=args.patience_steps,
                    min_steps=args.min_steps,
                    mixup_alpha=cfg.get("mixup_alpha", 0),
                    use_swa=cfg.get("use_swa", False),
                    use_cosine=cfg.get("use_cosine", True),
                )

                # Batched inference (avoids OOM on large test sets)
                y_pred = _predict_batched(final_model, X_test_t, device)
                elapsed = time.time() - t0

                del net, final_model

                summary, _ = evaluate_model(y[test_idx], y_pred, CLASS_NAMES)
                summary.update({
                    "name": name,
                    "fold": fold_id,
                    "stage": args.stage,
                    "feature_set": feat_set,
                    "arch": cfg["arch"],
                    "activation": cfg["activation"],
                    "norm_type": cfg.get("norm_type", "layernorm"),
                    "n_layers": cfg["n_layers"],
                    "d_model": cfg["d_model"],
                    "expansion": cfg.get("expansion", 2),
                    "dropout": cfg["dropout"],
                    "input_dropout": cfg.get("input_dropout", 0),
                    "lr": cfg["lr"],
                    "weight_decay": cfg["weight_decay"],
                    "mixup_alpha": cfg.get("mixup_alpha", 0),
                    "use_swa": cfg.get("use_swa", False),
                    "n_features": n_features,
                    "n_params": n_params,
                    "epochs": epochs,
                    "best_val_loss": round(best_val_loss, 6),
                    "elapsed_s": round(elapsed, 1),
                    "run_tag": run_tag,
                })
                results.append(summary)
                n_done += 1
                elapsed_sum += elapsed

                # Save periodically (every 5 results)
                if n_done % 5 == 0:
                    pd.DataFrame(results).to_csv(active_csv, index=False)

                r2 = summary["r2_uniform"]
                mae = summary["mae_mean_pp"]
                n_new = n_done - len(completed)  # only count new runs
                avg_elapsed = elapsed_sum / max(n_new, 1)
                eta_h = (total_runs - n_done) * avg_elapsed / 3600
                print(f"  [{n_done:5d}/{total_runs}] F{fold_id} {name:60s}  "
                      f"R2={r2:.4f}  MAE={mae:.2f}pp  "
                      f"ep={epochs:3d}  {elapsed:.0f}s  ETA={eta_h:.1f}h")

            except Exception as e:
                n_errors += 1
                print(f"  ERROR [{n_done}/{total_runs}] {name}: {e}")
                # Still record as done so we don't retry
                results.append({"name": name, "fold": fold_id, "stage": args.stage,
                                "run_tag": run_tag, "error": str(e)})
                n_done += 1
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Final save after each fold
        pd.DataFrame(results).to_csv(active_csv, index=False)

        del scaled_cache, y_trn_t, y_val_t
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- Aggregate ----
    total_time = time.time() - t0_total
    df = pd.DataFrame(results)
    # Remove error rows for aggregation
    df_ok = df.dropna(subset=["r2_uniform"]) if "r2_uniform" in df.columns else df

    print(f"\n{'='*100}")
    print(f"OVERNIGHT SWEEP COMPLETE: {len(df_ok)} results, {n_errors} errors, "
          f"{total_time/3600:.1f} hours")
    print(f"{'='*100}")

    if len(df_ok) > 0 and "fold" in df_ok.columns:
        agg = df_ok.groupby("name").agg(
            r2_mean=("r2_uniform", "mean"),
            r2_std=("r2_uniform", "std"),
            mae_mean=("mae_mean_pp", "mean"),
            mae_std=("mae_mean_pp", "std"),
            ait_mean=("aitchison_mean", "mean"),
            ait_std=("aitchison_mean", "std"),
            n_folds=("fold", "count"),
            epochs_mean=("epochs", "mean"),
            elapsed_mean=("elapsed_s", "mean"),
            n_features=("n_features", "first"),
            n_params=("n_params", "first"),
            feature_set=("feature_set", "first"),
            arch=("arch", "first"),
            activation=("activation", "first"),
            norm_type=("norm_type", "first"),
        ).sort_values("r2_mean", ascending=False).reset_index()

        summary_csv = SUMMARY_CSV.replace(".csv", f"_{args.stage}.csv")
        agg.to_csv(summary_csv, index=False)
        print(f"\nSummary: {summary_csv}")

        print("\nTOP 20 by mean R2:")
        for _, r in agg.head(20).iterrows():
            print(f"  {r['name']:60s}  "
                  f"R2={r['r2_mean']:.4f}+/-{r['r2_std']:.4f}  "
                  f"MAE={r['mae_mean']:.2f}  "
                  f"folds={r['n_folds']:.0f}")

        print("\nBEST PER NORM TYPE:")
        for nt in sorted(agg["norm_type"].dropna().unique()):
            best = agg[agg["norm_type"] == nt].iloc[0]
            print(f"  {nt:12s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  "
                  f"{best['name']}")

        print("\nBEST PER FEATURE SET:")
        for fs in sorted(agg["feature_set"].unique()):
            best = agg[agg["feature_set"] == fs].iloc[0]
            print(f"  {fs:30s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  "
                  f"{best['name']}")

        print("\nBEST PER ACTIVATION:")
        for act in sorted(agg["activation"].dropna().unique()):
            best = agg[agg["activation"] == act].iloc[0]
            print(f"  {act:8s}: R2={best['r2_mean']:.4f}+/-{best['r2_std']:.4f}  "
                  f"{best['name']}")


if __name__ == "__main__":
    main()
