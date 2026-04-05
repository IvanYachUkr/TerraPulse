"""
TempCNN architectures for pixel-wise land cover classification.

Instead of treating the 72 raw features as a flat vector (like MLPs),
these models reshape them to (6 time steps, 12 bands) and apply 1D
convolutions along the temporal axis. This lets the network learn
seasonal transition patterns (spring→summer→autumn) that distinguish
land cover classes.

Variants:
  tempcnn_1x1:      (6, 12) center pixel temporal conv
  tempcnn_3x3:      (6, 108) 3×3 patch temporal conv (9 pixels × 12 bands per step)
  tempcnn_1x1_plus: (6, 22) temporal + (85,) cross-time branch (raw + CatBoost indices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """Conv1D → BatchNorm → GELU → Dropout, operating on (N, C, T)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.15):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.gelu(self.bn(self.conv(x))))


class TempCNN(nn.Module):
    """
    1D Temporal CNN for satellite time series classification.

    Input shape: (N, T, C) where T=6 time steps, C=bands per step.
    Internally permuted to (N, C, T) for Conv1D.
    """
    def __init__(self, n_channels, n_classes=7, hidden_dims=(64, 128, 256),
                 kernel_size=3, dropout=0.20, pool="avg"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        layers = []
        prev = n_channels
        for hd in hidden_dims:
            layers.append(TemporalConvBlock(prev, hd, kernel_size, dropout))
            prev = hd
        self.conv_stack = nn.Sequential(*layers)

        self.pool_type = pool
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev, n_classes),
        )

    def forward(self, x):
        # x: (N, T, C) → (N, C, T) for Conv1D
        x = x.permute(0, 2, 1)
        x = self.conv_stack(x)  # (N, hidden[-1], T)
        if self.pool_type == "avg":
            x = x.mean(dim=2)  # global average pool → (N, hidden[-1])
        elif self.pool_type == "max":
            x = x.max(dim=2).values
        else:
            x = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)
        return F.log_softmax(self.head(x), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class TempCNNPlus(nn.Module):
    """
    Hybrid TempCNN: temporal branch for per-slot features + FC branch
    for cross-time engineered features (diffs, ranges).

    temporal_input: (N, 6, 22)  — 12 raw + 9 indices + 1 SAR ratio per slot
    crosstime_input: (N, 85)    — seasonal/inter-annual diffs, ranges
    """
    def __init__(self, n_temporal_ch=22, n_crosstime=85, n_classes=7,
                 hidden_dims=(64, 128, 256), kernel_size=3, dropout=0.20):
        super().__init__()
        self.n_temporal_ch = n_temporal_ch
        self.n_crosstime = n_crosstime

        # Temporal branch
        layers = []
        prev = n_temporal_ch
        for hd in hidden_dims:
            layers.append(TemporalConvBlock(prev, hd, kernel_size, dropout))
            prev = hd
        self.temporal_conv = nn.Sequential(*layers)
        self.temporal_out = prev

        # Cross-time branch
        self.crosstime_fc = nn.Sequential(
            nn.Linear(n_crosstime, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Combined head
        self.head = nn.Sequential(
            nn.Linear(prev + 64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_temporal, x_crosstime):
        # Temporal: (N, T, C) → (N, C, T)
        t = x_temporal.permute(0, 2, 1)
        t = self.temporal_conv(t)  # (N, hidden[-1], T)
        t = t.mean(dim=2)  # (N, hidden[-1])

        # Cross-time: (N, 85)
        c = self.crosstime_fc(x_crosstime)  # (N, 64)

        # Combine
        combined = torch.cat([t, c], dim=1)  # (N, hidden[-1] + 64)
        return F.log_softmax(self.head(combined), dim=-1)

    def predict(self, x_temporal, x_crosstime):
        self.eval()
        with torch.no_grad():
            return self.forward(x_temporal, x_crosstime).exp()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Pre-defined configurations ───────────────────────────────────────────────

MODEL_CONFIGS = {
    # Center pixel only: reshape 72 → (6, 12)
    "tempcnn_1x1": {
        "model_class": "TempCNN",
        "n_channels": 12,
        "hidden_dims": [64, 128, 256],
        "kernel_size": 3,
        "dropout": 0.20,
        "description": "1D temporal CNN on center pixel (6 steps x 12 bands)",
    },

    # 3×3 patch: reshape 648 → (6, 108)  [9 pixels × 12 bands per step]
    "tempcnn_3x3": {
        "model_class": "TempCNN",
        "n_channels": 108,
        "hidden_dims": [128, 256, 512],
        "kernel_size": 3,
        "dropout": 0.25,
        "description": "1D temporal CNN on 3x3 patch (6 steps x 108 bands)",
    },

    # Hybrid: temporal (6, 22) + cross-time (85)
    "tempcnn_1x1_plus": {
        "model_class": "TempCNNPlus",
        "n_temporal_ch": 22,
        "n_crosstime": 85,
        "hidden_dims": [64, 128, 256],
        "kernel_size": 3,
        "dropout": 0.20,
        "description": "Hybrid TempCNN: temporal (6x22) + cross-time (85) features",
    },
}


def build_tempcnn(config_name, n_classes=7, device="cpu"):
    """Build a TempCNN model from a pre-defined configuration."""
    cfg = MODEL_CONFIGS[config_name].copy()
    desc = cfg.pop("description", "")
    model_class = cfg.pop("model_class")

    if model_class == "TempCNN":
        model = TempCNN(
            n_channels=cfg["n_channels"],
            n_classes=n_classes,
            hidden_dims=cfg["hidden_dims"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        ).to(device)
    elif model_class == "TempCNNPlus":
        model = TempCNNPlus(
            n_temporal_ch=cfg["n_temporal_ch"],
            n_crosstime=cfg["n_crosstime"],
            n_classes=n_classes,
            hidden_dims=cfg["hidden_dims"],
            kernel_size=cfg["kernel_size"],
            dropout=cfg["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    return model, desc
