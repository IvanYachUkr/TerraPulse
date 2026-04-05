"""
MLP architectures for raw-band pixel classification.

Three variants:
  1. RawPixelMLP:    72 → expand → contract → 7   (single pixel)
  2. RawPatch3MLP:   648 → expand → contract → 7  (3×3 neighborhood)
  3. RawPatch5MLP:   1800 → expand → contract → 7 (5×5 neighborhood)

All follow the "expand then contract" pattern so the first layers can
learn spectral index combinations from raw bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainBlock(nn.Module):
    """Linear → BatchNorm → Activation → Dropout."""
    def __init__(self, in_dim, out_dim, dropout=0.15, activation="gelu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self._act_name = activation
        self.act_fn = {
            "gelu": lambda x: F.gelu(x, approximate="tanh"),
            "silu": F.silu,
            "relu": F.relu,
            "mish": F.mish,
        }[activation]

    def forward(self, x):
        return self.dropout(self.norm(self.act_fn(self.linear(x))))


class RawBandMLP(nn.Module):
    """
    Generic expand-contract MLP for raw satellite bands.

    The first layer expands to a wider representation (to learn combinations
    like spectral indices), then progressively contracts to the output.
    """
    def __init__(self, in_features, n_classes=7, widths=(512, 256, 128, 64),
                 dropout=0.20, activation="gelu", input_dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.widths = list(widths)
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()

        layers = []
        prev = in_features
        for w in widths:
            layers.append(PlainBlock(prev, w, dropout, activation))
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x):
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Pre-defined configurations ───────────────────────────────────────────────

MODEL_CONFIGS = {
    # Single pixel: 72 raw features → wide expansion to learn indices
    "mlp_1x1": {
        "in_features": 72,
        "widths": [512, 256, 128, 64],
        "dropout": 0.20,
        "activation": "gelu",
        "input_dropout": 0.0,
        "description": "Single-pixel MLP on 72 raw S1+S2 bands",
    },

    # 3×3 neighborhood: 648 features → learn spatial+spectral patterns
    "mlp_3x3": {
        "in_features": 648,
        "widths": [1024, 512, 256, 64],
        "dropout": 0.25,
        "activation": "gelu",
        "input_dropout": 0.01,
        "description": "3×3 neighborhood MLP on raw bands (648 features)",
    },

    # 5×5 neighborhood: 1800 features → comparable to handcrafted (1764)
    "mlp_5x5": {
        "in_features": 1800,
        "widths": [1024, 512, 256, 64],
        "dropout": 0.30,
        "activation": "gelu",
        "input_dropout": 0.01,
        "description": "5×5 neighborhood MLP on raw bands (1800 features)",
    },

    # Hybrid 3×3: 648 raw + 145 center indices = 793 features
    "mlp_3x3_plus": {
        "in_features": 793,
        "widths": [1024, 512, 256, 64],
        "dropout": 0.25,
        "activation": "gelu",
        "input_dropout": 0.01,
        "description": "3×3 raw + spectral indices (793 features)",
    },

    # Big hybrid 3×3: wider + deeper
    "mlp_3x3_plus_big": {
        "in_features": 793,
        "widths": [2048, 1024, 512, 256, 64],
        "dropout": 0.25,
        "activation": "gelu",
        "input_dropout": 0.02,
        "description": "3x3 raw + indices, WIDE (793->2048->64)",
    },
}


def build_model(config_name, n_classes=7, device="cpu"):
    """Build a model from a pre-defined configuration."""
    cfg = MODEL_CONFIGS[config_name].copy()
    desc = cfg.pop("description", "")
    in_features = cfg.pop("in_features")
    model = RawBandMLP(in_features, n_classes, **cfg).to(device)
    return model, desc
