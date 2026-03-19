"""
MLP model architectures for land cover classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    """Create normalisation layer."""
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(dim)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def _get_activation(name: str):
    """Return activation function by name."""
    return {
        "silu": F.silu,
        "gelu": lambda x: F.gelu(x, approximate="tanh"),
        "relu": F.relu,
        "mish": F.mish,
    }[name]


class PlainBlock(nn.Module):
    """Linear → Activation → Norm → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15,
                 activation: str = "silu", norm_type: str = "batchnorm"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = _make_norm(norm_type, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = _get_activation(activation)

    def forward(self, x):
        return self.dropout(self.norm(self.act_fn(self.linear(x))))


class TaperedMLP(nn.Module):
    """MLP with decreasing layer widths (e.g. 1024 → 512 → 256 → 64 → 7).

    During training, outputs log-softmax for soft cross-entropy loss.
    For ONNX export, use export_forward() which outputs softmax probabilities.
    """

    def __init__(self, in_features: int, n_classes: int, widths: list,
                 dropout: float = 0.15, activation: str = "silu",
                 input_dropout: float = 0.05, norm_type: str = "batchnorm"):
        super().__init__()
        self.input_drop = (nn.Dropout(input_dropout)
                           if input_dropout > 0 else nn.Identity())

        layers = []
        prev_dim = in_features
        for w in widths:
            layers.append(PlainBlock(prev_dim, w, dropout, activation, norm_type))
            prev_dim = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_classes)

    def forward(self, x):
        """Log-softmax output (for training with soft_cross_entropy)."""
        return F.log_softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def export_forward(self, x):
        """Softmax probabilities (for ONNX export / inference)."""
        return torch.softmax(self.head(self.backbone(self.input_drop(x))), dim=-1)

    def predict(self, x):
        """Softmax probabilities (eval mode, no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x).exp()


def make_model(n_features: int, n_classes: int, widths: list,
               **hparams) -> TaperedMLP:
    """Create a TaperedMLP with given hyperparameters.

    Args:
        n_features: number of input features
        n_classes:  number of output classes
        widths:     list of hidden layer widths (e.g. [1024, 512, 256, 64])
        **hparams:  optional overrides for dropout, activation, input_dropout, norm_type
    """
    return TaperedMLP(
        in_features=n_features,
        n_classes=n_classes,
        widths=widths,
        dropout=hparams.get("dropout", 0.15),
        activation=hparams.get("activation", "silu"),
        input_dropout=hparams.get("input_dropout", 0.05),
        norm_type=hparams.get("norm_type", "batchnorm"),
    )
