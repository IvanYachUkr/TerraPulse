"""
SpectralSpatialNet V2: Scaled-up architecture with modern CNN best practices.

Key changes from V1:
  - Inverted bottleneck residual blocks (MobileNetV2-style) with SE attention
  - Stacked spatial blocks with padding to reuse the 3×3 spatial dims
  - Deeper temporal attention (2 layers, 8 heads)
  - Bigger index and fusion branches
  - ~500K params (6x V1, comparable to TempCNN 3×3)

Input: 3×3 patch with 72 features per pixel = 648 total raw features
       + 145 CatBoost-style indices from center pixel
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    """Channel attention: learn per-channel importance weights."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N, C, H, W)
        w = x.mean(dim=(-2, -1))           # (N, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        return x * w


class InvertedBottleneck(nn.Module):
    """
    MobileNetV2-style inverted bottleneck residual:
      pointwise expand → depthwise 3×3 → SE → pointwise squeeze
    With residual connection when input/output dims match.
    """
    def __init__(self, in_ch, out_ch, expand_ratio=4, use_se=True):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch)

        self.block = nn.Sequential(
            # Pointwise expand
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            # Depthwise spatial
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.se = SqueezeExcite(mid) if use_se else nn.Identity()
        self.squeeze = nn.Sequential(
            # Pointwise squeeze
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        out = self.squeeze(out)
        if self.use_residual:
            out = out + x
        return out


class SpatialEncoder(nn.Module):
    """
    Multi-block spatial encoder processing 3×3 patches per time step.
    Uses padded convolutions to maintain spatial dims through stacked blocks.
    """
    def __init__(self, in_bands=12, dims=(32, 64, 128), expand_ratio=2):
        super().__init__()
        # Stem: pointwise to initial dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_bands, dims[0], 1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        # Stack inverted bottleneck blocks
        blocks = []
        prev_dim = dims[0]
        for dim in dims:
            blocks.append(InvertedBottleneck(prev_dim, dim, expand_ratio=expand_ratio))
            blocks.append(InvertedBottleneck(dim, dim, expand_ratio=expand_ratio))
            prev_dim = dim
        self.blocks = nn.Sequential(*blocks)

        # Global average pool collapses 3×3 → 1×1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = dims[-1]

    def forward(self, x):
        # x: (N*T, in_bands, 3, 3)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)  # (N*T, out_dim)
        return x


class TemporalAttentionBlock(nn.Module):
    """Single transformer encoder block: self-attention + feedforward."""
    def __init__(self, d_model, n_heads=8, ff_mult=4, dropout=0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm residual
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class TemporalEncoder(nn.Module):
    """Stacked transformer encoder with positional encoding and attention pooling."""
    def __init__(self, d_model, n_layers=2, n_heads=8, n_positions=6, dropout=0.15):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Learned attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        # x: (N, T, D)
        x = x + self.pos_enc[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        # Attention-weighted pooling
        scores = (self.pool_query * x).sum(dim=-1, keepdim=True)  # (N, T, 1)
        weights = F.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)  # (N, D)


# ── Main model ───────────────────────────────────────────────────────────────

class SpectralSpatialNetV2(nn.Module):
    """
    V2 architecture with ~500K params:
      Spatial: 3-stage inverted bottleneck with SE (shared across time steps)
      Temporal: 2-layer transformer encoder with attention pooling
      Index: 3-layer MLP on precomputed spectral indices
      Fusion: concat + 2-layer head
    """
    def __init__(self, n_bands=12, n_timesteps=6, n_indices=145,
                 spatial_dims=(32, 64, 128), expand_ratio=4, temporal_dim=128,
                 n_attn_layers=2, n_heads=8,
                 n_classes=7, dropout=0.15):
        super().__init__()
        self.n_bands = n_bands
        self.n_timesteps = n_timesteps

        # ── Spatial branch ──
        self.spatial = SpatialEncoder(n_bands, dims=spatial_dims, expand_ratio=expand_ratio)
        spatial_out = self.spatial.out_dim

        # Project spatial output to temporal dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_out, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
        )

        # ── Temporal branch ──
        self.temporal = TemporalEncoder(
            temporal_dim, n_layers=n_attn_layers, n_heads=n_heads,
            n_positions=n_timesteps, dropout=dropout,
        )

        # ── Index branch ──
        idx_hidden = temporal_dim * 2
        self.index_branch = nn.Sequential(
            nn.Linear(n_indices, idx_hidden),
            nn.BatchNorm1d(idx_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(idx_hidden, temporal_dim),
            nn.BatchNorm1d(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Fusion head ──
        fusion_dim = temporal_dim * 2
        fusion_hidden = temporal_dim * 2
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, n_classes),
        )

    def forward(self, patches, indices):
        """
        patches: (N, 648) — flattened 3×3 patch of raw bands
        indices: (N, 145) — precomputed spectral indices for center pixel
        """
        N = patches.shape[0]

        # Reshape (N, 648) → (N, 9, 6, 12) → (N, 6, 12, 3, 3)
        x = patches.reshape(N, 9, self.n_timesteps, self.n_bands)
        x = x.permute(0, 2, 3, 1)           # (N, 6, 12, 9)
        x = x.reshape(N, self.n_timesteps, self.n_bands, 3, 3)

        # Spatial: merge N×T, process all time steps in one batch
        x = x.reshape(N * self.n_timesteps, self.n_bands, 3, 3)
        x = self.spatial(x)                  # (N*T, spatial_out)
        x = x.reshape(N, self.n_timesteps, -1)  # (N, T, spatial_out)

        # Project and apply temporal attention
        x = self.spatial_proj(x)             # (N, T, temporal_dim)
        spatial_vec = self.temporal(x)       # (N, temporal_dim)

        # Index branch
        idx_vec = self.index_branch(indices)  # (N, temporal_dim)

        # Fusion
        combined = torch.cat([spatial_vec, idx_vec], dim=1)
        return F.log_softmax(self.head(combined), dim=-1)

    def predict(self, patches, indices):
        self.eval()
        with torch.no_grad():
            return self.forward(patches, indices).exp()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
