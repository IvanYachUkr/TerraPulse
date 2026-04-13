import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(-2, -1))
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class InvertedBottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 4,
                 use_se: bool = True, dropout: float = 0.0):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.se = SqueezeExcite(mid) if use_se else nn.Identity()
        self.squeeze = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        out = self.squeeze(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out


class CenterContextReadout(nn.Module):
    """
    Explicitly separates the center pixel from its 8-neighbour context.

    For center-pixel classification on a 3x3 patch, global average pooling is
    too symmetric. This readout keeps the center feature, neighbour mean, and
    center-minus-context contrast.
    """
    def __init__(self, channels: int, out_dim: int, dropout: float = 0.05):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(channels * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        # fmap: (N, C, 3, 3)
        center = fmap[:, :, 1, 1]
        total = fmap.sum(dim=(-2, -1))
        neigh_mean = (total - center) / 8.0
        contrast = center - neigh_mean
        feat = torch.cat([center, neigh_mean, contrast], dim=1)
        return self.proj(feat)


class SpatialEncoderV5(nn.Module):
    def __init__(self, in_bands: int = 12, dims=(32, 64, 128),
                 expand_ratio: int = 4, dropout: float = 0.05):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_bands, dims[0], 1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        blocks = []
        prev_dim = dims[0]
        for dim in dims:
            blocks.append(InvertedBottleneck(
                prev_dim, dim, expand_ratio=expand_ratio,
                use_se=True, dropout=dropout if dim == dims[-1] else 0.0,
            ))
            blocks.append(InvertedBottleneck(
                dim, dim, expand_ratio=expand_ratio,
                use_se=True, dropout=dropout if dim == dims[-1] else 0.0,
            ))
            prev_dim = dim
        self.blocks = nn.Sequential(*blocks)
        self.readout = CenterContextReadout(dims[-1], dims[-1], dropout=dropout)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.readout(x)
        return x


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class TemporalEncoderV5(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2,
                 n_heads: int = 4, n_positions: int = 6,
                 dropout: float = 0.15):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc = nn.Parameter(torch.randn(1, n_positions + 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D)
        n, t, d = x.shape
        cls = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_enc[:, :t + 1, :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x[:, 0]


class MLPBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralSpatialNetV5(nn.Module):
    """
    Main design changes versus V2/V4:
      - explicit center-vs-context spatial readout instead of pure GAP
      - branch dropout to stop the handcrafted index branch from dominating
      - gated fusion
      - auxiliary branch heads so the spatial pathway cannot be ignored
      - returns raw logits (loss decides how to normalize)
    """
    def __init__(self,
                 n_bands: int = 12,
                 n_timesteps: int = 6,
                 n_indices: int = 145,
                 spatial_dims=(32, 64, 128),
                 expand_ratio: int = 4,
                 temporal_dim: int = 128,
                 n_attn_layers: int = 2,
                 n_heads: int = 4,
                 n_classes: int = 7,
                 dropout: float = 0.15,
                 spatial_branch_drop: float = 0.10,
                 index_branch_drop: float = 0.25):
        super().__init__()
        self.n_bands = n_bands
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes
        self.spatial_branch_drop = spatial_branch_drop
        self.index_branch_drop = index_branch_drop

        self.spatial = SpatialEncoderV5(
            in_bands=n_bands,
            dims=spatial_dims,
            expand_ratio=expand_ratio,
            dropout=min(dropout, 0.10),
        )
        spatial_out = self.spatial.out_dim

        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_out, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.temporal = TemporalEncoderV5(
            d_model=temporal_dim,
            n_layers=n_attn_layers,
            n_heads=n_heads,
            n_positions=n_timesteps,
            dropout=dropout,
        )

        idx_hidden = temporal_dim * 2
        self.index_branch = MLPBranch(
            in_dim=n_indices,
            hidden_dim=idx_hidden,
            out_dim=temporal_dim,
            dropout=dropout,
        )

        self.spatial_norm = nn.LayerNorm(temporal_dim)
        self.index_norm = nn.LayerNorm(temporal_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, temporal_dim),
            nn.Sigmoid(),
        )

        fusion_dim = temporal_dim * 2
        hidden = temporal_dim * 2
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, temporal_dim),
            nn.BatchNorm1d(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, n_classes),
        )

        self.spatial_aux = nn.Linear(temporal_dim, n_classes)
        self.index_aux = nn.Linear(temporal_dim, n_classes)

    def _branch_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if (not self.training) or p <= 0.0:
            return x
        keep = (torch.rand(x.size(0), 1, device=x.device) > p).float()
        return x * keep / (1.0 - p)

    def _reshape_patches(self, patches: torch.Tensor) -> torch.Tensor:
        n = patches.shape[0]
        x = patches.reshape(n, 9, self.n_timesteps, self.n_bands)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n, self.n_timesteps, self.n_bands, 3, 3)
        return x

    def forward(self, patches: torch.Tensor, indices: torch.Tensor):
        n = patches.shape[0]
        x = self._reshape_patches(patches)

        x = x.reshape(n * self.n_timesteps, self.n_bands, 3, 3)
        x = self.spatial(x)
        x = x.reshape(n, self.n_timesteps, -1)
        x = self.spatial_proj(x)
        spatial_vec = self.temporal(x)
        spatial_vec = self.spatial_norm(spatial_vec)

        idx_vec = self.index_branch(indices)
        idx_vec = self.index_norm(idx_vec)

        spatial_vec = self._branch_dropout(spatial_vec, self.spatial_branch_drop)
        idx_vec = self._branch_dropout(idx_vec, self.index_branch_drop)

        gate = self.fusion_gate(torch.cat([spatial_vec, idx_vec], dim=1))
        gated_idx = idx_vec * gate
        fused = torch.cat([spatial_vec, gated_idx], dim=1)

        logits = self.head(fused)
        spatial_logits = self.spatial_aux(spatial_vec)
        index_logits = self.index_aux(idx_vec)
        return {
            "logits": logits,
            "spatial_logits": spatial_logits,
            "index_logits": index_logits,
            "spatial_vec": spatial_vec,
            "index_vec": idx_vec,
            "fusion_gate": gate,
        }

    def predict(self, patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(patches, indices)
            return F.softmax(out["logits"], dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
