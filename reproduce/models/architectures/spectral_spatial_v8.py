import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# Our 12-band ordering per timestep:
# [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, VV, VH]
# (10 Sentinel-2 spectral + 2 Sentinel-1 SAR)
DEFAULT_BAND_MAP = {
    "blue": 0,
    "green": 1,
    "red": 2,
    "nir": 6,
    "swir1": 8,
    "swir2": 9,
}


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
        self.use_residual = in_ch == out_ch
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
    def __init__(self, channels: int, out_dim: int, dropout: float = 0.05):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(channels * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        center = fmap[:, :, 1, 1]
        total = fmap.sum(dim=(-2, -1))
        neigh_mean = (total - center) / 8.0
        contrast = center - neigh_mean
        feat = torch.cat([center, neigh_mean, contrast], dim=1)
        return self.proj(feat)


class SpatialEncoderV8(nn.Module):
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
        return self.readout(x)


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.12):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2,
                 n_heads: int = 4, n_positions: int = 6,
                 dropout: float = 0.12):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc = nn.Parameter(torch.randn(1, n_positions + 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, _ = x.shape
        cls = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_enc[:, :t + 1, :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x[:, 0]


class ResidualMLPBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.10):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = x + self.block(x)
        return self.out_proj(x)


class PriorFeatureExtractor(nn.Module):
    def __init__(self, n_timesteps: int = 6,
                 band_map: Optional[Dict[str, int]] = None,
                 eps: float = 1e-6):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.band_map = dict(DEFAULT_BAND_MAP if band_map is None else band_map)
        self.eps = eps
        self.feature_dim = 7 * 6

    def _safe_ratio(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b) / (a + b + self.eps)

    def _summary(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        mn = x.min(dim=1).values
        mx = x.max(dim=1).values
        amp = mx - mn
        slope = x[:, -1] - x[:, 0]
        return torch.stack([mean, std, mn, mx, amp, slope], dim=1)

    def _heuristic_anchor_logits(self,
                                 ndvi_s: torch.Tensor,
                                 ndwi_s: torch.Tensor,
                                 mndwi_s: torch.Tensor,
                                 ndbi_s: torch.Tensor,
                                 bsi_s: torch.Tensor,
                                 ndmi_s: torch.Tensor,
                                 nbr2_s: torch.Tensor) -> torch.Tensor:
        ndvi_mean, _, _, _, ndvi_amp, _ = ndvi_s.unbind(dim=1)
        ndwi_mean, _, _, ndwi_max, _, _ = ndwi_s.unbind(dim=1)
        mndwi_mean, _, _, mndwi_max, _, _ = mndwi_s.unbind(dim=1)
        ndbi_mean, _, _, _, _, _ = ndbi_s.unbind(dim=1)
        bsi_mean, _, _, _, _, _ = bsi_s.unbind(dim=1)
        ndmi_mean, _, _, _, _, _ = ndmi_s.unbind(dim=1)
        nbr2_mean, _, _, _, _, _ = nbr2_s.unbind(dim=1)

        tree = (
            4.0 * (ndvi_mean - 0.58)
            + 2.0 * (ndmi_mean - 0.05)
            - 2.0 * (bsi_mean - 0.02)
            - 2.0 * (ndbi_mean - 0.02)
            - 1.5 * torch.relu(ndvi_amp - 0.18)
        )
        shrub = (
            2.2 * (ndvi_mean - 0.32)
            + 1.0 * (ndmi_mean - 0.00)
            - 1.3 * (bsi_mean - 0.05)
            - 1.1 * (ndbi_mean - 0.03)
            - 2.0 * torch.abs(ndvi_amp - 0.18)
        )
        grass = (
            2.4 * (ndvi_mean - 0.30)
            + 2.6 * (ndvi_amp - 0.18)
            - 1.2 * (ndbi_mean - 0.02)
            - 1.1 * (bsi_mean - 0.08)
            - 1.0 * (mndwi_mean - 0.02)
        )
        crop = (
            2.5 * (ndvi_mean - 0.28)
            + 3.6 * (ndvi_amp - 0.24)
            + 0.8 * (ndmi_mean - 0.00)
            - 1.5 * (ndbi_mean - 0.03)
            - 1.0 * (mndwi_mean - 0.02)
        )
        built = (
            3.6 * (ndbi_mean - 0.04)
            + 1.2 * (bsi_mean - 0.04)
            - 3.0 * (mndwi_mean - 0.03)
            - 2.6 * (ndvi_mean - 0.25)
            - 1.2 * (ndmi_mean - 0.00)
        )
        bare = (
            4.0 * (bsi_mean - 0.12)
            + 1.8 * (nbr2_mean - 0.03)
            + 0.8 * (ndbi_mean - 0.02)
            - 3.5 * (ndvi_mean - 0.18)
            - 1.5 * (mndwi_mean - 0.02)
        )
        water = (
            4.2 * (mndwi_max - 0.12)
            + 1.5 * (ndwi_max - 0.03)
            - 3.2 * (ndbi_mean - 0.00)
            - 3.0 * (ndvi_mean - 0.18)
            - 2.0 * (bsi_mean - 0.03)
        )
        logits = torch.stack([tree, shrub, grass, crop, built, bare, water], dim=1)
        return 2.0 * logits

    def forward(self, center_raw: torch.Tensor):
        n = center_raw.shape[0]
        tb = center_raw.shape[1]
        n_bands = tb // self.n_timesteps
        x = center_raw.view(n, self.n_timesteps, n_bands)

        blue = x[:, :, self.band_map["blue"]]
        green = x[:, :, self.band_map["green"]]
        red = x[:, :, self.band_map["red"]]
        nir = x[:, :, self.band_map["nir"]]
        swir1 = x[:, :, self.band_map["swir1"]]
        swir2 = x[:, :, self.band_map["swir2"]]

        ndvi = self._safe_ratio(nir, red)
        ndwi = self._safe_ratio(green, nir)
        mndwi = self._safe_ratio(green, swir1)
        ndbi = self._safe_ratio(swir1, nir)
        ndmi = self._safe_ratio(nir, swir1)
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + self.eps)
        nbr2 = self._safe_ratio(swir1, swir2)

        summaries = [
            self._summary(ndvi), self._summary(ndwi), self._summary(mndwi),
            self._summary(ndbi), self._summary(bsi), self._summary(ndmi), self._summary(nbr2),
        ]
        summary = torch.cat(summaries, dim=1)
        anchor_logits = self._heuristic_anchor_logits(*summaries)
        anchor_probs = F.softmax(anchor_logits, dim=1)
        anchor_conf, anchor_pred = anchor_probs.max(dim=1)
        return {
            "summary": summary,
            "anchor_logits": anchor_logits,
            "anchor_probs": anchor_probs,
            "anchor_conf": anchor_conf,
            "anchor_pred": anchor_pred,
        }


class SoftThresholdPriorHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 n_thresholds: int = 2, dropout: float = 0.08):
        super().__init__()
        self.thresholds = nn.Parameter(torch.zeros(in_dim, n_thresholds))
        self.log_scales = nn.Parameter(torch.zeros(in_dim, n_thresholds))
        self.rule_norm = nn.LayerNorm(in_dim * n_thresholds * 2 + in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim * n_thresholds * 2 + in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hidden_to_logits = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor):
        diff = x.unsqueeze(-1) - self.thresholds.unsqueeze(0)
        scales = F.softplus(self.log_scales).unsqueeze(0) + 0.5
        hi = torch.sigmoid(scales * diff)
        lo = torch.sigmoid(-scales * diff)
        rules = torch.cat([hi, lo], dim=-1).reshape(x.size(0), -1)
        full = torch.cat([x, rules], dim=1)
        full = self.rule_norm(full)
        hidden = self.net(full)
        logits = self.hidden_to_logits(hidden)
        return hidden, logits


class CenterExpert(nn.Module):
    """
    Pixel-specialist expert that focuses on the center cell itself.

    This is the anti-over-smoothing branch: if the neighborhood screams one class but
    the center spectral signature really looks like another, this branch should be able
    to cast a strong local vote.
    """
    def __init__(self, center_dim: int = 72, idx_dim: int = 145,
                 hidden: int = 160, out_dim: int = 128, n_classes: int = 7,
                 dropout: float = 0.10):
        super().__init__()
        self.center_proj = ResidualMLPBranch(center_dim, hidden, out_dim, dropout=dropout)
        self.index_proj = ResidualMLPBranch(idx_dim, hidden, out_dim, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.logits = nn.Linear(out_dim, n_classes)
        self.conf = nn.Linear(out_dim, 1)

    def forward(self, center_scaled: torch.Tensor, indices: torch.Tensor):
        c = self.center_proj(center_scaled)
        i = self.index_proj(indices)
        feat = self.fuse(torch.cat([c, i, c - i], dim=1))
        logits = self.logits(feat)
        conf = torch.sigmoid(self.conf(feat)).squeeze(1)
        return feat, logits, conf


class BoundaryAnomalyHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 96, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ambiguity = nn.Linear(hidden_dim, 1)
        self.prior_gate = nn.Linear(hidden_dim, 1)
        self.center_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.ambiguity(h), self.prior_gate(h), self.center_gate(h)


class SpectralSpatialNetV8(nn.Module):
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
                 dropout: float = 0.12,
                 prior_hidden: int = 96,
                 band_map: Optional[Dict[str, int]] = None):
        super().__init__()
        self.n_bands = n_bands
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes

        self.spatial = SpatialEncoderV8(
            in_bands=n_bands, dims=spatial_dims, expand_ratio=expand_ratio,
            dropout=min(dropout, 0.10),
        )
        spatial_out = self.spatial.out_dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_out, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.temporal = TemporalEncoder(
            d_model=temporal_dim, n_layers=n_attn_layers, n_heads=n_heads,
            n_positions=n_timesteps, dropout=dropout,
        )

        idx_hidden = temporal_dim * 2
        self.index_branch = ResidualMLPBranch(n_indices, idx_hidden, temporal_dim, dropout=dropout * 0.9)
        self.spatial_norm = nn.LayerNorm(temporal_dim)
        self.index_norm = nn.LayerNorm(temporal_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, temporal_dim),
            nn.Sigmoid(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(temporal_dim * 3, temporal_dim * 2),
            nn.LayerNorm(temporal_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.neural_logits = nn.Linear(temporal_dim, n_classes)

        self.prior_features = PriorFeatureExtractor(n_timesteps=n_timesteps, band_map=band_map)
        self.prior_head = SoftThresholdPriorHead(
            in_dim=self.prior_features.feature_dim, hidden_dim=prior_hidden,
            n_classes=n_classes, n_thresholds=2, dropout=dropout * 0.75,
        )
        self.prior_norm = nn.LayerNorm(prior_hidden)

        self.center_expert = CenterExpert(
            center_dim=n_timesteps * n_bands, idx_dim=n_indices,
            hidden=160, out_dim=temporal_dim, n_classes=n_classes, dropout=dropout * 0.85,
        )
        self.center_norm = nn.LayerNorm(temporal_dim)

        self.spatial_aux = nn.Linear(temporal_dim, n_classes)
        self.index_aux = nn.Linear(temporal_dim, n_classes)
        self.prior_aux = nn.Linear(prior_hidden, n_classes)
        self.center_aux = nn.Linear(temporal_dim, n_classes)

        ambiguity_in = temporal_dim * 3 + prior_hidden + 5
        self.boundary_head = BoundaryAnomalyHead(ambiguity_in, hidden_dim=96, dropout=dropout * 0.75)

        self.anchor_scale = nn.Parameter(torch.tensor(0.35))
        self.center_scale = nn.Parameter(torch.tensor(0.65))

    def _reshape_patches(self, patches: torch.Tensor) -> torch.Tensor:
        n = patches.shape[0]
        x = patches.reshape(n, 9, self.n_timesteps, self.n_bands)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(n, self.n_timesteps, self.n_bands, 3, 3)

    def _heterogeneity_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, B, 3, 3) on scaled inputs
        center = x[:, :, :, 1, 1]
        neigh = x.reshape(x.size(0), x.size(1), x.size(2), 9)
        neigh_mean = (neigh.sum(dim=-1) - center) / 8.0
        center_jump = (center - neigh_mean).abs().mean(dim=(1, 2))
        patch_std = x.std(dim=(-2, -1), unbiased=False).mean(dim=(1, 2))
        temporal_std = center.std(dim=1, unbiased=False).mean(dim=1)
        neighbor_quad = (neigh[:, :, :, :4].mean(dim=-1) - neigh[:, :, :, 4:8].mean(dim=-1)).abs().mean(dim=(1, 2))
        neigh_only = torch.cat([neigh[:, :, :, :4], neigh[:, :, :, 5:]], dim=-1)
        neigh_std = neigh_only.std(dim=-1, unbiased=False).mean(dim=(1, 2))
        return torch.stack([center_jump, patch_std, temporal_std, neighbor_quad, neigh_std], dim=1)

    def forward(self, patches: torch.Tensor, indices: torch.Tensor,
                center_raw: torch.Tensor):
        n = patches.shape[0]
        x = self._reshape_patches(patches)
        center_scaled = x[:, :, :, 1, 1].reshape(n, -1)
        hetero = self._heterogeneity_features(x)

        x = x.reshape(n * self.n_timesteps, self.n_bands, 3, 3)
        x = self.spatial(x)
        x = x.reshape(n, self.n_timesteps, -1)
        x = self.spatial_proj(x)
        spatial_vec = self.temporal(x)
        spatial_vec = self.spatial_norm(spatial_vec)

        idx_vec = self.index_branch(indices)
        idx_vec = self.index_norm(idx_vec)

        gate = self.fusion_gate(torch.cat([spatial_vec, idx_vec], dim=1))
        gated_idx = idx_vec * gate
        fused = torch.cat([spatial_vec, gated_idx, spatial_vec - gated_idx], dim=1)
        fused = self.fusion_head(fused)
        neural_logits = self.neural_logits(fused)

        prior_pack = self.prior_features(center_raw.float())
        prior_hidden, prior_branch_logits = self.prior_head(prior_pack["summary"])
        prior_hidden = self.prior_norm(prior_hidden)
        prior_aux_logits = self.prior_aux(prior_hidden)

        center_vec, center_logits, center_conf = self.center_expert(center_scaled, indices)
        center_vec = self.center_norm(center_vec)
        center_aux_logits = self.center_aux(center_vec)

        boundary_score = torch.sigmoid(1.35 * (hetero[:, 1] + hetero[:, 4] + 0.5 * hetero[:, 3]) - 1.15)
        anomaly_score = torch.sigmoid(1.65 * (hetero[:, 0] + 0.35 * hetero[:, 2]) - 1.0 * (hetero[:, 4] + 0.35 * hetero[:, 3]) - 0.70)

        control_in = torch.cat([spatial_vec, idx_vec, center_vec, prior_hidden, hetero], dim=1)
        ambiguity_logit, prior_gate_logit, center_gate_logit = self.boundary_head(control_in)
        ambiguity = torch.sigmoid(ambiguity_logit).squeeze(1)
        prior_gate = torch.sigmoid(prior_gate_logit).squeeze(1)
        center_gate_base = torch.sigmoid(center_gate_logit).squeeze(1)

        anchor_strength = torch.clamp((prior_pack["anchor_conf"] - 0.78) / 0.22, min=0.0, max=1.0)
        center_strength = torch.clamp((center_conf - 0.62) / 0.38, min=0.0, max=1.0)
        center_gate = center_gate_base * center_strength * (0.40 + 0.60 * anomaly_score)

        final_logits = (
            neural_logits
            + prior_gate.unsqueeze(1) * prior_branch_logits
            + self.anchor_scale * anchor_strength.unsqueeze(1) * prior_pack["anchor_logits"]
            + self.center_scale * center_gate.unsqueeze(1) * center_logits
        )

        return {
            "logits": final_logits,
            "neural_logits": neural_logits,
            "spatial_logits": self.spatial_aux(spatial_vec),
            "index_logits": self.index_aux(idx_vec),
            "prior_logits": prior_aux_logits,
            "prior_branch_logits": prior_branch_logits,
            "center_logits": center_aux_logits,
            "center_branch_logits": center_logits,
            "spatial_vec": spatial_vec,
            "index_vec": idx_vec,
            "center_vec": center_vec,
            "prior_hidden": prior_hidden,
            "fusion_gate": gate,
            "ambiguity": ambiguity,
            "ambiguity_logit": ambiguity_logit.squeeze(1),
            "prior_gate": prior_gate,
            "prior_gate_logit": prior_gate_logit.squeeze(1),
            "center_gate": center_gate,
            "center_gate_logit": center_gate_logit.squeeze(1),
            "center_conf": center_conf,
            "heterogeneity": hetero,
            "boundary_score": boundary_score,
            "anomaly_score": anomaly_score,
            "anchor_logits": prior_pack["anchor_logits"],
            "anchor_probs": prior_pack["anchor_probs"],
            "anchor_conf": prior_pack["anchor_conf"],
            "anchor_pred": prior_pack["anchor_pred"],
            "prior_summary": prior_pack["summary"],
        }

    def predict(self, patches: torch.Tensor, indices: torch.Tensor,
                center_raw: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(patches, indices, center_raw)
            return F.softmax(out["logits"], dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
