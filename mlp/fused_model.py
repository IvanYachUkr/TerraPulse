"""
Hybrid masked fusion MLP.

Design goals:
    - Keep a large all-feature joint branch so raw cross-modal interactions are not lost
    - Keep an optical specialist branch so visual-only signals stay explicit
    - Keep a SAR compatibility head that predicts class presence / plausibility
    - Use the SAR mask structurally as a soft log-gate on logits
    - Add a small residual refiner focused more strongly on rare classes
    - Stay strictly MLP-only

Architecture:
    optical features -> OpticalBranch -> emb_opt + optical_aux_logits
    SAR features     -> SARBranch     -> emb_sar + mask_logits + mask_probs
    all features     -> JointBranch   -> emb_joint

    concat(emb_opt, emb_sar, emb_joint, detach(mask_logits))
        -> FusionBackbone -> fusion_hidden -> base_logits

    concat(fusion_hidden,
           emb_joint,
           optical_aux_logits,
           detach(mask_logits),
           stopgrad(softmax(base_logits)))
        -> RefinerBackbone -> refine_logits

    final_logits = base_logits
                 + gate_strength * log(gate_floor + (1-gate_floor) * detach(mask_probs))
                 + refine_strength * refine_logits * rare_class_scale

Important training detail:
    The structural gate uses DETACHED mask outputs by default.
    This means classification gradients do not rewrite the meaning of the mask head.
    The mask head is updated by its own BCE loss, while the SAR embedding can still
    help the main task through emb_sar.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PlainBlock


# ---------------------------------------------------------------------------
# Feature splitting
# ---------------------------------------------------------------------------

def split_feature_indices(columns: List[str]) -> Dict[str, List[int]]:
    """Split feature columns into optical and SAR index groups."""
    optical_idx = []
    sar_idx = []
    for i, col in enumerate(columns):
        if col.startswith("SAR_"):
            sar_idx.append(i)
        else:
            optical_idx.append(i)
    return {"optical": optical_idx, "sar": sar_idx}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backbone(
    in_features: int,
    widths: List[int],
    dropout: float,
    activation: str,
    norm_type: str,
):
    """Build a PlainBlock backbone and return (module, output_dim)."""
    if not widths:
        return nn.Identity(), in_features

    layers = []
    prev_dim = in_features
    for w in widths:
        layers.append(PlainBlock(prev_dim, w, dropout, activation, norm_type))
        prev_dim = w
    return nn.Sequential(*layers), prev_dim


def _apply_branch_dropout(emb: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Sample-wise whole-branch dropout on an embedding block."""
    if (not training) or p <= 0.0:
        return emb

    keep_prob = 1.0 - p
    mask = (torch.rand(emb.shape[0], 1, device=emb.device) < keep_prob).to(emb.dtype)
    return emb * (mask / max(keep_prob, 1e-6))


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------

class AuxBranch(nn.Module):
    """MLP branch that produces an embedding plus auxiliary distribution logits."""

    def __init__(
        self,
        in_features: int,
        widths: List[int],
        n_classes: int,
        dropout: float = 0.15,
        activation: str = "silu",
        input_dropout: float = 0.05,
        norm_type: str = "batchnorm",
    ):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.backbone, out_dim = _make_backbone(
            in_features, widths, dropout, activation, norm_type
        )
        self.aux_head = nn.Linear(out_dim, n_classes)
        self.embed_dim = out_dim

    def forward(self, x: torch.Tensor):
        emb = self.backbone(self.input_drop(x))
        aux_logits = self.aux_head(emb)
        return emb, aux_logits


class EmbedBranch(nn.Module):
    """MLP branch that only produces an embedding."""

    def __init__(
        self,
        in_features: int,
        widths: List[int],
        dropout: float = 0.15,
        activation: str = "silu",
        input_dropout: float = 0.05,
        norm_type: str = "batchnorm",
    ):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.backbone, out_dim = _make_backbone(
            in_features, widths, dropout, activation, norm_type
        )
        self.embed_dim = out_dim

    def forward(self, x: torch.Tensor):
        return self.backbone(self.input_drop(x))


class SARBranch(nn.Module):
    """SAR branch: shared backbone + explicit class-compatibility mask head."""

    def __init__(
        self,
        in_features: int,
        widths: List[int],
        n_classes: int,
        dropout: float = 0.15,
        activation: str = "silu",
        input_dropout: float = 0.05,
        norm_type: str = "batchnorm",
    ):
        super().__init__()
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.backbone, out_dim = _make_backbone(
            in_features, widths, dropout, activation, norm_type
        )
        self.mask_head = nn.Linear(out_dim, n_classes)
        self.embed_dim = out_dim

    def forward(self, x: torch.Tensor):
        emb = self.backbone(self.input_drop(x))
        mask_logits = self.mask_head(emb)
        mask_probs = torch.sigmoid(mask_logits)
        return emb, mask_logits, mask_probs


# ---------------------------------------------------------------------------
# Hybrid masked fusion model
# ---------------------------------------------------------------------------

class HybridMaskedFusionMLP(nn.Module):
    """Optical specialist + SAR mask + all-feature interaction branch.

    Split-head architecture: common classes use softmax, rare classes
    (shrubland, bare_sparse) use independent sigmoid heads.
    """

    # Default: shrubland=1, bare_sparse=5 are rare
    DEFAULT_RARE_INDICES = [1, 5]

    def __init__(
        self,
        n_optical: int,
        n_sar: int,
        n_all: int,
        n_classes: int,
        optical_widths: List[int],
        sar_widths: List[int],
        joint_widths: List[int],
        fusion_widths: List[int],
        refiner_widths: List[int],
        optical_idx: List[int],
        sar_idx: List[int],
        rare_indices: Optional[List[int]] = None,
        rare_class_scale: Optional[List[float]] = None,
        gate_strength: float = 0.80,
        gate_floor: float = 0.10,
        refine_strength: float = 0.60,
        optical_branch_drop: float = 0.04,
        sar_branch_drop: float = 0.04,
        joint_branch_drop: float = 0.10,
        **hparams,
    ):
        super().__init__()
        dropout = hparams.get("dropout", 0.15)
        activation = hparams.get("activation", "silu")
        input_dropout = hparams.get("input_dropout", 0.05)
        norm_type = hparams.get("norm_type", "batchnorm")

        self.register_buffer("optical_idx", torch.tensor(optical_idx, dtype=torch.long))
        self.register_buffer("sar_idx", torch.tensor(sar_idx, dtype=torch.long))

        # Split-head: rare vs common class indices
        if rare_indices is None:
            rare_indices = self.DEFAULT_RARE_INDICES
        common_indices = [i for i in range(n_classes) if i not in rare_indices]
        self.register_buffer("rare_idx", torch.tensor(rare_indices, dtype=torch.long))
        self.register_buffer("common_idx", torch.tensor(common_indices, dtype=torch.long))
        self.n_rare = len(rare_indices)
        self.n_common = len(common_indices)

        if rare_class_scale is None:
            rare_class_scale = [1.0] * n_classes
        self.register_buffer(
            "rare_class_scale",
            torch.tensor(rare_class_scale, dtype=torch.float32),
        )

        self.gate_strength = float(gate_strength)
        self.gate_floor = float(gate_floor)
        self.refine_strength = float(refine_strength)
        self.optical_branch_drop = float(optical_branch_drop)
        self.sar_branch_drop = float(sar_branch_drop)
        self.joint_branch_drop = float(joint_branch_drop)
        self.n_classes = int(n_classes)

        self.optical_branch = AuxBranch(
            n_optical,
            optical_widths,
            n_classes,
            dropout=dropout,
            activation=activation,
            input_dropout=input_dropout,
            norm_type=norm_type,
        )
        self.sar_branch = SARBranch(
            n_sar,
            sar_widths,
            n_classes,
            dropout=dropout,
            activation=activation,
            input_dropout=input_dropout,
            norm_type=norm_type,
        )
        self.joint_branch = EmbedBranch(
            n_all,
            joint_widths,
            dropout=dropout,
            activation=activation,
            input_dropout=input_dropout,
            norm_type=norm_type,
        )

        fusion_in = (
            self.optical_branch.embed_dim
            + self.sar_branch.embed_dim
            + self.joint_branch.embed_dim
            + n_classes
        )
        self.fusion_backbone, fusion_out = _make_backbone(
            fusion_in, fusion_widths, dropout, activation, norm_type
        )
        # Split heads: common classes (softmax) + rare classes (sigmoid)
        self.common_head = nn.Linear(fusion_out, self.n_common)
        self.rare_head = nn.Linear(fusion_out, self.n_rare)
        # Near-zero init for rare head to avoid blowup
        with torch.no_grad():
            self.rare_head.weight.mul_(0.01)
            self.rare_head.bias.zero_()
        self.fusion_dim = fusion_out

        refiner_in = (
            fusion_out
            + self.joint_branch.embed_dim
            + n_classes   # optical aux logits
            + n_classes   # detached mask logits
            + n_classes   # detached base probs (common+rare combined)
        )
        self.refiner_backbone, refiner_out = _make_backbone(
            refiner_in, refiner_widths, dropout, activation, norm_type
        )
        # Refiner also has split heads
        self.refiner_common_head = nn.Linear(refiner_out, self.n_common)
        self.refiner_rare_head = nn.Linear(refiner_out, self.n_rare)
        # Near-zero init for refiner heads
        with torch.no_grad():
            self.refiner_common_head.weight.mul_(0.01)
            self.refiner_common_head.bias.zero_()
            self.refiner_rare_head.weight.mul_(0.01)
            self.refiner_rare_head.bias.zero_()
        self.refiner_dim = refiner_out

    def forward(
        self,
        x: torch.Tensor,
        gate_scale: float = 1.0,
        refine_scale: float = 1.0,
        detach_mask_inputs: bool = True,
        apply_branch_dropout: bool = True,
    ) -> Dict[str, torch.Tensor]:
        x_optical = x[:, self.optical_idx]
        x_sar = x[:, self.sar_idx]

        emb_opt, optical_logits = self.optical_branch(x_optical)
        emb_sar, mask_logits, mask_probs = self.sar_branch(x_sar)
        emb_joint = self.joint_branch(x)

        if apply_branch_dropout:
            emb_opt = _apply_branch_dropout(emb_opt, self.optical_branch_drop, self.training)
            emb_sar = _apply_branch_dropout(emb_sar, self.sar_branch_drop, self.training)
            emb_joint = _apply_branch_dropout(emb_joint, self.joint_branch_drop, self.training)

        if detach_mask_inputs:
            mask_logits_for_fusion = mask_logits.detach()
            mask_probs_for_gate = mask_probs.detach()
        else:
            mask_logits_for_fusion = mask_logits
            mask_probs_for_gate = mask_probs

        fusion_input = torch.cat(
            [emb_opt, emb_sar, emb_joint, mask_logits_for_fusion],
            dim=-1,
        )
        fusion_hidden = self.fusion_backbone(fusion_input)
        common_logits = self.common_head(fusion_hidden)
        rare_logits = self.rare_head(fusion_hidden)

        # Build full n_classes base probs for refiner context
        # (common via softmax scaled, rare via sigmoid)
        # Force float32: this is detached context, AMP dtype doesn't matter here
        with torch.no_grad():
            rare_probs_det = torch.sigmoid(rare_logits.detach().float())
            rare_sum_det = rare_probs_det.sum(dim=-1, keepdim=True).clamp(max=0.95)
            common_probs_det = torch.softmax(common_logits.detach().float(), dim=-1) * (1.0 - rare_sum_det)
            # Assemble full 7-class probs for refiner input
            full_probs = torch.zeros(
                x.size(0), self.n_classes,
                device=x.device, dtype=torch.float32,
            )
            full_probs[:, self.common_idx] = common_probs_det
            full_probs[:, self.rare_idx] = rare_probs_det

        refiner_input = torch.cat(
            [
                fusion_hidden,
                emb_joint,
                optical_logits,
                mask_logits_for_fusion,
                full_probs,          # detached base probs (7-class)
            ],
            dim=-1,
        )
        refiner_hidden = self.refiner_backbone(refiner_input)
        refine_common_raw = self.refiner_common_head(refiner_hidden)
        refine_rare_raw = self.refiner_rare_head(refiner_hidden)
        # Scale by rare_class_scale for both
        refine_common = refine_common_raw * self.rare_class_scale[self.common_idx].unsqueeze(0)
        refine_rare = refine_rare_raw * self.rare_class_scale[self.rare_idx].unsqueeze(0)

        # Gate bias: only for common classes (SAR presence is binary)
        gate_probs_common = (
            self.gate_floor
            + (1.0 - self.gate_floor) * mask_probs_for_gate[:, self.common_idx]
        )
        gate_bias_common = self.gate_strength * float(gate_scale) * torch.log(
            gate_probs_common.clamp_min(1e-6)
        )

        # Final outputs: common logits with gate + refiner, rare logits with refiner only
        final_common_logits = (
            common_logits
            + gate_bias_common
            + self.refine_strength * float(refine_scale) * refine_common
        )
        final_rare_logits = (
            rare_logits
            + self.refine_strength * float(refine_scale) * refine_rare
        )

        return {
            "final_common_logits": final_common_logits,
            "final_rare_logits": final_rare_logits,
            "common_logits": common_logits,       # base common (pre-gate/refine)
            "rare_logits": rare_logits,             # base rare (pre-refine)
            "optical_logits": optical_logits,
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
            "mask_logits_for_fusion": mask_logits_for_fusion,
            "gate_bias_common": gate_bias_common,
            "refine_common": refine_common,
            "refine_rare": refine_rare,
            "fusion_hidden": fusion_hidden,
            "emb_opt": emb_opt,
            "emb_sar": emb_sar,
            "emb_joint": emb_joint,
        }

    def _recombine(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Recombine split heads into 7-class probabilities that sum to 1."""
        rare_probs = torch.sigmoid(out["final_rare_logits"])
        rare_sum = rare_probs.sum(dim=-1, keepdim=True)
        # If rare fractions exceed 0.95, scale them down proportionally
        scale = torch.where(
            rare_sum > 0.95,
            0.95 / rare_sum.clamp_min(1e-8),
            torch.ones_like(rare_sum),
        )
        rare_probs = rare_probs * scale
        rare_sum = rare_probs.sum(dim=-1, keepdim=True)
        common_probs = torch.softmax(out["final_common_logits"], dim=-1)
        common_probs = common_probs * (1.0 - rare_sum)

        full = torch.zeros(rare_probs.size(0), self.n_classes, device=rare_probs.device)
        full[:, self.common_idx] = common_probs
        full[:, self.rare_idx] = rare_probs
        return full

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """7-class probabilities from the split heads."""
        self.eval()
        with torch.no_grad():
            out = self.forward(
                x,
                gate_scale=1.0,
                refine_scale=1.0,
                detach_mask_inputs=True,
                apply_branch_dropout=False,
            )
            return self._recombine(out)

    def export_forward(self, x: torch.Tensor) -> torch.Tensor:
        """7-class probabilities for ONNX export."""
        out = self.forward(
            x,
            gate_scale=1.0,
            refine_scale=1.0,
            detach_mask_inputs=True,
            apply_branch_dropout=False,
        )
        return self._recombine(out)


def make_fused_model(
    n_features: int,
    n_classes: int,
    feature_columns: List[str],
    **hparams,
) -> HybridMaskedFusionMLP:
    """Create the hybrid masked fusion model."""
    idx = split_feature_indices(feature_columns)
    n_optical = len(idx["optical"])
    n_sar = len(idx["sar"])

    return HybridMaskedFusionMLP(
        n_optical=n_optical,
        n_sar=n_sar,
        n_all=n_features,
        n_classes=n_classes,
        optical_widths=hparams.pop("optical_widths", [512, 256]),
        sar_widths=hparams.pop("sar_widths", [256, 128]),
        joint_widths=hparams.pop("joint_widths", [1024, 512, 256]),
        fusion_widths=hparams.pop("fusion_widths", [512, 256]),
        refiner_widths=hparams.pop("refiner_widths", [256, 128]),
        optical_idx=idx["optical"],
        sar_idx=idx["sar"],
        rare_indices=hparams.pop("rare_indices", None),
        rare_class_scale=hparams.pop("rare_class_scale", None),
        gate_strength=hparams.pop("gate_strength", 0.80),
        gate_floor=hparams.pop("gate_floor", 0.10),
        refine_strength=hparams.pop("refine_strength", 0.60),
        optical_branch_drop=hparams.pop("optical_branch_drop", 0.04),
        sar_branch_drop=hparams.pop("sar_branch_drop", 0.04),
        joint_branch_drop=hparams.pop("joint_branch_drop", 0.10),
        **hparams,
    )
