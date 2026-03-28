"""
Remote-Sensing Losses for BasicSR
===================================

Custom loss functions for spectral fidelity in satellite image
super-resolution, registered with BasicSR's LOSS_REGISTRY.

Use in YAML configs as::

    train:
      pixel_opt:
        type: L1Loss
        loss_weight: 1.0
      sam_opt:
        type: SAMLoss
        loss_weight: 0.1
      # --- or use the combined wrapper ---
      combined_opt:
        type: CombinedSRLoss
        l1_weight: 1.0
        sam_weight: 0.1
        loss_weight: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) loss.

    Measures the spectral angle between predicted and target pixel vectors.
    Critical for remote-sensing SR to preserve band ratios (e.g. NDVI).

    SAM = arccos( dot(x, y) / (||x|| * ||y||) )

    Lower is better.  Range: [0, π/2] for non-negative reflectance.
    """

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W) predicted SR image
            target: (B, C, H, W) ground-truth HR image
        Returns:
            Scalar mean SAM in radians, weighted by loss_weight.
        """
        # Flatten spatial dims → (B, C, N)
        b, c, h, w = pred.shape
        p = pred.reshape(b, c, -1)
        t = target.reshape(b, c, -1)

        dot = (p * t).sum(dim=1)                     # (B, N)
        norm_p = p.norm(dim=1).clamp(min=self.eps)    # (B, N)
        norm_t = t.norm(dim=1).clamp(min=self.eps)    # (B, N)

        cos_angle = (dot / (norm_p * norm_t)).clamp(-1 + self.eps,
                                                      1 - self.eps)
        sam = torch.acos(cos_angle)  # (B, N)

        return self.loss_weight * sam.mean()


@LOSS_REGISTRY.register()
class ERGASLoss(nn.Module):
    """
    Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) loss.

    Standard pan-sharpening / SR quality metric used as a differentiable
    loss.  Computes per-band RMSE normalised by band mean.

    ERGAS = 100 * (h/l) * sqrt( (1/C) * Σ (RMSE_c / μ_c)² )

    where h/l is the resolution ratio and μ_c is the band mean of GT.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        scale: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = scale
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        b, c, h, w = pred.shape
        # Per-band RMSE
        mse_per_band = ((pred - target) ** 2).mean(dim=(2, 3))  # (B, C)
        rmse = mse_per_band.sqrt()

        # Per-band mean of GT
        mu = target.mean(dim=(2, 3)).clamp(min=self.eps)  # (B, C)

        ratio = rmse / mu  # (B, C)
        ergas = 100.0 / self.scale * (ratio ** 2).mean(dim=1).sqrt()  # (B,)
        return self.loss_weight * ergas.mean()


@LOSS_REGISTRY.register()
class CombinedSRLoss(nn.Module):
    """
    Convenience wrapper:  L1 + SAM + optional Charbonnier.

    YAML::

        combined_opt:
          type: CombinedSRLoss
          l1_weight: 1.0
          sam_weight: 0.1
          charbonnier_weight: 0.0
          loss_weight: 1.0
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        l1_weight: float = 1.0,
        sam_weight: float = 0.1,
        charbonnier_weight: float = 0.0,
        charbonnier_eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1_weight = l1_weight
        self.sam_weight = sam_weight
        self.charbonnier_weight = charbonnier_weight
        self.charbonnier_eps = charbonnier_eps

        if sam_weight > 0:
            self.sam_loss = SAMLoss(loss_weight=1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)

        if self.l1_weight > 0:
            loss = loss + self.l1_weight * F.l1_loss(pred, target)

        if self.sam_weight > 0:
            loss = loss + self.sam_weight * self.sam_loss(pred, target)

        if self.charbonnier_weight > 0:
            diff = pred - target
            loss = loss + self.charbonnier_weight * torch.sqrt(
                diff * diff + self.charbonnier_eps ** 2
            ).mean()

        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class SpectralGradientLoss(nn.Module):
    """
    Preserves spectral gradient structure across bands.

    Computes the L1 distance between inter-band differences of
    prediction vs. target.  Encourages consistent spectral slopes
    (important for vegetation indices, water indices, etc.).
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Inter-band differences: (B, C-1, H, W)
        pred_grad = pred[:, 1:] - pred[:, :-1]
        tgt_grad = target[:, 1:] - target[:, :-1]
        return self.loss_weight * F.l1_loss(pred_grad, tgt_grad)
