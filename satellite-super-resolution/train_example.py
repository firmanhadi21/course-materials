"""
Example Training Script: Sentinel-2 → PlanetScope Super-Resolution
===================================================================

A minimal but complete training loop using the S2PSPatchDataset
and a lightweight EDSR model (easily swappable for ESRGAN, SwinIR, etc.)

Usage:
    python train_example.py \
        --lr_dir data/patches/train/sentinel2 \
        --hr_dir data/patches/train/planetscope \
        --val_lr_dir data/patches/val/sentinel2 \
        --val_hr_dir data/patches/val/planetscope \
        --epochs 100 \
        --batch_size 16 \
        --scale_factor 3 \
        --num_channels 4 \
        --lr 1e-4

Requires: torch, numpy, rasterio (for .tif), scikit-image (for SSIM)
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    S2PSPatchDataset,
    SRMetrics,
    create_dataloaders,
)


# ============================================================================
# Lightweight EDSR Model (Baseline)
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, num_feat: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.res_scale = 0.1  # Residual scaling for stability

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale


class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution (simplified).

    A clean baseline model for satellite image SR.
    Can be swapped out for any model from BasicSR.
    """

    def __init__(
        self,
        num_in_ch: int = 4,
        num_out_ch: int = 4,
        num_feat: int = 64,
        num_block: int = 16,
        scale_factor: int = 3,
    ):
        super().__init__()

        # Head
        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Body (residual blocks)
        body = [ResidualBlock(num_feat) for _ in range(num_block)]
        body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body = nn.Sequential(*body)

        # Upsampler (sub-pixel convolution)
        self.upsample = self._make_upsampler(
            num_feat, scale_factor
        )

        # Tail
        self.tail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def _make_upsampler(self, num_feat, scale):
        """Create pixel-shuffle upsampling layers."""
        layers = []
        if scale in (2, 4, 8):
            for _ in range(int(np.log2(scale))):
                layers.append(
                    nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
                )
                layers.append(nn.PixelShuffle(2))
                layers.append(nn.ReLU(inplace=True))
        elif scale == 3:
            layers.append(
                nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1)
            )
            layers.append(nn.PixelShuffle(3))
            layers.append(nn.ReLU(inplace=True))
        else:
            # General: use interpolation + conv
            layers.append(
                nn.Upsample(scale_factor=scale, mode='bilinear',
                            align_corners=False)
            )
            layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        body = body + head  # Global residual
        up = self.upsample(body)
        out = self.tail(up)
        return out


# ============================================================================
# Loss Functions
# ============================================================================

class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss (differentiable)."""

    def forward(self, pred, target):
        # pred, target: (B, C, H, W)
        dot = (pred * target).sum(dim=1, keepdim=True)
        norm_p = pred.norm(dim=1, keepdim=True).clamp(min=1e-8)
        norm_t = target.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_angle = (dot / (norm_p * norm_t)).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos_angle).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: L1 + SAM.

    For GAN-based training, add perceptual and adversarial losses.
    """

    def __init__(self, l1_weight=1.0, sam_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sam = SAMLoss()
        self.l1_weight = l1_weight
        self.sam_weight = sam_weight

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_sam = self.sam(pred, target)
        return (
            self.l1_weight * loss_l1 + self.sam_weight * loss_sam,
            {'l1': loss_l1.item(), 'sam': loss_sam.item()},
        )


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_l1 = 0
    total_sam = 0
    n_batches = 0

    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)

        loss, loss_dict = criterion(sr_imgs, hr_imgs)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_l1 += loss_dict['l1']
        total_sam += loss_dict['sam']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'l1': total_l1 / n_batches,
        'sam': total_sam / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, metrics_fn, device, scale_factor):
    model.eval()
    all_psnr = []
    all_ssim = []
    all_sam = []

    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.to(device)
        sr_imgs = model(lr_imgs).cpu().numpy()
        hr_imgs = hr_imgs.numpy()

        for i in range(sr_imgs.shape[0]):
            m = metrics_fn.evaluate(
                sr_imgs[i], hr_imgs[i], scale_factor
            )
            all_psnr.append(m['PSNR'])
            all_ssim.append(m['SSIM'])
            all_sam.append(m['SAM'])

    return {
        'PSNR': np.mean(all_psnr),
        'SSIM': np.mean(all_ssim),
        'SAM': np.mean(all_sam),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scale_factor', type=int, default=3)
    parser.add_argument('--num_channels', type=int, default=4)
    parser.add_argument('--num_feat', type=int, default=64)
    parser.add_argument('--num_block', type=int, default=16)
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--sam_weight', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    if args.val_lr_dir and args.val_hr_dir:
        # Separate val directories provided
        train_dataset = S2PSPatchDataset(
            lr_dir=args.lr_dir,
            hr_dir=args.hr_dir,
            scale_factor=args.scale_factor,
            augment=True,
        )
        val_dataset = S2PSPatchDataset(
            lr_dir=args.val_lr_dir,
            hr_dir=args.val_hr_dir,
            scale_factor=args.scale_factor,
            augment=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        # Auto-split from single directory
        train_loader, val_loader, _ = create_dataloaders(
            lr_dir=args.lr_dir,
            hr_dir=args.hr_dir,
            batch_size=args.batch_size,
            scale_factor=args.scale_factor,
            num_workers=args.num_workers,
        )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = EDSR(
        num_in_ch=args.num_channels,
        num_out_ch=args.num_channels,
        num_feat=args.num_feat,
        num_block=args.num_block,
        scale_factor=args.scale_factor,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: EDSR ({n_params:.2f}M parameters)")

    # ---------------------------------------------------------------
    # Optimizer, Scheduler, Loss
    # ---------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    criterion = CombinedLoss(
        l1_weight=args.l1_weight,
        sam_weight=args.sam_weight,
    )
    metrics = SRMetrics()

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    best_psnr = 0

    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'L1':>8} | {'SAM':>8} | "
          f"{'PSNR':>7} | {'SSIM':>7} | {'SAM°':>7} | {'Time':>6}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(
            model, val_loader, metrics, device, args.scale_factor
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Print progress
        print(
            f"{epoch:5d} | "
            f"{train_metrics['loss']:8.4f} | "
            f"{train_metrics['l1']:8.4f} | "
            f"{train_metrics['sam']:8.4f} | "
            f"{val_metrics['PSNR']:7.2f} | "
            f"{val_metrics['SSIM']:7.4f} | "
            f"{val_metrics['SAM']:7.2f} | "
            f"{elapsed:5.1f}s"
        )

        # Save best model
        if val_metrics['PSNR'] > best_psnr:
            best_psnr = val_metrics['PSNR']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': best_psnr,
                    'config': vars(args),
                },
                os.path.join(args.output_dir, 'best_model.pth'),
            )
            print(f"  → New best PSNR: {best_psnr:.2f} dB (saved)")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': vars(args),
                },
                os.path.join(
                    args.output_dir, f'checkpoint_epoch{epoch:03d}.pth'
                ),
            )

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in: {args.output_dir}/")


if __name__ == '__main__':
    main()
