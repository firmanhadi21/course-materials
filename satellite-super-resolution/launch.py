#!/usr/bin/env python3
"""
Launch BasicSR training/testing with S2/PS custom components.

Usage:
    # Train ESRGAN (PSNR stage)
    python launch.py -opt options/esrgan_s2ps_psnr.yml

    # Train SwinIR
    python launch.py -opt options/swinir_s2ps.yml

    # Multi-GPU
    python -m torch.distributed.launch --nproc_per_node=4 \\
        launch.py -opt options/swinir_s2ps.yml --launcher pytorch

    # Test / inference
    python launch.py -opt options/esrgan_s2ps_psnr.yml --test
"""

# ──────────────────────────────────────────────────────────────────────────────
# Fix BasicSR/torchvision compatibility (MUST be before any basicsr imports)
# Newer torchvision versions removed torchvision.transforms.functional_tensor
# ──────────────────────────────────────────────────────────────────────────────
import sys
from torchvision.transforms import functional as _F
sys.modules['torchvision.transforms.functional_tensor'] = _F

import argparse
import os

# Ensure parent dir is on path so our package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True,
                        help='Path to YAML option file')
    parser.add_argument('--launcher', default='none',
                        choices=['none', 'pytorch', 'slurm'])
    parser.add_argument('--test', action='store_true',
                        help='Run testing instead of training')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────
    #  CRITICAL: Import our package FIRST to register all
    #  custom datasets, archs, losses, metrics with BasicSR.
    # ──────────────────────────────────────────────────────────
    import s2ps_dataset   # noqa: registers datasets
    import s2ps_archs     # noqa: registers architectures
    import s2ps_losses    # noqa: registers losses
    import s2ps_metrics   # noqa: registers metrics
    import s2ps_model     # noqa: registers models (MultiBandSRModel)

    print('[S2PS] Custom components registered with BasicSR:')
    print('  Datasets: S2PSBasicSRDataset, S2PSLMDBDataset')
    print('  Archs:    MultiBandRRDBNet, MultiBandSwinIR, '
          'BandAdapterNet, MultiBandEDSR')
    print('  Models:   MultiBandSRModel')
    print('  Losses:   SAMLoss, ERGASLoss, CombinedSRLoss, '
          'SpectralGradientLoss')
    print('  Metrics:  calculate_sam, calculate_ergas, '
          'calculate_psnr_per_band, calculate_ssim_per_band')

    root_path = os.path.dirname(os.path.abspath(__file__))

    if args.test:
        from basicsr.test import test_pipeline
        test_pipeline(root_path)
    else:
        from basicsr.train import train_pipeline
        train_pipeline(root_path)


if __name__ == '__main__':
    main()
