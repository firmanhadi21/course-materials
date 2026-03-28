"""
Remote-Sensing Metrics for BasicSR
====================================

Registered with BasicSR's METRIC_REGISTRY so they appear automatically
in validation logs.

YAML usage::

    val:
      metrics:
        psnr:
          type: calculate_psnr
          crop_border: 0
          test_y_channel: false   # multi-band, not Y-channel
        sam:
          type: calculate_sam
        ergas:
          type: calculate_ergas
          scale: 3
"""

import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_sam(
    img: np.ndarray,
    img2: np.ndarray,
    **kwargs,
) -> float:
    """
    Spectral Angle Mapper (SAM) — lower is better.

    Args:
        img:  (H, W, C) or (C, H, W) predicted image, float32, [0, 1]
        img2: (H, W, C) or (C, H, W) ground truth image, float32, [0, 1]

    Returns:
        Mean SAM in degrees.
    """
    # Squeeze batch dimension if present
    if img.ndim == 4:
        img = img.squeeze(0)
        img2 = img2.squeeze(0)

    # Handle (C, H, W) format from PyTorch - channels < spatial dims
    if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

    eps = 1e-8
    # Flatten to (N, C)
    p = img.reshape(-1, img.shape[-1]).astype(np.float64)
    t = img2.reshape(-1, img2.shape[-1]).astype(np.float64)

    dot = (p * t).sum(axis=1)
    norm_p = np.linalg.norm(p, axis=1).clip(min=eps)
    norm_t = np.linalg.norm(t, axis=1).clip(min=eps)

    cos_angle = np.clip(dot / (norm_p * norm_t), -1 + eps, 1 - eps)
    sam_rad = np.arccos(cos_angle)
    return float(np.degrees(sam_rad.mean()))


@METRIC_REGISTRY.register()
def calculate_ergas(
    img: np.ndarray,
    img2: np.ndarray,
    scale: int = 3,
    **kwargs,
) -> float:
    """
    ERGAS — lower is better.

    Args:
        img:   (H, W, C) or (C, H, W) predicted
        img2:  (H, W, C) or (C, H, W) ground truth
        scale: SR scale factor
    """
    # Squeeze batch dimension if present
    if img.ndim == 4:
        img = img.squeeze(0)
        img2 = img2.squeeze(0)

    # Handle (C, H, W) format from PyTorch - channels < spatial dims
    if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

    eps = 1e-8
    c = img.shape[-1]
    ergas_sum = 0.0
    for band in range(c):
        p = img[:, :, band].astype(np.float64)
        t = img2[:, :, band].astype(np.float64)
        rmse = np.sqrt(np.mean((p - t) ** 2))
        mu = np.mean(t)
        ergas_sum += (rmse / max(mu, eps)) ** 2

    ergas = 100.0 / scale * np.sqrt(ergas_sum / c)
    return float(ergas)


@METRIC_REGISTRY.register()
def calculate_psnr_per_band(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 0,
    **kwargs,
) -> float:
    """
    Per-band PSNR averaged across channels.

    BasicSR's built-in PSNR works on Y-channel or single-band.
    This computes per-band PSNR and returns the mean.

    Handles (H, W, C), (C, H, W), and 4D batch formats.
    """
    # Squeeze batch dimension if present
    if img.ndim == 4:
        img = img.squeeze(0)
        img2 = img2.squeeze(0)

    # Handle (C, H, W) format from PyTorch - channels < spatial dims
    if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    c = img.shape[-1]
    psnrs = []
    for band in range(c):
        mse = np.mean((img[:, :, band].astype(np.float64) -
                        img2[:, :, band].astype(np.float64)) ** 2)
        if mse < 1e-10:
            psnrs.append(100.0)
        else:
            psnrs.append(10.0 * np.log10(1.0 / mse))
    return float(np.mean(psnrs))


@METRIC_REGISTRY.register()
def calculate_ssim_per_band(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 0,
    **kwargs,
) -> float:
    """
    Per-band SSIM averaged across channels.
    Uses scikit-image structural_similarity.

    Handles (H, W, C), (C, H, W), and 4D batch formats.
    """
    from skimage.metrics import structural_similarity as ssim

    # Squeeze batch dimension if present (B, C, H, W) -> (C, H, W)
    if img.ndim == 4:
        img = img.squeeze(0)
        img2 = img2.squeeze(0)

    # Handle (C, H, W) format from PyTorch - channels < spatial dims
    if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    c = img.shape[-1]
    h, w = img.shape[:2]

    # Adjust win_size for small images (must be odd and <= min dimension)
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(3, win_size)

    ssims = []
    for band in range(c):
        val = ssim(
            img[:, :, band], img2[:, :, band],
            data_range=1.0,
            win_size=win_size,
        )
        ssims.append(val)
    return float(np.mean(ssims))
