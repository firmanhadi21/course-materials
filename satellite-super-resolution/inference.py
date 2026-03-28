#!/usr/bin/env python3
"""
Inference Script — Apply Trained SR Model to Full Sentinel-2 Scenes
=====================================================================

Processes full-resolution Sentinel-2 scenes through a trained BasicSR
model using tiled (sliding-window) inference to handle arbitrarily
large images within GPU memory.

Output: GeoTIFF at PlanetScope resolution (~3m) with correct
georeferencing and projection metadata.

Usage:
    # Single scene
    python inference.py \\
        --model_path experiments/ESRGAN_S2PS_PSNR_4band_x3/models/net_g_latest.pth \\
        --arch MultiBandRRDBNet \\
        --input scene_10m.tif \\
        --output scene_3m_sr.tif \\
        --scale 3

    # Batch (directory of scenes)
    python inference.py \\
        --model_path experiments/SwinIR_S2PS_4band_x3/models/net_g_latest.pth \\
        --arch MultiBandSwinIR \\
        --input_dir sentinel2_scenes/ \\
        --output_dir sr_output/ \\
        --scale 3 --tile_size 64 --tile_overlap 8

    # With band-adapter wrapper
    python inference.py \\
        --model_path experiments/BandAdapter_ESRGAN/models/net_g_latest.pth \\
        --arch BandAdapterNet \\
        --arch_opts '{"num_bands": 4, "backbone": {"type": "RRDBNet", ...}}' \\
        --input scene.tif --output sr.tif
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Patch torchvision compatibility (functional_tensor removed in torchvision ≥0.20)
import types
import torchvision.transforms
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    _ft = types.ModuleType('torchvision.transforms.functional_tensor')
    from torchvision.transforms.functional import rgb_to_grayscale
    _ft.rgb_to_grayscale = rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _ft

# Register custom archs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import s2ps_archs   # noqa
except ImportError:
    pass


def load_model(
    model_path: str,
    arch: str = 'MultiBandRRDBNet',
    arch_opts: dict = None,
    device: str = 'cuda',
) -> torch.nn.Module:
    """Load a trained generator from checkpoint."""
    from basicsr.utils.registry import ARCH_REGISTRY

    if arch_opts is None:
        # Defaults for each arch
        defaults = {
            'MultiBandRRDBNet': dict(
                num_in_ch=4, num_out_ch=4, scale=3,
                num_feat=64, num_block=23, num_grow_ch=32,
            ),
            'MultiBandSwinIR': dict(
                num_in_ch=4, num_out_ch=4, upscale=3,
                img_size=64, window_size=8,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6],
                embed_dim=180, mlp_ratio=2,
                upsampler='pixelshuffle', resi_connection='1conv',
            ),
            'MultiBandEDSR': dict(
                num_in_ch=4, num_out_ch=4, num_feat=64,
                num_block=16, scale=3,
            ),
        }
        arch_opts = defaults.get(arch, {})

    model_cls = ARCH_REGISTRY.get(arch)
    model = model_cls(**arch_opts)

    # Load weights
    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    # Handle BasicSR checkpoint formats
    if 'params_ema' in ckpt:
        state_dict = ckpt['params_ema']
    elif 'params' in ckpt:
        state_dict = ckpt['params']
    else:
        state_dict = ckpt

    # Strip 'module.' prefix from DDP wrapping
    clean = {}
    for k, v in state_dict.items():
        clean[k.replace('module.', '')] = v

    # Try loading as-is first; if that fails, strip 'net.' prefix
    # (needed for BandAdapterNet checkpoints)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing and unexpected:
        # Likely a prefix mismatch — try stripping 'net.'
        clean2 = {k.replace('net.', '', 1): v for k, v in clean.items()}
        missing2, _ = model.load_state_dict(clean2, strict=False)
        if len(missing2) < len(missing):
            missing = missing2
            print(f'[Model] Stripped net. prefix for loading')

    if missing:
        print(f'[Model] Warning: {len(missing)} missing keys (first 5: '
              f'{missing[:5]})')
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[Model] {arch}  |  {n_params:.1f}M params  |  {device}')
    return model


def read_geotiff(path: str):
    """Read GeoTIFF → (C, H, W) float32 + metadata dict."""
    import rasterio
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': 'float32',
            'driver': 'GTiff',
        }
    return data, meta


def write_geotiff(
    path: str,
    data: np.ndarray,
    meta: dict,
    scale: int = 3,
):
    """Write (C, H, W) array as GeoTIFF with updated resolution."""
    import rasterio
    from rasterio.transform import Affine

    c, h, w = data.shape
    # Update transform for higher resolution
    old_t = meta['transform']
    new_t = Affine(
        old_t.a / scale, old_t.b, old_t.c,
        old_t.d, old_t.e / scale, old_t.f,
    )

    # Use BigTIFF for outputs > 4GB
    estimated_bytes = c * h * w * 4  # float32 = 4 bytes
    use_bigtiff = estimated_bytes > 3.5e9

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': w,
        'height': h,
        'count': c,
        'crs': meta['crs'],
        'transform': new_t,
        'compress': 'deflate',
        'predictor': 2,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'BIGTIFF': 'YES' if use_bigtiff else 'IF_SAFER',
    }

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
    print(f'[Output] {path}  ({c} bands, {h}×{w}, '
          f'{os.path.getsize(path) / 1e6:.1f} MB)')


def percentile_normalize(img: np.ndarray, lo: float = 2, hi: float = 98):
    """Per-band percentile clipping to [0, 1]."""
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        band = img[c]
        valid = band[band > 0]
        if valid.size == 0:
            out[c] = 0
            continue
        vmin = np.percentile(valid, lo)
        vmax = np.percentile(valid, hi)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1
        out[c] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return out, (lo, hi)


def tiled_inference(
    model: torch.nn.Module,
    img: np.ndarray,
    scale: int = 3,
    tile_size: int = 64,
    tile_overlap: int = 8,
    device: str = 'cuda',
    normalize: bool = True,
) -> np.ndarray:
    """
    Sliding-window inference on a full scene.

    Args:
        model:        Trained SR model
        img:          (C, H, W) float32 input image
        scale:        SR scale factor
        tile_size:    LR tile size in pixels
        tile_overlap: Overlap between adjacent tiles
        device:       'cuda' or 'cpu'
        normalize:    Apply percentile normalization

    Returns:
        (C, H*scale, W*scale) float32 SR output
    """
    c, h, w = img.shape

    # Normalize
    if normalize:
        img_norm, _ = percentile_normalize(img)
    else:
        img_norm = img.copy()

    # Output buffer
    out_h, out_w = h * scale, w * scale
    output = np.zeros((c, out_h, out_w), dtype=np.float32)
    weight = np.zeros((1, out_h, out_w), dtype=np.float32)

    # Compute tile grid
    stride = tile_size - tile_overlap
    n_rows = max(1, math.ceil((h - tile_overlap) / stride))
    n_cols = max(1, math.ceil((w - tile_overlap) / stride))
    total_tiles = n_rows * n_cols

    print(f'[Tiled] {h}×{w} → {out_h}×{out_w}  |  '
          f'{n_rows}×{n_cols} = {total_tiles} tiles  |  '
          f'tile={tile_size} overlap={tile_overlap}')

    # Blending window (raised cosine)
    blend = _make_blend_window(tile_size * scale, tile_overlap * scale)

    t0 = time.time()
    tile_idx = 0

    with torch.no_grad():
        for row in range(n_rows):
            for col in range(n_cols):
                tile_idx += 1

                # LR tile coordinates
                y0 = min(row * stride, h - tile_size)
                x0 = min(col * stride, w - tile_size)
                y1 = y0 + tile_size
                x1 = x0 + tile_size

                # Extract LR tile
                tile_lr = img_norm[:, y0:y1, x0:x1]
                tile_t = torch.from_numpy(tile_lr).unsqueeze(0).to(device)

                # Forward pass
                tile_sr = model(tile_t)
                tile_sr = tile_sr.squeeze(0).cpu().numpy()

                # HR tile coordinates
                oy0, oy1 = y0 * scale, y1 * scale
                ox0, ox1 = x0 * scale, x1 * scale

                # Accumulate with blending
                output[:, oy0:oy1, ox0:ox1] += tile_sr * blend
                weight[:, oy0:oy1, ox0:ox1] += blend

                if tile_idx % 50 == 0 or tile_idx == total_tiles:
                    elapsed = time.time() - t0
                    rate = tile_idx / elapsed
                    eta = (total_tiles - tile_idx) / rate
                    print(f'  [{tile_idx}/{total_tiles}]  '
                          f'{rate:.1f} tiles/s  ETA {eta:.0f}s')

    # Normalise by weight
    output /= np.maximum(weight, 1e-8)

    elapsed = time.time() - t0
    print(f'[Done] {elapsed:.1f}s  ({total_tiles / elapsed:.1f} tiles/s)')
    return output


def _make_blend_window(size: int, overlap: int) -> np.ndarray:
    """Raised-cosine blending window for smooth tile stitching."""
    if overlap <= 0:
        return np.ones((1, size, size), dtype=np.float32)

    w = np.ones(size, dtype=np.float32)
    # Ramp at edges
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    cos_ramp = 0.5 * (1 - np.cos(np.pi * ramp))
    w[:overlap] *= cos_ramp
    w[-overlap:] *= cos_ramp[::-1]

    # 2D separable window
    window = w[np.newaxis, :] * w[:, np.newaxis]
    return window[np.newaxis]  # (1, size, size)


# ====================================================================== #
#  CLI
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description='Apply trained SR model to Sentinel-2 scenes',
    )
    parser.add_argument('--model_path', required=True,
                        help='Path to trained .pth checkpoint')
    parser.add_argument('--arch', default='MultiBandRRDBNet',
                        choices=['MultiBandRRDBNet', 'MultiBandSwinIR',
                                 'MultiBandEDSR', 'BandAdapterNet'])
    parser.add_argument('--arch_opts', default=None,
                        help='JSON string of arch kwargs (overrides defaults)')
    parser.add_argument('--input', default=None,
                        help='Single input GeoTIFF')
    parser.add_argument('--output', default=None,
                        help='Output GeoTIFF path')
    parser.add_argument('--input_dir', default=None,
                        help='Directory of input scenes (batch mode)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (batch mode)')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--tile_size', type=int, default=64,
                        help='LR tile size (default: 64)')
    parser.add_argument('--tile_overlap', type=int, default=8,
                        help='Tile overlap in LR pixels (default: 8)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip percentile normalization')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 for inference (faster, less memory)')
    args = parser.parse_args()

    # Parse arch opts
    arch_opts = json.loads(args.arch_opts) if args.arch_opts else None

    # Load model
    model = load_model(
        args.model_path, args.arch, arch_opts, args.device,
    )
    if args.fp16:
        model = model.half()

    # Collect input files
    if args.input:
        pairs = [(args.input, args.output or args.input.replace('.tif', '_sr.tif'))]
    elif args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir or 'sr_output')
        out_dir.mkdir(parents=True, exist_ok=True)
        tifs = sorted(in_dir.glob('*.tif'))
        pairs = [(str(f), str(out_dir / f'{f.stem}_sr.tif')) for f in tifs]
    else:
        parser.error('Provide --input or --input_dir')

    print(f'\n[Inference] {len(pairs)} scene(s) to process\n')

    for i, (inp, outp) in enumerate(pairs, 1):
        print(f'═══ Scene {i}/{len(pairs)}: {inp} ═══')

        # Read
        data, meta = read_geotiff(inp)
        print(f'  Input: {data.shape[0]} bands, '
              f'{data.shape[1]}×{data.shape[2]}')

        # Inference
        sr = tiled_inference(
            model, data,
            scale=args.scale,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            device=args.device,
            normalize=not args.no_normalize,
        )

        # Write with updated georeference
        os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
        write_geotiff(outp, sr, meta, scale=args.scale)
        print()

    print(f'[Complete] {len(pairs)} scene(s) processed.')


if __name__ == '__main__':
    main()
