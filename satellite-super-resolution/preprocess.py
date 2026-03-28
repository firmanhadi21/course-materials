"""
Preprocessing Pipeline: Sentinel-2 × PlanetScope Super-Resolution
==================================================================

Run this script to prepare your data for SR training:
  1. Co-register PlanetScope to Sentinel-2 (AROSICS)
  2. Band-match and stack
  3. Extract and save aligned patch pairs
  4. Split into train/val/test

Usage:
    python preprocess.py \
        --s2_dir data/sentinel2_scenes \
        --ps_dir data/planetscope_scenes \
        --output_dir data/patches \
        --patch_size 64 \
        --scale_factor 3 \
        --stride 32 \
        --bands B02,B03,B04,B08 \
        --coregister
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dataset import (
    CoRegistrator,
    PatchExtractor,
    RadiometricNormalizer,
    read_s2_bands,
    read_ps_scene,
    align_hr_to_lr_grid,
    PS_SUPERDOVE_BANDS,
    PS_4BAND,
    HAS_AROSICS,
    HAS_RASTERIO,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess Sentinel-2 / PlanetScope data for SR training'
    )

    # Input/Output
    parser.add_argument(
        '--s2_dir', type=str, required=True,
        help='Directory containing Sentinel-2 scenes (.tif files)'
    )
    parser.add_argument(
        '--ps_dir', type=str, required=True,
        help='Directory containing PlanetScope scenes (.tif files)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/patches',
        help='Output directory for extracted patches'
    )

    # Patch extraction
    parser.add_argument(
        '--patch_size', type=int, default=64,
        help='LR patch size in pixels (HR = LR × scale_factor)'
    )
    parser.add_argument(
        '--scale_factor', type=int, default=3,
        help='Super-resolution scale factor'
    )
    parser.add_argument(
        '--stride', type=int, default=32,
        help='Stride between patches (in LR pixels)'
    )
    parser.add_argument(
        '--min_valid', type=float, default=0.9,
        help='Minimum fraction of valid pixels per patch'
    )

    # Band selection
    parser.add_argument(
        '--bands', type=str, default='B02,B03,B04,B08',
        help='Comma-separated Sentinel-2 bands to use'
    )
    parser.add_argument(
        '--ps_type', type=str, default='4band',
        choices=['4band', '8band'],
        help='PlanetScope sensor type'
    )

    # Co-registration
    parser.add_argument(
        '--skip-coregister', action='store_true',
        help='Skip AROSICS co-registration (not recommended, only use if already aligned)'
    )
    parser.add_argument(
        '--coreg_method', type=str, default='local',
        choices=['local', 'global'],
        help='Co-registration method'
    )
    parser.add_argument(
        '--max_shift', type=int, default=10,
        help='Maximum allowed shift in pixels for co-registration'
    )
    parser.add_argument(
        '--coreg_ref_band', type=int, default=4,
        help='Reference (S2) band for co-registration matching (1-indexed, default=4 for Red)'
    )
    parser.add_argument(
        '--coreg_tgt_band', type=int, default=4,
        help='Target (PS) band for co-registration matching (1-indexed, default=4 for Red)'
    )

    # Normalization
    parser.add_argument(
        '--norm_method', type=str, default='percentile',
        choices=['percentile', 'minmax', 'zscore', 'none'],
        help='Radiometric normalization method'
    )

    # Data split
    parser.add_argument(
        '--val_split', type=float, default=0.1,
        help='Fraction of scenes for validation'
    )
    parser.add_argument(
        '--test_split', type=float, default=0.1,
        help='Fraction of scenes for testing'
    )

    # Save format
    parser.add_argument(
        '--format', type=str, default='npy',
        choices=['npy', 'tif'],
        help='Output patch format'
    )
    parser.add_argument(
        '--save_normalized', action='store_true',
        help='Save patches after normalization (otherwise raw)'
    )

    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    return parser.parse_args()


def find_scene_pairs(
    s2_dir: str,
    ps_dir: str,
) -> List[Tuple[str, str]]:
    """
    Find matching Sentinel-2 / PlanetScope scene pairs.

    Matching strategy (in order of priority):
      1. Exact filename match (stem)
      2. Date-based matching (extract date from filename)
      3. Sorted order fallback
    """
    s2_path = Path(s2_dir)
    ps_path = Path(ps_dir)

    s2_files = sorted(
        list(s2_path.glob('*.tif')) + list(s2_path.glob('*.tiff'))
    )
    ps_files = sorted(
        list(ps_path.glob('*.tif')) + list(ps_path.glob('*.tiff'))
    )

    # Also consider directories (per-band Sentinel-2)
    s2_dirs = sorted([d for d in s2_path.iterdir() if d.is_dir()])
    if s2_dirs and not s2_files:
        s2_files = s2_dirs

    # Strategy 1: Match by exact stem
    s2_stems = {f.stem: f for f in s2_files}
    ps_stems = {f.stem: f for f in ps_files}
    common_stems = set(s2_stems.keys()) & set(ps_stems.keys())

    if common_stems:
        pairs = [
            (str(s2_stems[s]), str(ps_stems[s]))
            for s in sorted(common_stems)
        ]
        print(f"Matched {len(pairs)} pairs by filename")
        return pairs

    # Strategy 2: Match by date pattern in filename
    import re

    def extract_date(filename: str) -> str:
        """Extract date string (YYYYMMDD) from filename."""
        patterns = [
            r'(\d{8})',                    # YYYYMMDD
            r'(\d{4}-\d{2}-\d{2})',        # YYYY-MM-DD
            r'(\d{4}_\d{2}_\d{2})',        # YYYY_MM_DD
        ]
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1).replace('-', '').replace('_', '')
        return ''

    s2_by_date: Dict[str, Path] = {}
    for f in s2_files:
        d = extract_date(f.stem)
        if d:
            s2_by_date[d] = f

    ps_by_date: Dict[str, Path] = {}
    for f in ps_files:
        d = extract_date(f.stem)
        if d:
            ps_by_date[d] = f

    common_dates = set(s2_by_date.keys()) & set(ps_by_date.keys())
    if common_dates:
        pairs = [
            (str(s2_by_date[d]), str(ps_by_date[d]))
            for d in sorted(common_dates)
        ]
        print(f"Matched {len(pairs)} pairs by date")
        return pairs

    # Strategy 3: Sorted-order fallback
    n_pairs = min(len(s2_files), len(ps_files))
    pairs = [
        (str(s2_files[i]), str(ps_files[i]))
        for i in range(n_pairs)
    ]
    print(
        f"Matched {len(pairs)} pairs by sorted order "
        f"(no filename/date match found)"
    )
    return pairs


def compute_global_stats(
    scene_pairs: List[Tuple[str, str]],
    bands_s2: List[str],
    max_scenes: int = 20,
) -> Dict:
    """
    Compute global per-band statistics across scenes.
    Useful for z-score normalization.
    """
    print("Computing global band statistics...")

    s2_sums = None
    s2_sq_sums = None
    s2_counts = None

    ps_sums = None
    ps_sq_sums = None
    ps_counts = None

    n = min(len(scene_pairs), max_scenes)

    for i, (s2_path, ps_path) in enumerate(scene_pairs[:n]):
        print(f"  Scanning scene {i + 1}/{n}...")

        try:
            s2_img, _ = read_s2_bands(s2_path, bands=bands_s2)
            ps_img, _ = read_ps_scene(ps_path)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Sentinel-2 stats
        for c in range(s2_img.shape[0]):
            band = s2_img[c]
            valid = band[band > 0].astype(np.float64)
            if valid.size == 0:
                continue
            if s2_sums is None:
                nc = s2_img.shape[0]
                s2_sums = np.zeros(nc)
                s2_sq_sums = np.zeros(nc)
                s2_counts = np.zeros(nc)
            s2_sums[c] += valid.sum()
            s2_sq_sums[c] += (valid ** 2).sum()
            s2_counts[c] += valid.size

        # PlanetScope stats
        for c in range(ps_img.shape[0]):
            band = ps_img[c]
            valid = band[band > 0].astype(np.float64)
            if valid.size == 0:
                continue
            if ps_sums is None:
                nc = ps_img.shape[0]
                ps_sums = np.zeros(nc)
                ps_sq_sums = np.zeros(nc)
                ps_counts = np.zeros(nc)
            ps_sums[c] += valid.sum()
            ps_sq_sums[c] += (valid ** 2).sum()
            ps_counts[c] += valid.size

    stats = {}

    if s2_sums is not None:
        s2_mean = s2_sums / np.maximum(s2_counts, 1)
        s2_var = s2_sq_sums / np.maximum(s2_counts, 1) - s2_mean ** 2
        s2_std = np.sqrt(np.maximum(s2_var, 0))
        stats['sentinel2'] = {
            'mean': s2_mean.tolist(),
            'std': s2_std.tolist(),
        }

    if ps_sums is not None:
        ps_mean = ps_sums / np.maximum(ps_counts, 1)
        ps_var = ps_sq_sums / np.maximum(ps_counts, 1) - ps_mean ** 2
        ps_std = np.sqrt(np.maximum(ps_var, 0))
        stats['planetscope'] = {
            'mean': ps_mean.tolist(),
            'std': ps_std.tolist(),
        }

    return stats


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    bands_s2 = [b.strip() for b in args.bands.split(',')]

    print("=" * 60)
    print("Sentinel-2 × PlanetScope SR Data Preparation")
    print("=" * 60)
    print(f"  S2 directory:    {args.s2_dir}")
    print(f"  PS directory:    {args.ps_dir}")
    print(f"  Output:          {args.output_dir}")
    print(f"  Patch size (LR): {args.patch_size}")
    print(f"  Scale factor:    {args.scale_factor}")
    print(f"  Stride:          {args.stride}")
    print(f"  Bands (S2):      {bands_s2}")
    print(f"  PS type:         {args.ps_type}")
    print(f"  Co-register:     {not args.skip_coregister}")
    print(f"  Normalization:   {args.norm_method}")
    print()

    # ---------------------------------------------------------------
    # Step 1: Find scene pairs
    # ---------------------------------------------------------------
    print("[1/5] Finding scene pairs...")
    scene_pairs = find_scene_pairs(args.s2_dir, args.ps_dir)

    if not scene_pairs:
        print("ERROR: No scene pairs found. Check directories.")
        return

    # ---------------------------------------------------------------
    # Step 2: Co-registration (optional)
    # ---------------------------------------------------------------
    if not args.skip_coregister:
        print(f"\n[2/5] Co-registering ({args.coreg_method})...")

        if not HAS_AROSICS:
            print(
                "ERROR: AROSICS is required for co-registration."
            )
            print("  Install with: pip install arosics")
            print("  Or use --skip-coregister if images are already aligned (not recommended)")
            return
        else:
            coreg_dir = os.path.join(args.output_dir, 'coregistered_ps')
            os.makedirs(coreg_dir, exist_ok=True)

            coreg = CoRegistrator(
                method=args.coreg_method,
                max_shift=args.max_shift,
                ref_band=args.coreg_ref_band,
                tgt_band=args.coreg_tgt_band,
            )

            new_pairs = []
            for s2_path, ps_path in scene_pairs:
                try:
                    out_path = coreg.coregister(
                        reference_path=s2_path,
                        target_path=ps_path,
                        output_path=os.path.join(
                            coreg_dir,
                            Path(ps_path).stem + '_coreg.tif',
                        ),
                    )
                    new_pairs.append((s2_path, out_path))
                except Exception as e:
                    print(f"  ✗ Failed on {Path(ps_path).name}: {e}")
                    # Keep original if co-registration fails
                    new_pairs.append((s2_path, ps_path))

            scene_pairs = new_pairs
    else:
        print("\n[2/5] Co-registration skipped (WARNING: not recommended for training)")

    # ---------------------------------------------------------------
    # Step 3: Compute statistics (if using zscore)
    # ---------------------------------------------------------------
    global_stats = None
    if args.norm_method == 'zscore':
        print("\n[3/5] Computing global statistics...")
        global_stats = compute_global_stats(scene_pairs, bands_s2)
        # Save stats
        stats_path = os.path.join(args.output_dir, 'global_stats.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(global_stats, f, indent=2)
        print(f"  Saved to {stats_path}")
    else:
        print("\n[3/5] Global statistics not needed for this norm method")

    # ---------------------------------------------------------------
    # Step 4: Split scenes into train/val/test
    # ---------------------------------------------------------------
    print("\n[4/5] Splitting scenes...")
    random.shuffle(scene_pairs)

    n = len(scene_pairs)
    n_test = max(1, int(n * args.test_split))
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val - n_test

    splits = {
        'train': scene_pairs[:n_train],
        'val': scene_pairs[n_train : n_train + n_val],
        'test': scene_pairs[n_train + n_val :],
    }

    for split_name, pairs in splits.items():
        print(f"  {split_name}: {len(pairs)} scenes")

    # ---------------------------------------------------------------
    # Step 5: Extract patches
    # ---------------------------------------------------------------
    print("\n[5/5] Extracting patches...")

    extractor = PatchExtractor(
        patch_size_lr=args.patch_size,
        scale_factor=args.scale_factor,
        stride=args.stride,
        min_valid_fraction=args.min_valid,
    )

    normalizer = None
    if args.save_normalized and args.norm_method != 'none':
        normalizer = RadiometricNormalizer(
            method=args.norm_method,
            global_stats=global_stats,
        )

    # PS band names for matching
    if args.ps_type == '4band':
        ps_bands = ['blue', 'green', 'red', 'nir']
        ps_band_map = PS_4BAND
    else:
        # SuperDove 8-band: select 6 bands that match S2
        # Excludes: green_i (531nm) and yellow (610nm) - no S2 equivalent
        ps_bands = ['coastal_blue', 'blue', 'green', 'red', 'red_edge', 'nir']
        ps_band_map = PS_SUPERDOVE_BANDS

    # Compute 0-indexed band indices for align_hr_to_lr_grid
    ps_band_indices = [ps_band_map[b]['index'] for b in ps_bands]

    total_patches = 0
    metadata = {'splits': {}, 'config': vars(args)}

    for split_name, pairs in splits.items():
        lr_out = os.path.join(args.output_dir, split_name, 'sentinel2')
        hr_out = os.path.join(args.output_dir, split_name, 'planetscope')
        os.makedirs(lr_out, exist_ok=True)
        os.makedirs(hr_out, exist_ok=True)

        split_count = 0

        for i, (s2_path, ps_path) in enumerate(pairs):
            scene_name = Path(s2_path).stem
            print(
                f"  [{split_name}] Scene {i + 1}/{len(pairs)}: "
                f"{scene_name}"
            )

            try:
                # Load S2 scene
                s2_img, _ = read_s2_bands(
                    s2_path, bands=bands_s2, target_resolution=10.0
                )
                # Resample PS onto S2-aligned grid (scale_factor × S2 shape)
                # This ensures exact pixel correspondence for patch extraction
                ps_img, _ = align_hr_to_lr_grid(
                    hr_path=ps_path,
                    lr_path=s2_path,
                    scale_factor=args.scale_factor,
                    hr_band_indices=ps_band_indices,
                )

                # Optional: normalize before saving
                if normalizer:
                    s2_img = normalizer.normalize(
                        s2_img, sensor='sentinel2'
                    )
                    ps_img = normalizer.normalize(
                        ps_img, sensor='planetscope'
                    )

                # Extract and save patches
                n_patches = extractor.extract_and_save(
                    lr_image=s2_img,
                    hr_image=ps_img,
                    lr_output_dir=lr_out,
                    hr_output_dir=hr_out,
                    scene_id=scene_name,
                    save_format=args.format,
                )

                split_count += n_patches
                print(f"    → {n_patches} patches extracted")

            except Exception as e:
                print(f"    ✗ Error: {e}")

        total_patches += split_count
        metadata['splits'][split_name] = {
            'n_scenes': len(pairs),
            'n_patches': split_count,
        }

        print(
            f"  [{split_name}] Total: {split_count} patches"
        )

    # Save metadata
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done! Total patches extracted: {total_patches}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata saved to: {meta_path}")
    print()
    print("Directory structure:")
    print(f"  {args.output_dir}/")
    print(f"    train/sentinel2/     (LR patches)")
    print(f"    train/planetscope/   (HR patches)")
    print(f"    val/sentinel2/")
    print(f"    val/planetscope/")
    print(f"    test/sentinel2/")
    print(f"    test/planetscope/")
    print(f"    metadata.json")
    if global_stats:
        print(f"    global_stats.json")
    print()
    print("Next steps:")
    print("  Load with PyTorch:")
    print("    from dataset import create_dataloaders")
    print("    train_dl, val_dl, test_dl = create_dataloaders(")
    print(f"        lr_dir='{args.output_dir}/train/sentinel2',")
    print(f"        hr_dir='{args.output_dir}/train/planetscope',")
    print("        batch_size=16,")
    print(f"        scale_factor={args.scale_factor},")
    print("    )")
    print("=" * 60)


if __name__ == '__main__':
    main()
