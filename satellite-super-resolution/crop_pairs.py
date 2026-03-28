#!/usr/bin/env python3
"""
Crop Sentinel-2 Scenes to Match PlanetScope Extents
=====================================================

PlanetScope scenes are much smaller than Sentinel-2 scenes.
This script crops Sentinel-2 to match each PlanetScope scene's
geographic extent, creating properly paired files for training.

Before:
    Sentinel-2:   110km × 110km (one large scene)
    PlanetScope:  24km × 8km (many small scenes)

After:
    Sentinel-2:   24km × 8km (cropped to match each PS scene)
    PlanetScope:  24km × 8km

Usage:
    # Basic usage
    python crop_pairs.py \
        --s2_input data/sentinel2_full/S2_20230615.tif \
        --ps_dir data/planetscope_raw \
        --output_dir data/paired_scenes

    # Multiple S2 scenes (finds best date match)
    python crop_pairs.py \
        --s2_dir data/sentinel2_full \
        --ps_dir data/planetscope_raw \
        --output_dir data/paired_scenes \
        --max_days 7

    # Specify bands to extract from S2
    python crop_pairs.py \
        --s2_input data/sentinel2_full/S2_20230615.tif \
        --ps_dir data/planetscope_raw \
        --output_dir data/paired_scenes \
        --s2_bands 2,3,4,8

Output structure:
    data/paired_scenes/
    ├── sentinel2/
    │   ├── pair_001.tif
    │   ├── pair_002.tif
    │   └── ...
    └── planetscope/
        ├── pair_001.tif
        ├── pair_002.tif
        └── ...

Then run the main preprocessing:
    python preprocess.py \
        --s2_dir data/paired_scenes/sentinel2 \
        --ps_dir data/paired_scenes/planetscope \
        --output_dir data/patches \
        --coregister
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.windows import from_bounds
    from rasterio import Affine
except ImportError:
    print("ERROR: rasterio is required. Install with: pip install rasterio")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop Sentinel-2 scenes to match PlanetScope extents'
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--s2_input', type=str,
        help='Single Sentinel-2 GeoTIFF file'
    )
    input_group.add_argument(
        '--s2_dir', type=str,
        help='Directory with multiple Sentinel-2 scenes (for date matching)'
    )

    parser.add_argument(
        '--ps_dir', type=str, required=True,
        help='Directory containing PlanetScope scenes'
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/paired_scenes',
        help='Output directory for paired scenes'
    )

    # Processing options
    parser.add_argument(
        '--s2_bands', type=str, default=None,
        help='Comma-separated band indices to extract from S2 (e.g., "2,3,4,8" for B02,B03,B04,B08). '
             'Default: all bands'
    )
    parser.add_argument(
        '--max_days', type=int, default=7,
        help='Maximum days difference for date matching (default: 7)'
    )
    parser.add_argument(
        '--buffer', type=float, default=0,
        help='Buffer in meters to add around PS bounds (default: 0)'
    )
    parser.add_argument(
        '--min_overlap', type=float, default=0.9,
        help='Minimum overlap fraction required (default: 0.9)'
    )
    parser.add_argument(
        '--nodata', type=float, default=0,
        help='NoData value for output (default: 0)'
    )

    return parser.parse_args()


def extract_date(filename: str) -> Optional[datetime]:
    """Extract date from filename using common patterns."""
    patterns = [
        r'(\d{4})(\d{2})(\d{2})',      # YYYYMMDD
        r'(\d{4})-(\d{2})-(\d{2})',    # YYYY-MM-DD
        r'(\d{4})_(\d{2})_(\d{2})',    # YYYY_MM_DD
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3))
                )
            except ValueError:
                continue
    return None


def get_raster_bounds(path: str) -> Tuple[float, float, float, float]:
    """Get geographic bounds of a raster."""
    with rasterio.open(path) as src:
        return src.bounds


def get_raster_crs(path: str):
    """Get CRS of a raster."""
    with rasterio.open(path) as src:
        return src.crs


def check_overlap(bounds1, bounds2) -> float:
    """Calculate overlap fraction between two bounding boxes."""
    # Intersection
    xmin = max(bounds1.left, bounds2.left)
    ymin = max(bounds1.bottom, bounds2.bottom)
    xmax = min(bounds1.right, bounds2.right)
    ymax = min(bounds1.top, bounds2.top)

    if xmin >= xmax or ymin >= ymax:
        return 0.0

    intersection_area = (xmax - xmin) * (ymax - ymin)
    bounds2_area = (bounds2.right - bounds2.left) * (bounds2.top - bounds2.bottom)

    return intersection_area / bounds2_area


def find_best_s2_match(
    ps_path: str,
    s2_files: List[str],
    max_days: int
) -> Optional[str]:
    """Find the S2 scene with closest date that overlaps with PS scene."""
    ps_date = extract_date(Path(ps_path).stem)
    ps_bounds = get_raster_bounds(ps_path)
    ps_crs = get_raster_crs(ps_path)

    best_match = None
    best_days_diff = float('inf')

    for s2_path in s2_files:
        # Check overlap
        s2_bounds = get_raster_bounds(s2_path)
        s2_crs = get_raster_crs(s2_path)

        # Simple bounds check (assumes same CRS or close enough)
        with rasterio.open(s2_path) as src:
            overlap = check_overlap(src.bounds, ps_bounds)
            if overlap < 0.5:  # Need at least 50% overlap
                continue

        # Check date if available
        if ps_date:
            s2_date = extract_date(Path(s2_path).stem)
            if s2_date:
                days_diff = abs((s2_date - ps_date).days)
                if days_diff <= max_days and days_diff < best_days_diff:
                    best_days_diff = days_diff
                    best_match = s2_path
            else:
                # No date in S2 filename, use if no better match
                if best_match is None:
                    best_match = s2_path
        else:
            # No date in PS filename, just check overlap
            if best_match is None:
                best_match = s2_path

    return best_match


def crop_s2_to_ps(
    s2_path: str,
    ps_path: str,
    output_path: str,
    bands: Optional[List[int]] = None,
    buffer: float = 0,
    nodata: float = 0,
) -> bool:
    """
    Crop Sentinel-2 to match PlanetScope extent.

    Returns True if successful, False if insufficient overlap.
    """
    with rasterio.open(ps_path) as ps_src:
        ps_bounds = ps_src.bounds
        ps_crs = ps_src.crs

        # Add buffer if specified
        if buffer > 0:
            ps_bounds = rasterio.coords.BoundingBox(
                ps_bounds.left - buffer,
                ps_bounds.bottom - buffer,
                ps_bounds.right + buffer,
                ps_bounds.top + buffer,
            )

    with rasterio.open(s2_path) as s2_src:
        s2_crs = s2_src.crs
        s2_bounds = s2_src.bounds

        # Check if PS is within S2 bounds
        overlap = check_overlap(s2_bounds, ps_bounds)
        if overlap < 0.5:
            return False

        # Determine which bands to read
        if bands is None:
            bands = list(range(1, s2_src.count + 1))

        # If CRS doesn't match, we need to reproject
        if ps_crs != s2_crs:
            # Calculate transform for reprojection
            transform, width, height = calculate_default_transform(
                s2_crs, ps_crs,
                s2_src.width, s2_src.height,
                *s2_src.bounds,
            )

            # Crop window in PS CRS
            window_transform, window_width, window_height = calculate_default_transform(
                ps_crs, ps_crs,
                int((ps_bounds.right - ps_bounds.left) / s2_src.res[0]),
                int((ps_bounds.top - ps_bounds.bottom) / s2_src.res[1]),
                *ps_bounds,
            )
        else:
            # Same CRS - simple window extraction
            window = from_bounds(*ps_bounds, s2_src.transform)

            # Handle partial overlap
            window_clipped = window.intersection(
                rasterio.windows.Window(0, 0, s2_src.width, s2_src.height)
            )

            if window_clipped.width < 10 or window_clipped.height < 10:
                return False

            # Read the data
            data = s2_src.read(bands, window=window_clipped)

            # Calculate new transform
            new_transform = s2_src.window_transform(window_clipped)

            # Write output
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            profile = s2_src.profile.copy()
            profile.update(
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=len(bands),
                transform=new_transform,
                nodata=nodata,
                compress='deflate',
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)

            return True

    return False


def copy_ps_scene(
    ps_path: str,
    output_path: str,
    nodata: float = 0,
):
    """Copy PlanetScope scene to output directory."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with rasterio.open(ps_path) as src:
        profile = src.profile.copy()
        profile.update(
            compress='deflate',
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(src.read())


def main():
    args = parse_args()

    # Parse bands if specified
    bands = None
    if args.s2_bands:
        bands = [int(b.strip()) for b in args.s2_bands.split(',')]
        print(f"Extracting S2 bands: {bands}")

    # Collect S2 files
    if args.s2_input:
        s2_files = [args.s2_input]
    else:
        s2_dir = Path(args.s2_dir)
        s2_files = sorted(
            list(s2_dir.glob('*.tif')) + list(s2_dir.glob('*.tiff'))
        )
        s2_files = [str(f) for f in s2_files]

    if not s2_files:
        print("ERROR: No Sentinel-2 files found")
        sys.exit(1)

    print(f"Found {len(s2_files)} Sentinel-2 scene(s)")

    # Collect PS files
    ps_dir = Path(args.ps_dir)
    ps_files = sorted(
        list(ps_dir.glob('*.tif')) + list(ps_dir.glob('*.tiff'))
    )

    if not ps_files:
        print(f"ERROR: No PlanetScope files found in {args.ps_dir}")
        sys.exit(1)

    print(f"Found {len(ps_files)} PlanetScope scene(s)")

    # Create output directories
    s2_out_dir = Path(args.output_dir) / 'sentinel2'
    ps_out_dir = Path(args.output_dir) / 'planetscope'
    s2_out_dir.mkdir(parents=True, exist_ok=True)
    ps_out_dir.mkdir(parents=True, exist_ok=True)

    # Process each PS scene
    print("\n" + "=" * 60)
    print("Processing PlanetScope scenes...")
    print("=" * 60)

    success_count = 0
    skip_count = 0

    for i, ps_path in enumerate(ps_files):
        ps_name = ps_path.stem
        print(f"\n[{i+1}/{len(ps_files)}] {ps_name}")

        # Find matching S2 scene
        if len(s2_files) == 1:
            s2_path = s2_files[0]
        else:
            s2_path = find_best_s2_match(
                str(ps_path), s2_files, args.max_days
            )
            if s2_path:
                s2_date = extract_date(Path(s2_path).stem)
                ps_date = extract_date(ps_name)
                if s2_date and ps_date:
                    days_diff = abs((s2_date - ps_date).days)
                    print(f"  Matched S2: {Path(s2_path).stem} ({days_diff} days diff)")

        if s2_path is None:
            print(f"  ✗ No matching S2 scene found")
            skip_count += 1
            continue

        # Generate output filenames
        pair_id = f"pair_{success_count + 1:04d}"
        s2_out_path = s2_out_dir / f"{pair_id}.tif"
        ps_out_path = ps_out_dir / f"{pair_id}.tif"

        # Crop S2 to PS bounds
        try:
            success = crop_s2_to_ps(
                s2_path,
                str(ps_path),
                str(s2_out_path),
                bands=bands,
                buffer=args.buffer,
                nodata=args.nodata,
            )

            if success:
                # Copy PS scene
                copy_ps_scene(str(ps_path), str(ps_out_path), args.nodata)

                # Get dimensions for info
                with rasterio.open(s2_out_path) as src:
                    s2_shape = (src.count, src.height, src.width)
                with rasterio.open(ps_out_path) as src:
                    ps_shape = (src.count, src.height, src.width)

                print(f"  ✓ Created {pair_id}")
                print(f"    S2: {s2_shape[1]}×{s2_shape[2]} ({s2_shape[0]} bands)")
                print(f"    PS: {ps_shape[1]}×{ps_shape[2]} ({ps_shape[0]} bands)")
                success_count += 1
            else:
                print(f"  ✗ Insufficient overlap with S2")
                skip_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            skip_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pairs created: {success_count}")
    print(f"  Skipped:       {skip_count}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  {s2_out_dir}/")
    print(f"  {ps_out_dir}/")

    if success_count > 0:
        print(f"\nNext step - run preprocessing:")
        print(f"  python preprocess.py \\")
        print(f"      --s2_dir {s2_out_dir} \\")
        print(f"      --ps_dir {ps_out_dir} \\")
        print(f"      --output_dir data/patches \\")
        print(f"      --coregister")


if __name__ == '__main__':
    main()
