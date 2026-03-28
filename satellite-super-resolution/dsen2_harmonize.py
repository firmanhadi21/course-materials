#!/usr/bin/env python3
"""
DSen2 Band Harmonization — Sentinel-2 L2A to 6-band 10 m GeoTIFF
==================================================================

Applies DSen2 super-resolution to harmonize all Sentinel-2 bands to a
common 10 m spatial resolution, then extracts and stacks the six bands
used by the S2→PlanetScope SR training pipeline:

    Band index  Band name        Native res.  Action
    -----------  ---------------  -----------  ----------------------------
    0            B01 (CoastalBlue)  60 m       → DSen2 to 10 m  (--run_60)
    1            B02 (Blue)         10 m       → copied as-is
    2            B03 (Green)        10 m       → copied as-is
    3            B04 (Red)          10 m       → copied as-is
    4            B05 (RedEdge)      20 m       → DSen2 to 10 m
    5            B8A (NIR)          20 m       → DSen2 to 10 m

Output: single 6-band GeoTIFF at 10 m resolution, ready for crop_pairs.py
and preprocess.py.

Prerequisites
-------------
1.  Clone and set up DSen2:
        git clone https://github.com/lanha/DSen2.git
        cd DSen2 && pip install -r requirements.txt

2.  Ensure model weights are available (DSen2 downloads them on first run,
    or place them in the DSen2 directory manually).

3.  This script additionally requires:
        pip install rasterio numpy

Usage
-----
# Single tile
python dsen2_harmonize.py \\
    --safe_dir /path/to/S2A_MSIL2A_20250815T.SAFE \\
    --output data/sentinel2_harmonized/S2_20250815_Semarang.tif \\
    --dsen2_dir /path/to/DSen2

# Batch: process all SAFE directories in a folder
python dsen2_harmonize.py \\
    --safe_dir_root /path/to/s2_downloads/ \\
    --output_dir data/sentinel2_harmonized/ \\
    --dsen2_dir /path/to/DSen2

# Skip DSen2, use bilinear resampling instead (useful for testing only)
python dsen2_harmonize.py \\
    --safe_dir /path/to/S2A.SAFE \\
    --output out.tif \\
    --no_dsen2

Notes
-----
*  This script is Step 0 in the S2→PS super-resolution pipeline, to be
   run before crop_pairs.py and preprocess.py.

*  DSen2 is released under the MIT License by Lanha et al. (2018):
   "Super-Resolution of Sentinel-2 Images: Learning a Globally
   Applicable Deep Neural Network." ISPRS JPRS 146, 305–319.
   https://github.com/lanha/DSen2

*  For the KDB paper pipeline, DSen2 was applied to Sentinel-2 L2A
   (surface reflectance) imagery from the Copernicus Data Space Ecosystem
   before passing data to the super-resolution training stage.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import glob
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ===========================================================================
# Band configuration for the 6-band pipeline
# ===========================================================================

# Sentinel-2 L2A band filenames within the SAFE directory (GRANULE subtree)
# Band name → (filename pattern, native resolution in metres)
S2_L2A_BAND_INFO = {
    'B01': ('*B01_60m.jp2', 60),
    'B02': ('*B02_10m.jp2', 10),
    'B03': ('*B03_10m.jp2', 10),
    'B04': ('*B04_10m.jp2', 10),
    'B05': ('*B05_20m.jp2', 20),
    'B8A': ('*B8A_20m.jp2', 20),
}

# Ordered list of bands in the output stack (must match training pipeline)
OUTPUT_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B8A']

# Which bands require DSen2 super-resolution
NEEDS_DSEN2 = {'B01', 'B05', 'B8A'}  # 60 m and 20 m bands


# ===========================================================================
# Utilities
# ===========================================================================

def find_safe_xml(safe_dir: str) -> Optional[str]:
    """
    Locate the top-level Sentinel-2 metadata XML inside a .SAFE directory.

    Tries MTD_MSIL2A.xml first (L2A), then MTD_MSIL1C.xml (L1C).
    """
    for name in ('MTD_MSIL2A.xml', 'MTD_MSIL1C.xml'):
        candidate = os.path.join(safe_dir, name)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: glob
    hits = glob.glob(os.path.join(safe_dir, 'MTD_MSI*.xml'))
    return hits[0] if hits else None


def find_band_file(safe_dir: str, pattern: str) -> Optional[str]:
    """
    Search inside a SAFE directory tree for a band file matching pattern.
    Returns the first match or None.
    """
    for root, _, files in os.walk(safe_dir):
        for fname in files:
            if glob.fnmatch.fnmatch(fname, pattern.lstrip('*').lstrip('/')):
                return os.path.join(root, fname)
    # Use glob for wildcards
    matches = glob.glob(os.path.join(safe_dir, '**', pattern), recursive=True)
    return matches[0] if matches else None


def read_band_at_10m(
    band_path: str,
    ref_transform: Optional['Affine'] = None,
    ref_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, 'rasterio.crs.CRS', 'Affine']:
    """
    Read a single Sentinel-2 band, resampling to 10 m if necessary.

    If ref_transform and ref_shape are provided (from the first 10m band),
    all other bands are warped to exactly the same pixel grid.

    Returns
    -------
    data       : (H, W) float32 array, DN values
    crs        : CRS
    transform  : Affine transform at 10 m
    """
    from rasterio.warp import reproject, calculate_default_transform

    with rasterio.open(band_path) as src:
        crs = src.crs
        native_res = src.res[0]

        if abs(native_res - 10.0) < 0.5 and ref_shape is None:
            # Already at 10 m — read directly
            data = src.read(1).astype(np.float32)
            return data, crs, src.transform

        if ref_transform is not None and ref_shape is not None:
            # Warp to the reference 10 m grid
            dst_data = np.zeros(ref_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
            return dst_data, crs, ref_transform
        else:
            # Compute 10 m transform and resample
            scale = native_res / 10.0
            new_h = int(src.height * scale)
            new_w = int(src.width * scale)
            new_transform = Affine(
                10.0, src.transform.b, src.transform.c,
                src.transform.d, -10.0, src.transform.f,
            )
            dst_data = np.zeros((new_h, new_w), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
            return dst_data, crs, new_transform


# ===========================================================================
# DSen2 runner
# ===========================================================================

def run_dsen2(
    xml_path: str,
    dsen2_output: str,
    dsen2_dir: str,
    run_60m: bool = True,
) -> bool:
    """
    Call DSen2's s2_tiles_supres.py via subprocess.

    Parameters
    ----------
    xml_path      : Path to S2 tile metadata XML (e.g. MTD_MSIL2A.xml)
    dsen2_output  : Destination GeoTIFF path
    dsen2_dir     : Root of DSen2 repository clone
    run_60m       : Whether to also super-resolve 60 m bands (needed for B01)

    Returns
    -------
    True on success, False on failure.
    """
    script = os.path.join(dsen2_dir, 's2_tiles_supres.py')
    if not os.path.isfile(script):
        print(f"  ERROR: DSen2 script not found at {script}")
        print(f"  Make sure --dsen2_dir points to the DSen2 repository root.")
        return False

    cmd = [
        sys.executable, script,
        xml_path,
        dsen2_output,
        '--copy_original_bands',
    ]
    if run_60m:
        cmd.append('--run_60')

    print(f"  Running DSen2: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, cwd=dsen2_dir)
    if result.returncode != 0:
        print(f"  ERROR: DSen2 exited with code {result.returncode}")
        return False
    return True


def extract_bands_from_dsen2_output(
    dsen2_tif: str,
    target_bands: List[str],
) -> Tuple[Optional[np.ndarray], Optional['rasterio.crs.CRS'], Optional['Affine']]:
    """
    Extract specific bands from a DSen2 output GeoTIFF.

    DSen2 with --copy_original_bands outputs all 10m-equivalent bands.
    The band order in the output is typically:
        [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12]
    (exact order may depend on DSen2 version; we search by known ordering)

    Parameters
    ----------
    dsen2_tif   : Path to DSen2 output GeoTIFF
    target_bands: List of band names to extract, in order (e.g. ['B01','B02',...])

    Returns
    -------
    stack  : (len(target_bands), H, W) float32 array, or None on error
    crs    : CRS
    transform : Affine transform
    """
    # DSen2 default band order (L2A, with --copy_original_bands --run_60)
    # This ordering matches the DSen2 repository implementation.
    DSEN2_BAND_ORDER = [
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
        'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
    ]

    with rasterio.open(dsen2_tif) as src:
        n_bands = src.count
        crs = src.crs
        transform = src.transform

        if n_bands != len(DSEN2_BAND_ORDER):
            print(
                f"  WARNING: DSen2 output has {n_bands} bands; "
                f"expected {len(DSEN2_BAND_ORDER)}. "
                f"Attempting to map by position."
            )
            # Fall back: use available bands in order
            available = DSEN2_BAND_ORDER[:n_bands]
        else:
            available = DSEN2_BAND_ORDER

        # Build index map: band_name → 1-based rasterio band index
        band_idx = {name: i + 1 for i, name in enumerate(available)}

        missing = [b for b in target_bands if b not in band_idx]
        if missing:
            print(f"  ERROR: Bands {missing} not found in DSen2 output.")
            return None, None, None

        stack = np.stack(
            [src.read(band_idx[b]).astype(np.float32) for b in target_bands],
            axis=0,
        )

    return stack, crs, transform


# ===========================================================================
# Fallback: bilinear resampling (no DSen2)
# ===========================================================================

def harmonize_by_resampling(
    safe_dir: str,
    target_bands: List[str],
) -> Tuple[Optional[np.ndarray], Optional['rasterio.crs.CRS'], Optional['Affine']]:
    """
    Bilinear resampling of all bands to 10 m (no DSen2).
    Suitable for quick tests; NOT recommended for training.
    """
    print("  [WARNING] Using bilinear resampling (DSen2 disabled). "
          "Quality will be lower than DSen2 super-resolution.")

    bands = []
    ref_transform = None
    ref_shape = None
    crs_out = None

    for band_name in target_bands:
        pattern, _ = S2_L2A_BAND_INFO[band_name]
        band_path = find_band_file(safe_dir, pattern)
        if band_path is None:
            print(f"  ERROR: Cannot find {band_name} in {safe_dir}")
            return None, None, None

        print(f"    {band_name}: {os.path.basename(band_path)}")
        data, crs_out, transform = read_band_at_10m(
            band_path, ref_transform, ref_shape
        )

        if ref_transform is None:
            ref_transform = transform
            ref_shape = data.shape

        bands.append(data)

    stack = np.stack(bands, axis=0)
    return stack, crs_out, ref_transform


# ===========================================================================
# Main processing function
# ===========================================================================

def harmonize_scene(
    safe_dir: str,
    output_path: str,
    dsen2_dir: Optional[str] = None,
    no_dsen2: bool = False,
    overwrite: bool = False,
) -> bool:
    """
    Harmonize a single Sentinel-2 L2A SAFE directory to a 6-band 10 m GeoTIFF.

    Parameters
    ----------
    safe_dir    : Path to the .SAFE directory
    output_path : Destination GeoTIFF path
    dsen2_dir   : Path to DSen2 repo root (required unless no_dsen2=True)
    no_dsen2    : If True, use bilinear resampling instead of DSen2
    overwrite   : Overwrite existing output file

    Returns
    -------
    True on success, False on failure.
    """
    if not HAS_RASTERIO:
        print("ERROR: rasterio is required. Install with: pip install rasterio")
        return False

    safe_name = Path(safe_dir).name
    print(f"\n{'='*60}")
    print(f"Processing: {safe_name}")
    print(f"  Output:   {output_path}")

    if os.path.isfile(output_path) and not overwrite:
        print(f"  SKIPPED (already exists; use --overwrite to force)")
        return True

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # -----------------------------------------------------------------------
    # Strategy A: DSen2 (preferred)
    # -----------------------------------------------------------------------
    if not no_dsen2:
        if dsen2_dir is None:
            print("  ERROR: --dsen2_dir is required when DSen2 is enabled.")
            print("  Use --no_dsen2 to fall back to bilinear resampling.")
            return False

        xml_path = find_safe_xml(safe_dir)
        if xml_path is None:
            print(f"  ERROR: Cannot find S2 metadata XML in {safe_dir}")
            return False

        print(f"  XML:     {os.path.basename(xml_path)}")

        with tempfile.TemporaryDirectory() as tmpdir:
            dsen2_tif = os.path.join(tmpdir, 'dsen2_output.tif')

            success = run_dsen2(xml_path, dsen2_tif, dsen2_dir, run_60m=True)
            if not success or not os.path.isfile(dsen2_tif):
                print("  DSen2 failed. Falling back to bilinear resampling.")
                stack, crs, transform = harmonize_by_resampling(
                    safe_dir, OUTPUT_BANDS
                )
            else:
                print("  Extracting bands from DSen2 output...")
                stack, crs, transform = extract_bands_from_dsen2_output(
                    dsen2_tif, OUTPUT_BANDS
                )

    # -----------------------------------------------------------------------
    # Strategy B: Bilinear resampling only
    # -----------------------------------------------------------------------
    else:
        stack, crs, transform = harmonize_by_resampling(safe_dir, OUTPUT_BANDS)

    if stack is None:
        print(f"  FAILED: could not extract bands for {safe_name}")
        return False

    # -----------------------------------------------------------------------
    # Write output GeoTIFF
    # -----------------------------------------------------------------------
    c, h, w = stack.shape
    print(f"  Writing {c}-band GeoTIFF: {h}×{w} pixels at 10 m")

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': w,
        'height': h,
        'count': c,
        'crs': crs,
        'transform': transform,
        'compress': 'deflate',
        'predictor': 2,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'BIGTIFF': 'IF_SAFER',
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(stack)
        # Write band descriptions
        for i, band_name in enumerate(OUTPUT_BANDS):
            dst.update_tags(i + 1, name=band_name)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  Bands: {', '.join(OUTPUT_BANDS)}")
    return True


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Step 0c: DSen2 band harmonization — convert S2 L2A SAFE to '
            '6-band 10 m GeoTIFF for the S2→PlanetScope SR pipeline.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single tile with DSen2
  python dsen2_harmonize.py \\
      --safe_dir /data/S2A_MSIL2A_20250815T.SAFE \\
      --output data/s2_harmonized/semarang_20250815.tif \\
      --dsen2_dir /opt/DSen2

  # Batch mode
  python dsen2_harmonize.py \\
      --safe_dir_root /data/downloads/semarang/ \\
      --output_dir data/s2_harmonized/ \\
      --dsen2_dir /opt/DSen2

  # Quick test without DSen2 (bilinear resampling — not for training)
  python dsen2_harmonize.py \\
      --safe_dir /data/S2A_MSIL2A_20250815T.SAFE \\
      --output test_output.tif \\
      --no_dsen2
""",
    )

    # Input — single or batch
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        '--safe_dir', type=str,
        help='Path to a single Sentinel-2 L2A .SAFE directory',
    )
    src_group.add_argument(
        '--safe_dir_root', type=str,
        help='Root directory containing multiple .SAFE directories (batch mode)',
    )

    # Output — single or batch
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output GeoTIFF path (single-tile mode)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='data/sentinel2_harmonized',
        help='Output directory (batch mode)',
    )

    # DSen2
    parser.add_argument(
        '--dsen2_dir', type=str, default=None,
        help=(
            'Path to DSen2 repository root '
            '(clone from https://github.com/lanha/DSen2). '
            'Required unless --no_dsen2 is set.'
        ),
    )
    parser.add_argument(
        '--no_dsen2', action='store_true',
        help=(
            'Skip DSen2 and use bilinear resampling for 20 m/60 m bands. '
            'Faster but lower quality — NOT recommended for model training.'
        ),
    )

    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing output files',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not HAS_RASTERIO:
        print("ERROR: rasterio is required.")
        print("  Install with: pip install rasterio")
        sys.exit(1)

    # Collect SAFE directories to process
    if args.safe_dir:
        safe_dirs = [args.safe_dir]
    else:
        # Find all .SAFE directories in the root
        safe_dirs = sorted(
            str(p) for p in Path(args.safe_dir_root).rglob('*.SAFE')
            if p.is_dir()
        )
        # Also accept non-.SAFE folder names that contain an MTD_MSI*.xml
        extra = []
        for entry in sorted(Path(args.safe_dir_root).iterdir()):
            if entry.is_dir() and not str(entry).endswith('.SAFE'):
                if list(entry.glob('MTD_MSI*.xml')):
                    extra.append(str(entry))
        safe_dirs = sorted(set(safe_dirs + extra))

        if not safe_dirs:
            print(f"No .SAFE directories found in {args.safe_dir_root}")
            sys.exit(1)

    print(f"Found {len(safe_dirs)} Sentinel-2 tile(s) to process.")

    # Build output path(s)
    if args.safe_dir and args.output:
        output_paths = [args.output]
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_paths = []
        for safe in safe_dirs:
            stem = Path(safe).stem
            output_paths.append(os.path.join(args.output_dir, f'{stem}_harmonized.tif'))

    # Validate DSen2 availability
    if not args.no_dsen2:
        if args.dsen2_dir is None:
            print("ERROR: --dsen2_dir is required when using DSen2.")
            print("  Clone DSen2 with:")
            print("    git clone https://github.com/lanha/DSen2.git")
            print("  Then pass: --dsen2_dir /path/to/DSen2")
            print()
            print("  Alternatively, use --no_dsen2 for bilinear resampling (testing only).")
            sys.exit(1)
        if not os.path.isdir(args.dsen2_dir):
            print(f"ERROR: DSen2 directory not found: {args.dsen2_dir}")
            sys.exit(1)

    # Process
    n_ok = 0
    n_fail = 0
    for safe_dir, output_path in zip(safe_dirs, output_paths):
        ok = harmonize_scene(
            safe_dir=safe_dir,
            output_path=output_path,
            dsen2_dir=args.dsen2_dir,
            no_dsen2=args.no_dsen2,
            overwrite=args.overwrite,
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"Completed: {n_ok} succeeded, {n_fail} failed")
    print()
    if n_ok > 0:
        out_display = args.output or args.output_dir
        print(f"Output(s): {out_display}")
        print()
        print("Next step: crop_pairs.py")
        print("  python crop_pairs.py \\")
        print(f"      --s2_dir {args.output or args.output_dir} \\")
        print("      --ps_dir data/planetscope_raw \\")
        print("      --output_dir data/paired_scenes")


if __name__ == '__main__':
    main()
