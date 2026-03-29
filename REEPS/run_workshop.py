#!/usr/bin/env python3
"""
Workshop Pipeline Runner
========================
Runs all stages of the biodiversity analysis workshop pipeline sequentially.

  Step 1: Generate synthetic data (from Workshop_Biodiversity_Database.xlsx)
  Step 2: Build Master GeoPackage
  Step 3: Build Grid Analyses
  Step 4: Generate interactive HTML maps + validation CSVs
  Step 5: Generate static matplotlib figures

Usage:
    python run_workshop.py
"""
import subprocess
import sys
import os
import time

BASE = '/Users/macbook/Dropbox/Works/Cisokan/2026/Feb26/data/biodiv/workshop'

STEPS = [
    {
        'num': 1,
        'name': 'Generate synthetic data',
        'script': 'generate_synthetic_data.py',
        'desc': 'Creates Workshop_Biodiversity_Database.xlsx and workshop_synthetic_records.csv',
    },
    {
        'num': 2,
        'name': 'Build Master GeoPackage',
        'script': 'workshop_stage1_master_gpkg.py',
        'desc': 'Reads Excel, harmonizes species, assigns H3, writes Workshop_Master_Database.gpkg',
    },
    {
        'num': 3,
        'name': 'Build Grid Analyses',
        'script': 'workshop_stage2_grid_analyses.py',
        'desc': 'Computes diversity, priority, temporal, corridor, and survey gap layers',
    },
    {
        'num': 4,
        'name': 'Generate interactive maps + validation',
        'script': 'workshop_stage3_maps_validation.py',
        'desc': 'Builds HTML maps (diversity, priority, connectivity, corridor, temporal, H3, co-occurrence) and validation CSVs',
    },
    {
        'num': 5,
        'name': 'Generate static figures',
        'script': 'workshop_stage4_figures.py',
        'desc': 'Produces species richness, Shannon entropy, and priority tier PNGs at 300 DPI',
    },
]


def run_step(step):
    """Run a single pipeline step, return True on success."""
    script_path = os.path.join(BASE, step['script'])
    if not os.path.exists(script_path):
        print(f"  ERROR: Script not found: {script_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Step {step['num']}: {step['name']}")
    print(f"  {step['desc']}")
    print(f"  Script: {step['script']}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE,
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\n  Completed in {elapsed:.1f}s")
    return True


def main():
    print("=" * 60)
    print("  WORKSHOP BIODIVERSITY ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"\nWorking directory: {BASE}")
    print(f"Steps to run: {len(STEPS)}\n")

    t_start = time.time()
    failed = []

    for step in STEPS:
        ok = run_step(step)
        if not ok:
            failed.append(step['num'])
            print(f"\n  Step {step['num']} failed. Stopping pipeline.")
            break

    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")

    if failed:
        print(f"  Status: FAILED at step(s) {failed}")
        sys.exit(1)
    else:
        print("  Status: ALL STEPS COMPLETED SUCCESSFULLY")
        print(f"\nOutputs in {BASE}/:")
        print("  - Workshop_Master_Database.gpkg")
        print("  - Workshop_GridAnalyses.gpkg")
        print("  - Workshop_*.html  (interactive maps)")
        print("  - figures/*.png    (static figures)")
        print("  - *.csv            (validation tables)")


if __name__ == '__main__':
    main()
