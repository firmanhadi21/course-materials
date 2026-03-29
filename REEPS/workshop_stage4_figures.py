"""
Workshop Stage 4: Generate static matplotlib figures.
Uses downsampled Sentinel-2 basemap for publication-quality maps.
Produces: species richness, Shannon entropy, and priority tier maps as PNG at 300 DPI.
"""
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import array_bounds, from_bounds as transform_from_bounds
import os

BASE = '/Users/macbook/Dropbox/Works/Cisokan/2026/Feb26/data/biodiv/workshop'
GRID_GPKG = os.path.join(BASE, 'Workshop_GridAnalyses.gpkg')
MASTER_GPKG = os.path.join(BASE, 'Workshop_Master_Database.gpkg')
S2_TIF = os.path.join(BASE, 'Sentinel2_S2DR3_30Sep24_RGB_lowres.tif')
FIG_DIR = os.path.join(BASE, 'figures')

# Load Sentinel-2 basemap once
S2_RGBA = None
S2_EXTENT = None
if os.path.exists(S2_TIF):
    with rasterio.open(S2_TIF) as src:
        t, w, h = calculate_default_transform(src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        bounds = array_bounds(h, w, t)
        dst_t = transform_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
        rgb = np.zeros((3, h, w), dtype=np.uint8)
        for b in range(1, 4):
            reproject(source=rasterio.band(src, b), destination=rgb[b-1],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_t, dst_crs='EPSG:4326',
                      resampling=Resampling.bilinear)
        S2_RGBA = np.zeros((h, w, 4), dtype=np.float32)
        S2_RGBA[:,:,0] = rgb[0] / 255.0
        S2_RGBA[:,:,1] = rgb[1] / 255.0
        S2_RGBA[:,:,2] = rgb[2] / 255.0
        S2_RGBA[:,:,3] = 1.0
        S2_EXTENT = (bounds[0], bounds[2], bounds[1], bounds[3])
    print(f"  Basemap loaded: {w}x{h} px")
else:
    print(f"  Basemap not found at {S2_TIF}, using white background")


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_base(ax, aoi, h3_all):
    """Draw Sentinel-2 basemap, AOI boundary and empty cells."""
    if S2_RGBA is not None:
        ax.imshow(S2_RGBA, extent=S2_EXTENT, origin='upper',
                  aspect='equal', interpolation='bilinear', zorder=0)
    aoi_wgs = aoi.to_crs(4326) if aoi.crs and aoi.crs.to_epsg() != 4326 else aoi
    aoi_wgs.boundary.plot(ax=ax, color='#E65100', linewidth=1.8, linestyle='--', zorder=5)
    empty = h3_all[h3_all['Occupied'] == 0]
    if len(empty) > 0:
        empty.plot(ax=ax, facecolor='#E8E8E8', edgecolor='#555555',
                   linewidth=0.35, alpha=0.22, zorder=1)


def fig_species_richness():
    """Species richness map (hex cells colored by richness)."""
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')
    h3_div = gpd.read_file(GRID_GPKG, layer='h3_diversity')
    aoi = gpd.read_file(MASTER_GPKG, layer='aoi_boundary')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_facecolor('white')
    plot_base(ax, aoi, h3_all)

    if len(h3_div) > 0:
        vmax = max(h3_div['Richness__S'].max(), 1)
        h3_div.plot(ax=ax, column='Richness__S', cmap='YlOrRd',
                    edgecolor='#666666', linewidth=0.4,
                    vmin=0, vmax=vmax, legend=True,
                    legend_kwds={'label': 'Species Richness (S)',
                                 'shrink': 0.6, 'pad': 0.02},
                    zorder=3)

    ax.set_title('Workshop: Species Richness per H3 Cell', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.ticklabel_format(useOffset=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'workshop_species_richness.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {path}")


def fig_shannon_entropy():
    """Shannon entropy map (hex cells colored by H')."""
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')
    h3_div = gpd.read_file(GRID_GPKG, layer='h3_diversity')
    aoi = gpd.read_file(MASTER_GPKG, layer='aoi_boundary')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_facecolor('white')
    plot_base(ax, aoi, h3_all)

    if len(h3_div) > 0:
        vmax = max(h3_div['Shannon__H'].max(), 0.1)
        h3_div.plot(ax=ax, column='Shannon__H', cmap='viridis',
                    edgecolor='#666666', linewidth=0.4,
                    vmin=0, vmax=vmax, legend=True,
                    legend_kwds={'label': "Shannon Entropy (H')",
                                 'shrink': 0.6, 'pad': 0.02},
                    zorder=3)

    ax.set_title("Workshop: Shannon Entropy (H') per H3 Cell", fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.ticklabel_format(useOffset=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'workshop_shannon_entropy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {path}")


def fig_priority_tiers():
    """Priority tier map (CRITICAL/HIGH/MEDIUM/LOW)."""
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')
    h3_pri = gpd.read_file(GRID_GPKG, layer='h3_priority')
    aoi = gpd.read_file(MASTER_GPKG, layer='aoi_boundary')

    tier_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    tier_colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#bd0026']

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_facecolor('white')
    plot_base(ax, aoi, h3_all)

    if len(h3_pri) > 0:
        # Map tiers to numeric for categorical colormap
        h3_pri = h3_pri.copy()
        h3_pri['Tier_num'] = h3_pri['Tier'].map(
            {t: i for i, t in enumerate(tier_order)})

        cmap = ListedColormap(tier_colors)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = BoundaryNorm(bounds, cmap.N)

        h3_pri.plot(ax=ax, column='Tier_num', cmap=cmap, norm=norm,
                    edgecolor='#666666', linewidth=0.4, zorder=3)

        # Legend
        patches = [mpatches.Patch(facecolor=c, edgecolor='#666',
                                   label=t) for t, c in zip(tier_order, tier_colors)]
        ax.legend(handles=patches, title='Priority Tier', loc='lower right',
                  fontsize=9, title_fontsize=10, framealpha=0.9)

    ax.set_title('Workshop: Conservation Priority Tiers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.ticklabel_format(useOffset=False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'workshop_priority_tiers.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    print("=" * 60)
    print("Workshop Stage 4: Generate static figures")
    print("=" * 60)

    ensure_fig_dir()

    print("\n--- Generating figures ---")
    fig_species_richness()
    fig_shannon_entropy()
    fig_priority_tiers()

    print(f"\nDone! All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    main()
