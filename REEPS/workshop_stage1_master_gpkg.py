"""
Workshop Stage 1: Build Workshop_Master_Database.gpkg from synthetic Excel.
Adapted from recalc_stage1_master_gpkg.py for workshop use.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
import os

BASE = '/Users/macbook/Dropbox/Works/Cisokan/2026/Feb26/data/biodiv/workshop'
EXCEL = os.path.join(BASE, 'Workshop_Biodiversity_Database.xlsx')
GPKG_OUT = os.path.join(BASE, 'Workshop_Master_Database.gpkg')
AOI_SRC = os.path.join(BASE, 'aoi.gpkg')

# ── Species harmonization (canonical names for REEPS analysis) ────────
SPECIES_MAP = {
    'Panthera pardus melas': 'Panthera pardus melas',
    'Panthera pardus': 'Panthera pardus melas',
    'macan': 'Panthera pardus melas',
    'Macan': 'Panthera pardus melas',
    'Nycticebus javanicus': 'Nycticebus javanicus',
    'Nycticebus coucang': 'Nycticebus javanicus',
    'Manis javanica': 'Manis javanica',
    'Trenggiling': 'Manis javanica',
    'Hylobates moloch': 'Hylobates moloch',
    'Owa': 'Hylobates moloch',
    'owa': 'Hylobates moloch',
    'Presbytis comata': 'Presbytis comata',
    'Presbytys comata': 'Presbytis comata',
    'surili': 'Presbytis comata',
    'surilil': 'Presbytis comata',
    'Trachypithecus auratus': 'Trachypithecus auratus',
    'Trachypthecus aureus': 'Trachypithecus auratus',
    'Lutung': 'Trachypithecus auratus',
    'lutung': 'Trachypithecus auratus',
    'Nisaetus bartelsi': 'Nisaetus bartelsi',
    'Aonyx cinereus': 'Aonyx cinereus',
    'Aonyx cinerea': 'Aonyx cinereus',
    'Aonyx aonyx': 'Aonyx cinereus',
    'Sero': 'Aonyx cinereus',
    'sero': 'Aonyx cinereus',
    'Herpestes javanicus': 'Herpestes javanicus',
    'Tragulus kanchil': 'Tragulus kanchil',
    'Tragulus javanicus': 'Tragulus kanchil',
    'pelanduk': 'Tragulus kanchil',
    'Prionodon linsang': 'Prionodon linsang',
    'Prionailurus bengalensis': 'Prionailurus bengalensis',
    'Paradoxurus hermaphroditus': 'Paradoxurus hermaphroditus',
    'Paradoxurus hemaphroditus': 'Paradoxurus hermaphroditus',
    'Paradoxurus hemaproditus': 'Paradoxurus hermaphroditus',
    'Hystrix javanica': 'Hystrix javanica',
    'Landak': 'Hystrix javanica',
    'landak': 'Hystrix javanica',
    'Pteropus vampyrus': 'Pteropus vampyrus',
    'Arctictis binturong': 'Arctictis binturong',
    'Sus scrofa': 'Sus scrofa',
    'Babi Hutan': 'Sus scrofa',
    'babi hutan': 'Sus scrofa',
    'Babi': 'Sus scrofa',
    'Macaca fascicularis': 'Macaca fascicularis',
    'Monyet': 'Macaca fascicularis',
    'monyet': 'Macaca fascicularis',
    'Paguma larvata': 'Paguma larvata',
    'Malayophyton reticulatus': 'Malayophyton reticulatus',
    'Centropus bengalensis': 'Centropus bengalensis',
    'Callosciurius sp': 'Callosciurus sp.',
    'Neofelis nebulosa': 'Neofelis nebulosa',
    'Prionailurus planiceps': 'Prionailurus planiceps',
}

H3_RES = 8


def load_excel():
    """Load Master Database sheet from Excel."""
    df = pd.read_excel(EXCEL, sheet_name='Master Database', header=3)
    df.columns = ['No', 'Source', 'Species', 'Common_Name', 'Status',
                  'Year', 'Month', 'Month_Year', 'Survey_Method',
                  'Location', 'Latitude', 'Longitude']
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])
    print(f"Loaded {len(df)} records from Excel")
    return df


def harmonize_species(df):
    """Apply species name harmonization."""
    df['Species_Original'] = df['Species']
    df['Species'] = df['Species'].map(lambda x: SPECIES_MAP.get(x, x))
    changed = (df['Species'] != df['Species_Original']).sum()
    print(f"Harmonized {changed} species name variants")
    print(f"Unique species: {df['Species'].nunique()}")
    return df


def assign_h3(df):
    """Assign H3 cell index to each record."""
    df['h3_index'] = df.apply(
        lambda r: h3.latlng_to_cell(r['Latitude'], r['Longitude'], H3_RES), axis=1)
    print(f"Assigned H3 res-{H3_RES} indices to {len(df)} records")
    print(f"Unique H3 cells with data: {df['h3_index'].nunique()}")
    return df


def build_occurrences_gdf(df):
    """Build GeoDataFrame of occurrences."""
    geometry = [Point(row['Longitude'], row['Latitude']) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    cols = ['No', 'Source', 'Species', 'Common_Name', 'Status', 'Year', 'Month',
            'Month_Year', 'Survey_Method', 'Location', 'Latitude', 'Longitude',
            'h3_index', 'geometry']
    return gdf[cols]


def build_h3_cells(df, aoi_gdf):
    """Build H3 cell summary layer covering the AOI.

    Workshop version: no pre-existing GridAnalyses.gpkg.
    Uses data cells + 2-ring buffer via h3.grid_disk to cover AOI.
    """
    # Start from data cells, expand by 2 rings to cover AOI
    data_cells = set(df['h3_index'].unique())
    expanded = set()
    for cell in data_cells:
        expanded.update(h3.grid_disk(cell, 2))

    # Filter to cells that intersect AOI (reproject AOI to WGS84 if needed)
    aoi_wgs = aoi_gdf.to_crs(4326) if aoi_gdf.crs and aoi_gdf.crs.to_epsg() != 4326 else aoi_gdf
    aoi_union = aoi_wgs.union_all()
    valid_cells = []
    for cell in expanded:
        poly = Polygon([(lng, lt) for lt, lng in h3.cell_to_boundary(cell)])
        if poly.intersects(aoi_union):
            valid_cells.append(cell)

    print(f"Total H3 cells in AOI: {len(valid_cells)} (data cells: {len(data_cells)}, expanded via 2-ring)")

    # Get all survey years
    all_years = sorted(df['Year'].dropna().unique())
    print(f"Survey years: {all_years}")

    # Per-cell statistics
    rows = []
    for cell in sorted(valid_cells):
        lat, lon = h3.cell_to_latlng(cell)
        cell_df = df[df['h3_index'] == cell]

        total = len(cell_df)
        species_list = sorted(cell_df['Species'].unique()) if total > 0 else []
        richness = len(species_list)

        # Temporal: years with data
        years_with_data = sorted(cell_df['Year'].dropna().unique()) if total > 0 else []
        n_years = len(years_with_data)
        first_year = min(years_with_data) if years_with_data else None
        last_year = max(years_with_data) if years_with_data else None

        # Trend (simple linear if >= 2 years)
        trend_dir = 'No Data'
        trend_slope = 0.0
        if n_years >= 2:
            year_richness = []
            for y in years_with_data:
                yr_species = cell_df[cell_df['Year'] == y]['Species'].nunique()
                year_richness.append((y, yr_species))
            if len(year_richness) >= 2:
                xs = [yr for yr, _ in year_richness]
                ys = [r for _, r in year_richness]
                slope = np.polyfit(xs, ys, 1)[0]
                trend_slope = round(slope, 4)
                if slope > 0.01:
                    trend_dir = 'Increasing'
                elif slope < -0.01:
                    trend_dir = 'Decreasing'
                else:
                    trend_dir = 'Stable'

        # Per-year record counts
        year_counts = {}
        for y in all_years:
            year_counts[f'Records_{int(y)}'] = len(cell_df[cell_df['Year'] == y])

        row = {
            'h3_index': cell,
            'Lat': round(lat, 6),
            'Lon': round(lon, 6),
            'Total_Records': total,
            'Species_Richness': richness,
            'Species_List': ', '.join(species_list),
            'Trend_Direction': trend_dir,
            'Trend_Slope': trend_slope,
            'First_Year': first_year,
            'Last_Year': last_year,
            'Years_w__Data': n_years,
        }
        row.update(year_counts)
        rows.append(row)

    cells_df = pd.DataFrame(rows)

    # Build geometry
    geoms = []
    for cell in cells_df['h3_index']:
        boundary = h3.cell_to_boundary(cell)
        geoms.append(Polygon([(lng, lt) for lt, lng in boundary]))

    gdf = gpd.GeoDataFrame(cells_df, geometry=geoms, crs='EPSG:4326')
    occupied = (gdf['Total_Records'] > 0).sum()
    print(f"Occupied cells: {occupied}/{len(gdf)}")
    return gdf


def main():
    print("=" * 60)
    print("Workshop Stage 1: Build Workshop_Master_Database.gpkg")
    print("=" * 60)

    # Load AOI
    aoi_gdf = gpd.read_file(AOI_SRC)
    print(f"Loaded AOI from {AOI_SRC}")

    # Load and process data
    df = load_excel()
    df = harmonize_species(df)
    df = assign_h3(df)

    # Build layers
    gdf_occ = build_occurrences_gdf(df)
    gdf_cells = build_h3_cells(df, aoi_gdf)

    # Write GeoPackage
    print(f"\nWriting {GPKG_OUT}...")
    if os.path.exists(GPKG_OUT):
        os.remove(GPKG_OUT)
    gdf_occ.to_file(GPKG_OUT, layer='reeps_occurrences', driver='GPKG')
    gdf_cells.to_file(GPKG_OUT, layer='reeps_h3_cells', driver='GPKG')
    aoi_gdf.to_file(GPKG_OUT, layer='aoi_boundary', driver='GPKG')

    print(f"\nDone! Workshop_Master_Database.gpkg built:")
    print(f"  reeps_occurrences: {len(gdf_occ)} records")
    print(f"  reeps_h3_cells: {len(gdf_cells)} cells")
    print(f"  aoi_boundary: {len(aoi_gdf)} polygon(s)")

    # Print species year summary
    print(f"\nRecords by year:")
    for y in sorted(df['Year'].dropna().unique()):
        n = len(df[df['Year'] == y])
        print(f"  {int(y)}: {n} records")


if __name__ == '__main__':
    main()
