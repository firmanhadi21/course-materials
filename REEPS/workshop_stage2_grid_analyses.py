"""
Workshop Stage 2: Build Workshop_GridAnalyses.gpkg -- all H3 analysis layers.
Adapted from recalc_stage2_grid_analyses.py for workshop use.
Includes 6-component CPI with permeability from land cover raster.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shp_transform
from collections import Counter
from math import log
import h3
import networkx as nx
import pyproj
import rasterio
from rasterio.mask import mask as rio_mask
import os

BASE = '/Users/macbook/Dropbox/Works/Cisokan/2026/Feb26/data/biodiv/workshop'
BIO_BASE = '/Users/macbook/Dropbox/Works/Cisokan/2026/Feb26/data/biodiv'
MASTER_GPKG = os.path.join(BASE, 'Workshop_Master_Database.gpkg')
OUT_GPKG = os.path.join(BASE, 'Workshop_GridAnalyses.gpkg')
LC_RASTER = os.path.join(BIO_BASE, 'cisokan-spatial/PS_Final_11class_Hierarchical.tif')

# Land cover resistance values (from Withaningsih 2022)
LC_RESISTANCE = {
    0: 50, 1: 90, 2: 50, 3: 100, 4: 1, 5: 5, 6: 5,
    7: 10, 8: 40, 9: 30, 10: 80, 11: 50
}

# REEPS species only (exclude Prionodon linsang per revision decision)
EXCLUDE_SPECIES = {'Prionodon linsang'}

REEPS_SPECIES = {
    'Panthera pardus melas', 'Nycticebus javanicus', 'Manis javanica',
    'Hylobates moloch', 'Presbytis comata', 'Trachypithecus auratus',
    'Nisaetus bartelsi', 'Aonyx cinereus', 'Tragulus kanchil',
    'Prionailurus bengalensis', 'Paradoxurus hermaphroditus',
    'Hystrix javanica', 'Pteropus vampyrus', 'Arctictis binturong',
    'Herpestes javanicus',
}

THREATENED_SPECIES = {
    'Panthera pardus melas', 'Nycticebus javanicus', 'Manis javanica',
    'Hylobates moloch', 'Presbytis comata', 'Nisaetus bartelsi',
    'Prionailurus bengalensis',
}

COMMON_NAMES = {
    'Panthera pardus melas': 'Javan Leopard',
    'Nycticebus javanicus': 'Javan Slow Loris',
    'Manis javanica': 'Sunda Pangolin',
    'Hylobates moloch': 'Javan Gibbon',
    'Presbytis comata': 'Grizzled Langur',
    'Trachypithecus auratus': 'Javan Langur',
    'Nisaetus bartelsi': "Bartels's Hawk-eagle",
    'Aonyx cinereus': 'Asian Small-clawed Otter',
    'Tragulus kanchil': 'Lesser Mouse-deer',
    'Prionailurus bengalensis': 'Leopard Cat',
    'Paradoxurus hermaphroditus': 'Common Palm Civet',
    'Hystrix javanica': 'Javan Porcupine',
    'Pteropus vampyrus': 'Large Flying Fox',
    'Arctictis binturong': 'Binturong',
    'Herpestes javanicus': 'Javan Mongoose',
}

def h3_to_polygon(cell):
    boundary = h3.cell_to_boundary(cell)
    return Polygon([(lng, lt) for lt, lng in boundary])


def load_data():
    gdf_occ = gpd.read_file(MASTER_GPKG, layer='reeps_occurrences')
    gdf_cells = gpd.read_file(MASTER_GPKG, layer='reeps_h3_cells')
    aoi = gpd.read_file(MASTER_GPKG, layer='aoi_boundary')

    # Filter to REEPS species, exclude linsang
    reeps_occ = gdf_occ[
        gdf_occ['Species'].isin(REEPS_SPECIES) &
        ~gdf_occ['Species'].isin(EXCLUDE_SPECIES)
    ].copy()
    print(f"REEPS occurrences (excl. linsang): {len(reeps_occ)}")
    print(f"All occurrences: {len(gdf_occ)}")
    return gdf_occ, reeps_occ, gdf_cells, aoi


def build_h3_all_cells(gdf_cells, reeps_occ):
    """h3_all_cells: base grid with connectivity metrics."""
    all_h3 = list(gdf_cells['h3_index'])
    occupied = set(reeps_occ['h3_index'].unique())

    # Build adjacency graph
    G = nx.Graph()
    for cell in all_h3:
        G.add_node(cell)
    for cell in all_h3:
        for nbr in h3.grid_disk(cell, 1):
            if nbr != cell and nbr in G:
                G.add_edge(cell, nbr)

    rows = []
    for cell in all_h3:
        lat, lon = h3.cell_to_latlng(cell)
        is_occ = cell in occupied

        # K1 occupied neighbors
        k1_nbrs = [n for n in G.neighbors(cell) if n in occupied]
        # K2 occupied neighbors (2-ring)
        k2_cells = h3.grid_disk(cell, 2)
        k2_occ = [c for c in k2_cells if c in occupied and c != cell]
        # Species reachable within K2
        sp_k2 = set()
        for c in k2_occ:
            sp_k2.update(reeps_occ[reeps_occ['h3_index'] == c]['Species'].unique())

        # Patch ID (connected component of occupied cells)
        occ_subgraph = G.subgraph([c for c in all_h3 if c in occupied])
        patch_id = 0
        if is_occ:
            for i, comp in enumerate(nx.connected_components(occ_subgraph)):
                if cell in comp:
                    patch_id = i + 1
                    break

        # Ecological score
        eco = len(k1_nbrs) * 2 + len(k2_occ)

        rows.append({
            'h3_index': cell, 'lat': round(lat, 6), 'lon': round(lon, 6),
            'Cell_Type': 'Occupied' if is_occ else 'Empty',
            'Occupied': 1 if is_occ else 0,
            'Patch_ID': patch_id,
            'Occ__Nbrs': len(k1_nbrs),
            'K2_Occ__Nbrs': len(k2_occ),
            'Sp__Reachable_K2': len(sp_k2),
            'Eco__Score': eco,
        })

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_richness_summary(gdf_cells, reeps_occ):
    """h3_richness_summary: per-cell richness + temporal records."""
    all_h3 = list(gdf_cells['h3_index'])
    all_years = sorted(reeps_occ['Year'].dropna().unique())

    rows = []
    for cell in all_h3:
        lat, lon = h3.cell_to_latlng(cell)
        cell_df = reeps_occ[reeps_occ['h3_index'] == cell]
        total = len(cell_df)
        species = sorted(cell_df['Species'].unique()) if total > 0 else []
        richness = len(species)
        years_data = sorted(cell_df['Year'].dropna().unique()) if total > 0 else []

        trend_dir = 'No Data'
        trend_slope = 0.0
        if len(years_data) >= 2:
            yr_rich = [(y, cell_df[cell_df['Year'] == y]['Species'].nunique()) for y in years_data]
            xs, ys = zip(*yr_rich)
            slope = np.polyfit(xs, ys, 1)[0]
            trend_slope = round(slope, 4)
            trend_dir = 'Increasing' if slope > 0.01 else ('Decreasing' if slope < -0.01 else 'Stable')

        row = {
            'h3_index': cell, 'lat': round(lat, 6), 'lon': round(lon, 6),
            'Total_Records': total, 'Species_Richness': richness,
            'Species_List': ', '.join(species),
            'Trend_Direction': trend_dir, 'Trend_Slope': trend_slope,
            'First_Year': min(years_data) if years_data else None,
            'Last_Year': max(years_data) if years_data else None,
            'Years_w__Data': len(years_data),
        }
        for y in all_years:
            row[f'Records_{int(y)}'] = len(cell_df[cell_df['Year'] == y])
        rows.append(row)

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_diversity(reeps_occ):
    """h3_diversity: Shannon, Simpson, Pielou for occupied cells."""
    occupied = reeps_occ.groupby('h3_index')
    rows = []
    for cell, group in occupied:
        lat, lon = h3.cell_to_latlng(cell)
        counts = Counter(group['Species'])
        N = sum(counts.values())
        S = len(counts)

        if S == 0 or N == 0:
            continue

        # Shannon H'
        H = -sum((n / N) * log(n / N) for n in counts.values() if n > 0)
        # Simpson D
        D = 1 - sum((n / N) ** 2 for n in counts.values())
        # Pielou J
        J = H / log(S) if S > 1 else 0
        # Berger-Parker
        BP = max(counts.values()) / N
        dominant = max(counts, key=counts.get)

        # Diversity trend
        years = sorted(group['Year'].dropna().unique())
        div_trend = 'Stable'
        trend_slope = 0.0
        if len(years) >= 2:
            yr_rich = [(y, group[group['Year'] == y]['Species'].nunique()) for y in years]
            xs, ys = zip(*yr_rich)
            slope = np.polyfit(xs, ys, 1)[0]
            trend_slope = round(slope, 4)
            div_trend = 'Increasing' if slope > 0.01 else ('Decreasing' if slope < -0.01 else 'Stable')

        rows.append({
            'h3_index': cell, 'lat': round(lat, 6), 'lon': round(lon, 6),
            'Richness__S': S, 'Records__N': N,
            'Shannon__H': round(H, 4), 'Simpson__D': round(D, 4),
            'Pielou__J': round(J, 4), 'Berger_Parker__BP': round(BP, 4),
            'Dominant_Species': dominant,
            'Common_Name': COMMON_NAMES.get(dominant, ''),
            'Diversity_Trend': div_trend, 'Trend_Slope': trend_slope,
        })

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def compute_permeability(h3_cells):
    """Compute habitat permeability (100 - mean_resistance) per H3 cell from land cover raster."""
    project_to_utm = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32748', always_xy=True).transform
    permeability = {}

    if not os.path.exists(LC_RASTER):
        print(f"  WARNING: Land cover raster not found at {LC_RASTER}, permeability set to 50")
        return {cell: 50.0 for cell in h3_cells}

    with rasterio.open(LC_RASTER) as src:
        for cell in h3_cells:
            boundary = h3.cell_to_boundary(cell)
            coords = [(lng, lat) for lat, lng in boundary]
            coords.append(coords[0])
            poly_wgs = Polygon(coords)
            poly_utm = shp_transform(project_to_utm, poly_wgs)
            try:
                out, _ = rio_mask(src, [mapping(poly_utm)], crop=True, filled=True, nodata=0)
                data = out[0]
                valid = data[(data >= 1) & (data <= 10)]
                if len(valid) > 0:
                    vals, counts = np.unique(valid, return_counts=True)
                    total_px = counts.sum()
                    mean_r = sum(LC_RESISTANCE.get(int(v), 50) * c for v, c in zip(vals, counts)) / total_px
                    permeability[cell] = round(100.0 - mean_r, 2)
                else:
                    permeability[cell] = 50.0
            except Exception:
                permeability[cell] = 50.0

    return permeability


def build_priority(gdf_diversity, gdf_all_cells, reeps_occ):
    """h3_priority: 6-component PCA-weighted Composite Priority Index."""
    occ_cells = set(reeps_occ['h3_index'].unique())
    div_data = gdf_diversity.set_index('h3_index')
    all_data = gdf_all_cells.set_index('h3_index')

    # Compute permeability for occupied cells
    print("  Computing permeability from land cover raster...")
    perm_scores = compute_permeability(occ_cells)

    rows = []
    for cell in occ_cells:
        lat, lon = h3.cell_to_latlng(cell)
        cell_occ = reeps_occ[reeps_occ['h3_index'] == cell]

        richness = cell_occ['Species'].nunique()
        diversity = div_data.loc[cell, 'Shannon__H'] if cell in div_data.index else 0
        connectivity = all_data.loc[cell, 'Occ__Nbrs'] if cell in all_data.index else 0

        # Co-occurrence: pairs of species in this cell
        species_in_cell = set(cell_occ['Species'].unique())
        cooc = len(species_in_cell) * (len(species_in_cell) - 1) / 2 if len(species_in_cell) > 1 else 0

        # Threatened count
        threatened = len(species_in_cell & THREATENED_SPECIES)

        # Permeability (100 - mean resistance)
        permeability = perm_scores.get(cell, 50.0)

        rows.append({
            'h3_index': cell, 'lat': round(lat, 6), 'lon': round(lon, 6),
            'Richness': richness, 'Diversity': round(diversity, 4),
            'Connectivity': connectivity, 'Co_occurrence': cooc,
            'Threatened': threatened, 'Permeability': permeability,
        })

    df = pd.DataFrame(rows)

    # Normalize all 6 components to 0-1
    for col in ['Richness', 'Diversity', 'Connectivity', 'Co_occurrence', 'Threatened', 'Permeability']:
        vmin, vmax = df[col].min(), df[col].max()
        if vmax > vmin:
            df[f'n_{col}'] = (df[col] - vmin) / (vmax - vmin)
        else:
            df[f'n_{col}'] = 0

    # 6-component PCA-derived CPI weights (from manuscript PCA analysis)
    weights = {
        'Richness': 0.177, 'Connectivity': 0.181, 'Threatened': 0.169,
        'Diversity': 0.180, 'Co_occurrence': 0.180, 'Permeability': 0.113
    }
    df['Priority_Index'] = sum(df[f'n_{k}'] * v for k, v in weights.items())
    df['Priority_Index'] = df['Priority_Index'].round(4)

    # Rank and tier
    df = df.sort_values('Priority_Index', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    df['Tier'] = df['Priority_Index'].apply(
        lambda x: 'CRITICAL' if x >= 0.7 else ('HIGH' if x >= 0.5 else ('MEDIUM' if x >= 0.3 else 'LOW')))

    # Clean up
    df = df.drop(columns=[c for c in df.columns if c.startswith('n_')])
    cols = ['Rank', 'h3_index', 'Tier', 'Richness', 'Diversity', 'Connectivity',
            'Co_occurrence', 'Threatened', 'Permeability', 'Priority_Index', 'lat', 'lon']
    df = df[cols]

    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_temporal_traj(gdf_cells, reeps_occ):
    """h3_temporal_traj: temporal trajectory per cell."""
    all_h3 = list(gdf_cells['h3_index'])
    all_years = sorted(reeps_occ['Year'].dropna().unique())

    rows = []
    for cell in all_h3:
        lat, lon = h3.cell_to_latlng(cell)
        cell_df = reeps_occ[reeps_occ['h3_index'] == cell]
        years_present = sorted(cell_df['Year'].dropna().unique())

        if len(years_present) == 0:
            row = {'h3_index': cell, 'Status': 'Never Detected',
                   'First_Detected': None, 'Last_Detected': None,
                   'Years_Detected': 0, 'Total_Species_Ever': 0,
                   'Total_Colonisations': 0, 'Total_Extinctions': 0,
                   'Mean_Beta_Temporal': 0, 'Richness_Trend': 'No Data',
                   'lat': round(lat, 6), 'lon': round(lon, 6)}
            for y in all_years:
                row[f'Rich_{int(y)}'] = 0
            rows.append(row)
            continue

        # Species per year
        sp_per_year = {}
        for y in all_years:
            sp_per_year[y] = set(cell_df[cell_df['Year'] == y]['Species'].unique())

        # Colonisations/extinctions between consecutive survey years
        total_col, total_ext = 0, 0
        betas = []
        sorted_present = [y for y in all_years if len(sp_per_year[y]) > 0]
        for i in range(1, len(sorted_present)):
            prev_sp = sp_per_year[sorted_present[i - 1]]
            curr_sp = sp_per_year[sorted_present[i]]
            gained = len(curr_sp - prev_sp)
            lost = len(prev_sp - curr_sp)
            total_col += gained
            total_ext += lost
            union = len(prev_sp | curr_sp)
            if union > 0:
                betas.append((gained + lost) / union)

        all_sp = set()
        for sp_set in sp_per_year.values():
            all_sp.update(sp_set)

        # Richness trend
        yr_rich = [(y, len(sp_per_year[y])) for y in sorted_present]
        trend = 'Stable'
        if len(yr_rich) >= 2:
            xs, ys = zip(*yr_rich)
            slope = np.polyfit(xs, ys, 1)[0]
            trend = 'Increasing' if slope > 0.01 else ('Decreasing' if slope < -0.01 else 'Stable')

        status = 'Persistent' if len(sorted_present) >= 3 else (
            'Recent' if sorted_present[-1] >= 2020 else 'Historical')

        row = {
            'h3_index': cell, 'Status': status,
            'First_Detected': min(sorted_present),
            'Last_Detected': max(sorted_present),
            'Years_Detected': len(sorted_present),
            'Total_Species_Ever': len(all_sp),
            'Total_Colonisations': total_col,
            'Total_Extinctions': total_ext,
            'Mean_Beta_Temporal': round(np.mean(betas), 4) if betas else 0,
            'Richness_Trend': trend,
            'lat': round(lat, 6), 'lon': round(lon, 6),
        }
        for y in all_years:
            row[f'Rich_{int(y)}'] = len(sp_per_year[y])
        rows.append(row)

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_temporal_matrix(gdf_cells, reeps_occ):
    """h3_temporal_matrix: records/species per year per cell."""
    all_h3 = list(gdf_cells['h3_index'])
    all_years = sorted(reeps_occ['Year'].dropna().unique())

    rows = []
    for cell in all_h3:
        lat, lon = h3.cell_to_latlng(cell)
        cell_df = reeps_occ[reeps_occ['h3_index'] == cell]
        row = {'h3_index': cell, 'lat': round(lat, 6), 'lon': round(lon, 6)}
        total = 0
        yrs_with = 0
        for y in all_years:
            yr_df = cell_df[cell_df['Year'] == y]
            n_rec = len(yr_df)
            n_sp = yr_df['Species'].nunique()
            row[f'{int(y)}_Records'] = n_rec
            row[f'{int(y)}_Species'] = n_sp
            total += n_rec
            if n_rec > 0:
                yrs_with += 1
        row['Total_Records'] = total
        row['Years_With_Data'] = yrs_with
        row['Status'] = 'Occupied' if total > 0 else 'Empty'
        rows.append(row)

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_chao1(reeps_occ):
    """h3_chao1: Chao1 richness estimator per occupied cell."""
    rows = []
    for cell, group in reeps_occ.groupby('h3_index'):
        lat, lon = h3.cell_to_latlng(cell)
        counts = Counter(group['Species'])
        N = sum(counts.values())
        S_obs = len(counts)
        f1 = sum(1 for v in counts.values() if v == 1)
        f2 = sum(1 for v in counts.values() if v == 2)

        if f2 > 0:
            chao1 = S_obs + (f1 ** 2) / (2 * f2)
        elif f1 > 0:
            chao1 = S_obs + f1 * (f1 - 1) / 2
        else:
            chao1 = S_obs

        completeness = round(S_obs / chao1 * 100, 1) if chao1 > 0 else 100
        years = sorted(group['Year'].dropna().unique())

        rows.append({
            'h3_index': cell,
            'Total_Records': N, 'S_obs': S_obs,
            'Singletons_f1': f1, 'Doubletons_f2': f2,
            'Chao1_Estimate': round(chao1, 2),
            'Completeness_pct': completeness,
            'Undetected_est': round(chao1 - S_obs, 1),
            'Years_Surveyed': len(years),
            'Survey_Years': ', '.join(str(int(y)) for y in years),
            'lat': round(lat, 6), 'lon': round(lon, 6),
        })

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_corridor_gaps(gdf_all_cells, reeps_occ):
    """h3_corridor_gaps: stepping stone analysis for unoccupied cells."""
    all_data = gdf_all_cells.set_index('h3_index')
    occ_cells = set(reeps_occ['h3_index'].unique())
    empty_cells = set(all_data.index) - occ_cells

    # Build graph
    G = nx.Graph()
    for cell in all_data.index:
        G.add_node(cell)
        for nbr in h3.grid_disk(cell, 1):
            if nbr != cell and nbr in all_data.index:
                G.add_edge(cell, nbr)

    # Betweenness on occupied subgraph + gap cells
    betweenness = nx.betweenness_centrality(G)

    rows = []
    for cell in empty_cells:
        lat, lon = h3.cell_to_latlng(cell)
        k1_occ = [n for n in G.neighbors(cell) if n in occ_cells]
        k2_ring = h3.grid_disk(cell, 2)
        k2_richness = 0
        for c in k2_ring:
            if c in occ_cells and c != cell:
                k2_richness += reeps_occ[reeps_occ['h3_index'] == c]['Species'].nunique()

        dist_nearest = float('inf')
        for n in G.neighbors(cell):
            if n in occ_cells:
                dist_nearest = 1
                break
        if dist_nearest == float('inf'):
            for c in h3.grid_disk(cell, 3):
                if c in occ_cells:
                    dist_nearest = h3.grid_distance(cell, c)
                    break

        is_corridor = len(k1_occ) >= 2
        score = (len(k1_occ) * 3 + k2_richness * 0.5 +
                 betweenness.get(cell, 0) * 10)
        tier = 'High' if score > 5 else ('Medium' if score > 2 else 'Low')

        rows.append({
            'h3_index': cell, 'lat': round(lat, 6), 'lon': round(lon, 6),
            'Betweenness': round(betweenness.get(cell, 0), 6),
            'K1_Occ_Neighbors': len(k1_occ),
            'K2_Richness_Reachable': k2_richness,
            'Dist_to_Nearest_Occ': dist_nearest if dist_nearest != float('inf') else -1,
            'Is_Corridor_Candidate': 1 if is_corridor else 0,
            'Connects_Priority_Cells': 0,
            'Stepping_Stone_Score': round(score, 2),
            'Corridor_Tier': tier,
        })

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_isolation_risk(gdf_priority, gdf_all_cells, reeps_occ):
    """h3_isolation_risk: occupied cells at risk of isolation."""
    occ_cells = set(reeps_occ['h3_index'].unique())
    all_idx = set(gdf_all_cells['h3_index'])
    priority_data = gdf_priority.set_index('h3_index')

    rows = []
    for cell in occ_cells:
        lat, lon = h3.cell_to_latlng(cell)
        k1 = [n for n in h3.grid_disk(cell, 1) if n != cell and n in all_idx]
        direct_occ = [n for n in k1 if n in occ_cells]
        gap_nbrs = [n for n in k1 if n not in occ_cells]

        # Extra occupied cells reachable via gap cells (K2)
        extra_via_gaps = set()
        for gap in gap_nbrs:
            for n2 in h3.grid_disk(gap, 1):
                if n2 in occ_cells and n2 != cell and n2 not in set(direct_occ):
                    extra_via_gaps.add(n2)

        pi = priority_data.loc[cell, 'Priority_Index'] if cell in priority_data.index else 0
        tier = priority_data.loc[cell, 'Tier'] if cell in priority_data.index else 'LOW'
        richness = reeps_occ[reeps_occ['h3_index'] == cell]['Species'].nunique()

        rows.append({
            'h3_index': cell, 'Priority_Index': pi, 'Priority_Tier': tier,
            'Direct_Occ_Neighbors': len(direct_occ),
            'Gap_Neighbors': len(gap_nbrs),
            'Extra_Occ_via_Gaps': len(extra_via_gaps),
            'Total_Connectivity': len(direct_occ) + len(extra_via_gaps),
            'Species_Richness': richness,
            'lat': round(lat, 6), 'lon': round(lon, 6),
        })

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_survey_gaps(gdf_all_cells, reeps_occ):
    """h3_survey_gaps: unoccupied cells ranked by survey priority."""
    occ_cells = set(reeps_occ['h3_index'].unique())
    all_idx = set(gdf_all_cells['h3_index'])
    empty = all_idx - occ_cells

    rows = []
    for cell in empty:
        lat, lon = h3.cell_to_latlng(cell)
        k1 = [n for n in h3.grid_disk(cell, 1) if n != cell and n in all_idx]
        k1_occ = [n for n in k1 if n in occ_cells]
        k2 = [n for n in h3.grid_disk(cell, 2) if n != cell and n in all_idx]
        k2_occ = [n for n in k2 if n in occ_cells]

        # Distance to nearest occupied
        dist = -1
        for r in range(1, 6):
            ring = h3.grid_disk(cell, r)
            if any(c in occ_cells for c in ring):
                dist = r
                break

        # Neighbor richness
        k1_rich = sum(reeps_occ[reeps_occ['h3_index'] == n]['Species'].nunique()
                       for n in k1_occ)
        k2_rich = sum(reeps_occ[reeps_occ['h3_index'] == n]['Species'].nunique()
                       for n in k2_occ)

        is_corridor = len(k1_occ) >= 2
        gap_score = k1_rich * 2 + k2_rich * 0.5 + (10 if is_corridor else 0)
        priority = 'High' if gap_score > 10 else ('Medium' if gap_score > 3 else 'Low')

        rows.append({
            'h3_index': cell, 'lat': round(lat, 6), 'lon': round(lon, 6),
            'K1_Occ_Neighbors': len(k1_occ), 'K2_Occ_Neighbors': len(k2_occ),
            'Dist_to_Nearest_Occ_km': dist,
            'K1_Nbr_Richness': k1_rich, 'K2_Nbr_Richness': k2_rich,
            'Corridor_Candidate': 1 if is_corridor else 0,
            'Gap_Priority_Score': round(gap_score, 2),
            'Survey_Priority': priority,
        })

    df = pd.DataFrame(rows)
    geoms = [h3_to_polygon(c) for c in df['h3_index']]
    return gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')


def build_master_occupied(gdf_richness, gdf_diversity, gdf_priority,
                          gdf_temporal, gdf_chao1, gdf_isolation):
    """h3_master_occupied: merged view of all metrics for occupied cells."""
    occ = gdf_richness[gdf_richness['Total_Records'] > 0].copy()
    base = occ.set_index('h3_index')

    # Merge diversity
    if len(gdf_diversity) > 0:
        div = gdf_diversity.drop(columns=['geometry', 'lat', 'lon']).set_index('h3_index')
        overlap = base.columns.intersection(div.columns)
        div = div.drop(columns=overlap, errors='ignore')
        base = base.join(div, how='left')

    # Merge priority
    if len(gdf_priority) > 0:
        pri = gdf_priority.drop(columns=['geometry', 'lat', 'lon']).set_index('h3_index')
        for c in ['Rank']:
            if c in pri.columns and c in base.columns:
                pri = pri.rename(columns={c: f'Priority_{c}'})
        base = base.join(pri, how='left', rsuffix='_pri')

    # Merge temporal
    if len(gdf_temporal) > 0:
        tmp = gdf_temporal.drop(columns=['geometry', 'lat', 'lon']).set_index('h3_index')
        base = base.join(tmp, how='left', rsuffix='_temp')

    # Merge chao1
    if len(gdf_chao1) > 0:
        ch = gdf_chao1.drop(columns=['geometry', 'lat', 'lon']).set_index('h3_index')
        base = base.join(ch, how='left', rsuffix='_chao')

    # Merge isolation
    if len(gdf_isolation) > 0:
        iso = gdf_isolation.drop(columns=['geometry', 'lat', 'lon']).set_index('h3_index')
        base = base.join(iso, how='left', rsuffix='_iso')

    base = base.reset_index()
    return gpd.GeoDataFrame(base, geometry='geometry', crs='EPSG:4326')


def main():
    print("=" * 60)
    print("Workshop Stage 2: Build Workshop_GridAnalyses.gpkg")
    print("=" * 60)

    gdf_occ, reeps_occ, gdf_cells, aoi = load_data()

    print("\n--- Building h3_all_cells ---")
    gdf_all = build_h3_all_cells(gdf_cells, reeps_occ)
    print(f"  {len(gdf_all)} cells")

    print("\n--- Building h3_richness_summary ---")
    gdf_rich = build_richness_summary(gdf_cells, reeps_occ)
    print(f"  {len(gdf_rich)} cells")

    print("\n--- Building h3_diversity ---")
    gdf_div = build_diversity(reeps_occ)
    print(f"  {len(gdf_div)} occupied cells with diversity")

    print("\n--- Building h3_priority ---")
    gdf_pri = build_priority(gdf_div, gdf_all, reeps_occ)
    print(f"  {len(gdf_pri)} ranked cells")
    print(f"  Tiers: {gdf_pri['Tier'].value_counts().to_dict()}")

    print("\n--- Building h3_temporal_traj ---")
    gdf_temp = build_temporal_traj(gdf_cells, reeps_occ)
    print(f"  {len(gdf_temp)} cells")

    print("\n--- Building h3_temporal_matrix ---")
    gdf_tmat = build_temporal_matrix(gdf_cells, reeps_occ)
    print(f"  {len(gdf_tmat)} cells")

    print("\n--- Building h3_chao1 ---")
    gdf_chao = build_chao1(reeps_occ)
    print(f"  {len(gdf_chao)} occupied cells")

    print("\n--- Building h3_corridor_gaps ---")
    gdf_corr = build_corridor_gaps(gdf_all, reeps_occ)
    print(f"  {len(gdf_corr)} gap cells")

    print("\n--- Building h3_isolation_risk ---")
    gdf_iso = build_isolation_risk(gdf_pri, gdf_all, reeps_occ)
    print(f"  {len(gdf_iso)} occupied cells")

    print("\n--- Building h3_survey_gaps ---")
    gdf_sgap = build_survey_gaps(gdf_all, reeps_occ)
    print(f"  {len(gdf_sgap)} gap cells")

    print("\n--- Building h3_master_occupied ---")
    gdf_master = build_master_occupied(gdf_rich, gdf_div, gdf_pri,
                                        gdf_temp, gdf_chao, gdf_iso)
    print(f"  {len(gdf_master)} occupied cells")

    # Write all layers
    print(f"\nWriting {OUT_GPKG}...")
    if os.path.exists(OUT_GPKG):
        os.remove(OUT_GPKG)

    layers = [
        ('h3_all_cells', gdf_all),
        ('h3_richness_summary', gdf_rich),
        ('h3_diversity', gdf_div),
        ('h3_priority', gdf_pri),
        ('h3_temporal_traj', gdf_temp),
        ('h3_temporal_matrix', gdf_tmat),
        ('h3_chao1', gdf_chao),
        ('h3_corridor_gaps', gdf_corr),
        ('h3_isolation_risk', gdf_iso),
        ('h3_survey_gaps', gdf_sgap),
        ('h3_master_occupied', gdf_master),
        ('reeps_occurrences', reeps_occ),
        ('aoi_boundary', aoi),
    ]

    for name, gdf in layers:
        gdf.to_file(OUT_GPKG, layer=name, driver='GPKG')
        print(f"  Wrote {name}: {len(gdf)} rows")

    print("\nDone! Workshop_GridAnalyses.gpkg built.")


if __name__ == '__main__':
    main()
