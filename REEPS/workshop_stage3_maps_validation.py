"""
Workshop Stage 3: Regenerate HTML maps and validation CSVs.
Adapted from recalc_stage5_maps_validation.py for workshop use.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter
from scipy.stats import spearmanr
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
import os

BASE = '/Users/macbook/Dropbox/Works/Cisokan/2026/Feb26/data/biodiv/workshop'
GRID_GPKG = os.path.join(BASE, 'Workshop_GridAnalyses.gpkg')
MASTER_GPKG = os.path.join(BASE, 'Workshop_Master_Database.gpkg')

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
    'Herpestes javanicus': 'Javan Mongoose',
}


def style_hex(feature, value_key, colormap):
    val = feature['properties'].get(value_key, 0)
    return {
        'fillColor': colormap(val) if val and val > 0 else '#ffffff',
        'color': '#333333',
        'weight': 1,
        'fillOpacity': 0.6 if val and val > 0 else 0.1,
    }


def build_diversity_map():
    """Workshop_Diversity_Map.html"""
    h3_div = gpd.read_file(GRID_GPKG, layer='h3_diversity')
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')

    center = [h3_all.geometry.centroid.y.mean(), h3_all.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    # Richness layer
    if len(h3_div) > 0:
        max_r = h3_div['Richness__S'].max()
        cmap_r = cm.LinearColormap(['#ffffcc', '#fd8d3c', '#bd0026'],
                                    vmin=0, vmax=max_r, caption='Species Richness')
        folium.GeoJson(
            h3_div.__geo_interface__,
            name='Species Richness',
            style_function=lambda f: style_hex(f, 'Richness__S', cmap_r),
            tooltip=folium.GeoJsonTooltip(
                fields=['h3_index', 'Richness__S', 'Shannon__H', 'Simpson__D',
                         'Dominant_Species', 'Diversity_Trend'],
                aliases=['Cell', 'Richness', "Shannon H'", "Simpson D",
                         'Dominant', 'Trend']),
        ).add_to(m)
        cmap_r.add_to(m)

    # Empty cells
    empty = h3_all[h3_all['Occupied'] == 0]
    if len(empty) > 0:
        folium.GeoJson(
            empty.__geo_interface__, name='Empty Cells',
            style_function=lambda f: {'fillColor': '#f0f0f0', 'color': '#ccc',
                                       'weight': 0.5, 'fillOpacity': 0.2},
        ).add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(BASE, 'Workshop_Diversity_Map.html')
    m.save(path)
    print(f"  Saved {path}")


def build_priority_map():
    """Workshop_Priority_SurveyGap_Map.html"""
    h3_pri = gpd.read_file(GRID_GPKG, layer='h3_priority')
    h3_sgap = gpd.read_file(GRID_GPKG, layer='h3_survey_gaps')
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')

    center = [h3_all.geometry.centroid.y.mean(), h3_all.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    tier_colors = {'CRITICAL': '#bd0026', 'HIGH': '#fd8d3c',
                   'MEDIUM': '#fecc5c', 'LOW': '#ffffb2'}

    if len(h3_pri) > 0:
        folium.GeoJson(
            h3_pri.__geo_interface__, name='Priority (CPI)',
            style_function=lambda f: {
                'fillColor': tier_colors.get(f['properties'].get('Tier', 'LOW'), '#eee'),
                'color': '#333', 'weight': 1, 'fillOpacity': 0.7},
            tooltip=folium.GeoJsonTooltip(
                fields=['Rank', 'Tier', 'Priority_Index', 'Richness', 'Threatened'],
                aliases=['Rank', 'Tier', 'CPI', 'Richness', 'Threatened']),
        ).add_to(m)

    gap_colors = {'High': '#e31a1c', 'Medium': '#ff7f00', 'Low': '#bbb'}
    if len(h3_sgap) > 0:
        folium.GeoJson(
            h3_sgap.__geo_interface__, name='Survey Gaps',
            style_function=lambda f: {
                'fillColor': gap_colors.get(f['properties'].get('Survey_Priority', 'Low'), '#eee'),
                'color': '#999', 'weight': 0.5, 'fillOpacity': 0.4},
            tooltip=folium.GeoJsonTooltip(
                fields=['h3_index', 'Survey_Priority', 'Gap_Priority_Score'],
                aliases=['Cell', 'Priority', 'Score']),
            show=False,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(BASE, 'Workshop_Priority_SurveyGap_Map.html')
    m.save(path)
    print(f"  Saved {path}")


def build_connectivity_map():
    """Workshop_Connectivity_Map.html"""
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')

    center = [h3_all.geometry.centroid.y.mean(), h3_all.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    max_eco = h3_all['Eco__Score'].max()
    cmap = cm.LinearColormap(['#f7fcf5', '#238b45', '#00441b'],
                              vmin=0, vmax=max(max_eco, 1),
                              caption='Ecological Connectivity Score')
    folium.GeoJson(
        h3_all.__geo_interface__, name='Connectivity',
        style_function=lambda f: style_hex(f, 'Eco__Score', cmap),
        tooltip=folium.GeoJsonTooltip(
            fields=['h3_index', 'Cell_Type', 'Occ__Nbrs', 'Eco__Score'],
            aliases=['Cell', 'Type', 'Occ. Neighbors', 'Eco Score']),
    ).add_to(m)
    cmap.add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(BASE, 'Workshop_Connectivity_Map.html')
    m.save(path)
    print(f"  Saved {path}")


def build_corridor_map():
    """Workshop_Corridor_Map.html"""
    h3_corr = gpd.read_file(GRID_GPKG, layer='h3_corridor_gaps')
    h3_iso = gpd.read_file(GRID_GPKG, layer='h3_isolation_risk')
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')

    center = [h3_all.geometry.centroid.y.mean(), h3_all.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    tier_colors = {'High': '#e31a1c', 'Medium': '#ff7f00', 'Low': '#ccc'}
    if len(h3_corr) > 0:
        folium.GeoJson(
            h3_corr.__geo_interface__, name='Corridor Gaps',
            style_function=lambda f: {
                'fillColor': tier_colors.get(f['properties'].get('Corridor_Tier', 'Low'), '#eee'),
                'color': '#666', 'weight': 0.5, 'fillOpacity': 0.5},
            tooltip=folium.GeoJsonTooltip(
                fields=['h3_index', 'Corridor_Tier', 'Stepping_Stone_Score',
                         'K1_Occ_Neighbors'],
                aliases=['Cell', 'Tier', 'Score', 'Occ. Neighbors']),
        ).add_to(m)

    if len(h3_iso) > 0:
        folium.GeoJson(
            h3_iso.__geo_interface__, name='Isolation Risk',
            style_function=lambda f: {
                'fillColor': '#4575b4' if f['properties'].get('Total_Connectivity', 0) <= 1
                else '#91bfdb',
                'color': '#333', 'weight': 1, 'fillOpacity': 0.6},
            tooltip=folium.GeoJsonTooltip(
                fields=['h3_index', 'Priority_Tier', 'Direct_Occ_Neighbors',
                         'Total_Connectivity', 'Species_Richness'],
                aliases=['Cell', 'Priority', 'Direct Nbrs', 'Total Connect.', 'Richness']),
            show=False,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(BASE, 'Workshop_Corridor_Map.html')
    m.save(path)
    print(f"  Saved {path}")


def build_temporal_map():
    """Workshop_Temporal_Turnover_Map.html"""
    h3_temp = gpd.read_file(GRID_GPKG, layer='h3_temporal_traj')

    occ = h3_temp[h3_temp['Status'] != 'Never Detected']
    center = [h3_temp.geometry.centroid.y.mean(), h3_temp.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    trend_colors = {'Increasing': '#1a9850', 'Stable': '#fee08b',
                    'Decreasing': '#d73027', 'No Data': '#f0f0f0'}

    if len(occ) > 0:
        folium.GeoJson(
            occ.__geo_interface__, name='Temporal Trajectories',
            style_function=lambda f: {
                'fillColor': trend_colors.get(
                    f['properties'].get('Richness_Trend', 'No Data'), '#eee'),
                'color': '#333', 'weight': 1, 'fillOpacity': 0.6},
            tooltip=folium.GeoJsonTooltip(
                fields=['h3_index', 'Status', 'Years_Detected', 'Richness_Trend',
                         'Total_Species_Ever', 'Mean_Beta_Temporal'],
                aliases=['Cell', 'Status', 'Years', 'Trend', 'Total Spp', 'Beta']),
        ).add_to(m)

    # Empty cells
    empty = h3_temp[h3_temp['Status'] == 'Never Detected']
    if len(empty) > 0:
        folium.GeoJson(
            empty.__geo_interface__, name='Undetected',
            style_function=lambda f: {'fillColor': '#f0f0f0', 'color': '#ccc',
                                       'weight': 0.5, 'fillOpacity': 0.15},
            show=False,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(BASE, 'Workshop_Temporal_Turnover_Map.html')
    m.save(path)
    print(f"  Saved {path}")


def build_h3_map():
    """Workshop_H3_Map.html"""
    h3_rich = gpd.read_file(GRID_GPKG, layer='h3_richness_summary')

    center = [h3_rich.geometry.centroid.y.mean(), h3_rich.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    max_r = max(h3_rich['Total_Records'].max(), 1)
    cmap = cm.LinearColormap(['#f7fbff', '#6baed6', '#08306b'],
                              vmin=0, vmax=max_r, caption='Total Records')

    folium.GeoJson(
        h3_rich.__geo_interface__, name='H3 Grid Summary',
        style_function=lambda f: style_hex(f, 'Total_Records', cmap),
        tooltip=folium.GeoJsonTooltip(
            fields=['h3_index', 'Total_Records', 'Species_Richness',
                     'Trend_Direction', 'Years_w__Data'],
            aliases=['Cell', 'Records', 'Richness', 'Trend', 'Years']),
    ).add_to(m)
    cmap.add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(BASE, 'Workshop_H3_Map.html')
    m.save(path)
    print(f"  Saved {path}")


def build_cooccurrence_map():
    """Workshop_CoOccurrence_Map.html"""
    gdf_occ = gpd.read_file(MASTER_GPKG, layer='reeps_occurrences')
    reeps = gdf_occ[gdf_occ['Status'] == 'REEPS']
    h3_all = gpd.read_file(GRID_GPKG, layer='h3_all_cells')

    center = [h3_all.geometry.centroid.y.mean(), h3_all.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    # Color by number of species pairs
    occ_cells = reeps.groupby('h3_index')['Species'].nunique().reset_index()
    occ_cells.columns = ['h3_index', 'n_species']
    occ_cells['n_pairs'] = occ_cells['n_species'] * (occ_cells['n_species'] - 1) / 2

    h3_cooc = h3_all.merge(occ_cells, on='h3_index', how='left')
    h3_cooc['n_pairs'] = h3_cooc['n_pairs'].fillna(0)

    max_p = max(h3_cooc['n_pairs'].max(), 1)
    cmap = cm.LinearColormap(['#fff7ec', '#fc8d59', '#7f0000'],
                              vmin=0, vmax=max_p, caption='Species Pairs')

    folium.GeoJson(
        h3_cooc.__geo_interface__, name='Co-occurrence',
        style_function=lambda f: style_hex(f, 'n_pairs', cmap),
        tooltip=folium.GeoJsonTooltip(
            fields=['h3_index', 'n_species', 'n_pairs'],
            aliases=['Cell', 'Species', 'Pairs']),
    ).add_to(m)
    cmap.add_to(m)

    folium.LayerControl().add_to(m)
    path = os.path.join(BASE, 'Workshop_CoOccurrence_Map.html')
    m.save(path)
    print(f"  Saved {path}")


def generate_validation_csvs():
    """Regenerate key validation CSV files."""
    gdf_occ = gpd.read_file(MASTER_GPKG, layer='reeps_occurrences')
    reeps = gdf_occ[gdf_occ['Status'] == 'REEPS']
    reeps = reeps[reeps['Species'] != 'Prionodon linsang']

    # 1. chao1_per_period.csv
    years = sorted(reeps['Year'].dropna().unique())
    rows = []
    for y in years:
        yr = reeps[reeps['Year'] == y]
        counts = Counter(yr['Species'])
        N = sum(counts.values())
        S = len(counts)
        f1 = sum(1 for v in counts.values() if v == 1)
        f2 = sum(1 for v in counts.values() if v == 2)
        chao1 = S + (f1**2)/(2*f2) if f2 > 0 else (S + f1*(f1-1)/2 if f1 > 0 else S)
        coverage = 1 - f1/N if N > 0 else 0
        rows.append({
            'Year': int(y), 'N': N, 'S_obs': S, 'f1': f1, 'f2': f2,
            'Chao1': round(chao1, 2),
            'Completeness_pct': round(S/chao1*100, 1) if chao1 > 0 else 100,
            'Coverage': round(coverage, 4),
        })
    pd.DataFrame(rows).to_csv(os.path.join(BASE, 'chao1_per_period.csv'), index=False)
    print("  Saved chao1_per_period.csv")

    # 2. chao1_comparison.csv
    comp_rows = []
    for label, subset in [('Full', reeps),
                           ('Recent (2020+)', reeps[reeps['Year'] >= 2020])]:
        counts = Counter(subset['Species'])
        N = sum(counts.values())
        S = len(counts)
        f1 = sum(1 for v in counts.values() if v == 1)
        f2 = sum(1 for v in counts.values() if v == 2)
        chao1 = S + (f1**2)/(2*f2) if f2 > 0 else (S + f1*(f1-1)/2 if f1 > 0 else S)
        comp_rows.append({
            'Dataset': label, 'N': N, 'S_obs': S, 'f1': f1, 'f2': f2,
            'Chao1': round(chao1, 2),
            'Completeness_pct': round(S/chao1*100, 1) if chao1 > 0 else 100,
        })
    pd.DataFrame(comp_rows).to_csv(os.path.join(BASE, 'chao1_comparison.csv'), index=False)
    print("  Saved chao1_comparison.csv")

    # 3. spearman_trends.csv
    species_list = sorted(reeps['Species'].unique())
    trend_rows = []
    for sp in species_list:
        sp_df = reeps[reeps['Species'] == sp]
        sp_years = sorted(sp_df['Year'].dropna().unique())
        if len(sp_years) >= 3:
            yr_counts = [len(sp_df[sp_df['Year'] == y]) for y in sp_years]
            rho, pval = spearmanr(sp_years, yr_counts)
            trend_rows.append({
                'Species': sp, 'Common_Name': COMMON_NAMES.get(sp, ''),
                'N_Years': len(sp_years), 'Total_Records': len(sp_df),
                'Spearman_rho': round(rho, 4), 'p_value': round(pval, 4),
                'Trend': 'Increasing' if rho > 0.3 else ('Decreasing' if rho < -0.3 else 'Stable'),
            })
    pd.DataFrame(trend_rows).to_csv(os.path.join(BASE, 'spearman_trends.csv'), index=False)
    print("  Saved spearman_trends.csv")

    # 4. effort_per_period.csv
    effort_rows = []
    for y in years:
        yr = reeps[reeps['Year'] == y]
        effort_rows.append({
            'Year': int(y), 'Records': len(yr),
            'Species': yr['Species'].nunique(),
            'Cells': yr['h3_index'].nunique(),
            'Methods': yr['Survey_Method'].nunique(),
            'Method_List': ', '.join(sorted(yr['Survey_Method'].dropna().unique())),
        })
    pd.DataFrame(effort_rows).to_csv(os.path.join(BASE, 'effort_per_period.csv'), index=False)
    print("  Saved effort_per_period.csv")

    # 5. cpi_correlation_matrix.csv
    h3_pri = gpd.read_file(GRID_GPKG, layer='h3_priority')
    if len(h3_pri) > 2:
        comp_cols = ['Richness', 'Diversity', 'Connectivity', 'Co_occurrence', 'Threatened']
        corr = h3_pri[comp_cols].corr(method='spearman')
        corr.to_csv(os.path.join(BASE, 'cpi_correlation_matrix.csv'))
        print("  Saved cpi_correlation_matrix.csv")


def main():
    print("=" * 60)
    print("Workshop Stage 3: Regenerate HTML maps and validation")
    print("=" * 60)

    print("\n--- HTML Maps ---")
    build_diversity_map()
    build_priority_map()
    build_connectivity_map()
    build_corridor_map()
    build_temporal_map()
    build_h3_map()
    build_cooccurrence_map()

    print("\n--- Validation CSVs ---")
    generate_validation_csvs()

    print("\nDone! All workshop maps and validation files generated.")


if __name__ == '__main__':
    main()
