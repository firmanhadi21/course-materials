# How-To Guides

Practical recipes for specific tasks. Each guide assumes you have already run the full pipeline at least once and have working output files.

---

## How to add new species observations

To add new records to the synthetic dataset, edit `generate_synthetic_data.py`.

**Add a new REEPS target species:**

In the `REEPS_SPECIES` list (line 19), add a new dictionary entry:

```python
{'sci': 'Helarctos malayanus', 'common': 'Sun Bear', 'iucn': 'VU', 'weight': 0.05},
```

The `weight` controls relative abundance. Higher weight = more records generated for this species. Weights do not need to sum to 1.0 -- they are used by `random.choices()` as relative probabilities.

**Add a new non-REEPS bycatch species:**

In the `NON_REEPS` list (line 34), add:

```python
{'sci': 'Muntiacus muntjak', 'common': 'Indian Muntjac'},
```

After editing, re-run the full pipeline:

```bash
python generate_synthetic_data.py
python workshop_stage1_master_gpkg.py
python workshop_stage2_grid_analyses.py
python workshop_stage3_maps_validation.py
python workshop_stage4_figures.py
```

**Important:** If the new species should be included in REEPS analysis, also add it to the `REEPS_SPECIES` set in `workshop_stage2_grid_analyses.py` (line 22). If it is threatened (CR, EN, or VU), also add it to the `THREATENED_SPECIES` set (line 31).

---

## How to change H3 resolution

The H3 resolution controls hexagonal cell size. The default is resolution 8 (average edge length ~460 m, area ~0.74 km2).

Edit `workshop_stage1_master_gpkg.py`, line 73:

```python
H3_RES = 8   # Change to 7 for larger cells (~5.2 km2) or 9 for smaller (~0.10 km2)
```

Common resolutions for ecological analysis:

| Resolution | Avg Edge Length | Avg Area | Use Case |
|-----------|----------------|----------|----------|
| 7 | ~1.2 km | ~5.2 km2 | Landscape-scale, sparse data |
| 8 | ~460 m | ~0.74 km2 | Default, suitable for UCPS area |
| 9 | ~174 m | ~0.10 km2 | Fine-grained, dense camera trap data |

After changing resolution, re-run from stage 1 onward:

```bash
python workshop_stage1_master_gpkg.py
python workshop_stage2_grid_analyses.py
python workshop_stage3_maps_validation.py
python workshop_stage4_figures.py
```

**Note:** Higher resolution = more cells = longer computation. Resolution 9 on the UCPS AOI may produce 800+ cells. Lower resolution = fewer cells = may merge distinct habitats into single cells.

---

## How to modify CPI weights

The Composite Priority Index uses 6 weighted components. To change the weights, edit `workshop_stage2_grid_analyses.py`, line 262-265:

```python
weights = {
    'Richness': 0.177, 'Connectivity': 0.181, 'Threatened': 0.169,
    'Diversity': 0.180, 'Co_occurrence': 0.180, 'Permeability': 0.113
}
```

**Rules:**
- Weights should sum to 1.0 (or close to it -- the index is a weighted sum of 0-1 normalized values)
- All 6 components must be present
- Permeability is computed as 100 minus the mean resistance value from the PlanetScope 3m land cover classification (PS_Final_11class_Hierarchical.tif)

**Example: emphasize threatened species:**

```python
weights = {
    'Richness': 0.15, 'Connectivity': 0.10, 'Threatened': 0.40,
    'Diversity': 0.10, 'Co_occurrence': 0.10, 'Permeability': 0.15
}
```

After editing, re-run from stage 2:

```bash
python workshop_stage2_grid_analyses.py
python workshop_stage3_maps_validation.py
python workshop_stage4_figures.py
```

To see the effect, compare the before/after priority tier maps or check how `Tier` distribution changes in the console output.

---

## How to add a new survey year

**Step 1:** In `generate_synthetic_data.py`, add the year to `SURVEY_YEARS` (line 46):

```python
SURVEY_YEARS = [
    (2009, 0.002), (2012, 0.02), (2014, 0.06), (2017, 0.25),
    (2018, 0.005), (2020, 0.25), (2022, 0.20), (2024, 0.10),
    (2025, 0.07), (2026, 0.04), (2027, 0.08),  # <-- new year
]
```

The second value is the relative weight -- higher = more records generated for that year.

**Step 2:** Add a data source label in `SOURCES` (line 63):

```python
SOURCES = {
    ...
    2027: 'BMP 2027',  # <-- new entry
}
```

**Step 3:** If the new year has specific months, add to `MONTHS_BY_YEAR` (line 71):

```python
MONTHS_BY_YEAR = {
    2025: [6, 7, 8, 9, 10, 11, 12],
    2026: [1, 2],
    2027: [3, 4, 5],  # <-- new entry
}
```

Then re-run the full pipeline from stage 1.

---

## How to customize map colors

### Priority tier colors

In `workshop_stage3_maps_validation.py`, the `build_priority_map` function (line 95) defines tier colors:

```python
tier_colors = {'CRITICAL': '#bd0026', 'HIGH': '#fd8d3c',
               'MEDIUM': '#fecc5c', 'LOW': '#ffffb2'}
```

Change any hex color code. For example, to use blues instead:

```python
tier_colors = {'CRITICAL': '#08306b', 'HIGH': '#2171b5',
               'MEDIUM': '#6baed6', 'LOW': '#c6dbef'}
```

### Diversity map colormap

In the same file, `build_diversity_map` (line 57) uses a 3-color ramp:

```python
cmap_r = cm.LinearColormap(['#ffffcc', '#fd8d3c', '#bd0026'],
                            vmin=0, vmax=max_r, caption='Species Richness')
```

Replace the color list with any valid hex colors.

### Static figure colormaps

In `workshop_stage4_figures.py`, matplotlib named colormaps are used:
- Species richness: `cmap='YlOrRd'` (line 44)
- Shannon entropy: `cmap='viridis'` (line 74)

Change to any matplotlib colormap name: `'plasma'`, `'inferno'`, `'RdYlGn'`, `'coolwarm'`, etc.

Re-run stage 3 (maps) and/or stage 4 (figures) after changes.

---

## How to export data for QGIS

The GeoPackage files are natively compatible with QGIS. Simply open them:

**In QGIS:**
1. Layer > Add Layer > Add Vector Layer
2. Browse to `Workshop_GridAnalyses.gpkg`
3. Select the layers you want (e.g., `h3_priority`, `h3_diversity`)
4. Click Add

**Export a single layer to Shapefile:**

```python
import geopandas as gpd

gdf = gpd.read_file('Workshop_GridAnalyses.gpkg', layer='h3_priority')
gdf.to_file('h3_priority.shp')
```

**Export to GeoJSON (for web mapping):**

```python
gdf.to_file('h3_priority.geojson', driver='GeoJSON')
```

**Export attribute table only (no geometry) to CSV:**

```python
gdf.drop(columns='geometry').to_csv('h3_priority_attributes.csv', index=False)
```

---

## How to run the full pipeline in one command

```bash
python run_workshop.py
```

This runs all 5 stages in order: generate data, build master GPKG, build grid analyses, create maps and CSVs, create figures. If any stage fails, the pipeline stops and reports which step failed.

To run individual stages (e.g., after modifying only stage 2 parameters):

```bash
python workshop_stage2_grid_analyses.py
python workshop_stage3_maps_validation.py
python workshop_stage4_figures.py
```

You only need to re-run stages downstream of your change. The dependency chain is:

```
generate_synthetic_data.py
  --> workshop_stage1_master_gpkg.py
       --> workshop_stage2_grid_analyses.py
            --> workshop_stage3_maps_validation.py
            --> workshop_stage4_figures.py
```

---

## How to interpret Chao1 results

Open `chao1_per_period.csv` or the `h3_chao1` layer in the GeoPackage. Key columns:

| Column | What it means |
|--------|---------------|
| `S_obs` | Species actually observed |
| `Chao1_Estimate` | Estimated true richness (always >= S_obs) |
| `Completeness_pct` | S_obs / Chao1 * 100 |
| `Singletons_f1` | Species seen exactly once |
| `Doubletons_f2` | Species seen exactly twice |

**Interpretation rules:**
- **Completeness >= 90%**: Survey effort is likely adequate. Most species present were detected.
- **Completeness 70-90%**: Moderate coverage. Some species likely missed. Consider additional surveys.
- **Completeness < 70%**: Poor coverage. The observed richness substantially underestimates true richness.
- **High singletons, low doubletons**: The Chao1 estimate may be unstable. More sampling needed.
- **f1 = 0**: Every species was seen at least twice. Chao1 = S_obs. High confidence.

**Per-cell vs. per-period:** The `h3_chao1` layer gives cell-level estimates. Cells with few records (< 5) will have unreliable Chao1 values. The `chao1_per_period.csv` gives year-level estimates for the entire study area.

---

## How to identify priority cells for field surveys

Combine CPI priority with survey gap analysis:

**High-priority occupied cells needing re-survey:**
Look at the `h3_master_occupied` layer. Filter for cells where:
- `Tier` = "CRITICAL" or "HIGH"
- `Completeness_pct` < 80 (Chao1 suggests missing species)
- `Years_w__Data` < 3 (sparse temporal coverage)

**High-priority unoccupied cells to survey:**
Look at the `h3_survey_gaps` layer. Filter for:
- `Survey_Priority` = "High"
- `Corridor_Candidate` = 1 (connects occupied patches)
- `K1_Nbr_Richness` > 5 (surrounded by species-rich cells)

**In Python:**

```python
import geopandas as gpd

master = gpd.read_file('Workshop_GridAnalyses.gpkg', layer='h3_master_occupied')

# Under-surveyed high-priority cells
targets = master[
    (master['Tier'].isin(['CRITICAL', 'HIGH'])) &
    (master['Completeness_pct'] < 80)
]
print(targets[['h3_index', 'Tier', 'Completeness_pct', 'Species_Richness']])

# High-priority survey gaps
gaps = gpd.read_file('Workshop_GridAnalyses.gpkg', layer='h3_survey_gaps')
gap_targets = gaps[
    (gaps['Survey_Priority'] == 'High') &
    (gaps['Corridor_Candidate'] == 1)
]
print(gap_targets[['h3_index', 'Gap_Priority_Score', 'K1_Nbr_Richness']])
```

---

## How to add a species to the harmonization map

When real data contains variant spellings or local names for a species, add entries to the `SPECIES_MAP` dictionary in `workshop_stage1_master_gpkg.py` (line 18).

For example, if field records contain "Berang-berang" for Aonyx cinereus:

```python
SPECIES_MAP = {
    ...
    'Berang-berang': 'Aonyx cinereus',
    'berang-berang': 'Aonyx cinereus',
    ...
}
```

Each key is a raw input name; the value is the canonical scientific name. Add both capitalized and lowercase variants if data entry is inconsistent.

After editing, re-run from stage 1.

---

## How to change priority tier thresholds

In `workshop_stage2_grid_analyses.py`, line 272-273:

```python
df['Tier'] = df['Priority_Index'].apply(
    lambda x: 'CRITICAL' if x >= 0.7 else ('HIGH' if x >= 0.5 else ('MEDIUM' if x >= 0.3 else 'LOW')))
```

To add a fifth tier or change breakpoints:

```python
def assign_tier(x):
    if x >= 0.80: return 'CRITICAL'
    if x >= 0.60: return 'HIGH'
    if x >= 0.40: return 'MEDIUM'
    if x >= 0.20: return 'MODERATE'
    return 'LOW'

df['Tier'] = df['Priority_Index'].apply(assign_tier)
```

If you add a new tier name, also update the color dictionaries in `workshop_stage3_maps_validation.py` (`tier_colors` on line 95) and `workshop_stage4_figures.py` (`tier_order` and `tier_colors` on lines 98-99).
