# Technical Reference

Complete technical documentation for the biodiversity spatial analysis workshop pipeline.

---

## Pipeline Architecture

```
Workshop_Biodiversity_Database.xlsx    aoi.gpkg
         |                                |
         v                                |
  generate_synthetic_data.py              |
         |                                |
         v                                v
  workshop_stage1_master_gpkg.py  --------+
         |
         v
  Workshop_Master_Database.gpkg
    |- reeps_occurrences (points)
    |- reeps_h3_cells (hexagons)
    |- aoi_boundary (polygon)
         |
         v
  workshop_stage2_grid_analyses.py
         |
         v
  Workshop_GridAnalyses.gpkg
    |- h3_all_cells
    |- h3_richness_summary
    |- h3_diversity
    |- h3_priority
    |- h3_temporal_traj
    |- h3_temporal_matrix
    |- h3_chao1
    |- h3_corridor_gaps
    |- h3_isolation_risk
    |- h3_survey_gaps
    |- h3_master_occupied
    |- reeps_occurrences
    |- aoi_boundary
         |
         +---> workshop_stage3_maps_validation.py
         |        |- 7 HTML maps
         |        |- 5 validation CSVs
         |
         +---> workshop_stage4_figures.py
                  |- 3 PNG figures (300 DPI)
```

---

## Script Reference

### generate_synthetic_data.py

**Purpose:** Create a randomized synthetic biodiversity dataset with the same structure as the real REEPS database.

**Inputs:** None (all parameters are defined in the script).

**Outputs:**
- `Workshop_Biodiversity_Database.xlsx` -- Excel workbook with 3 sheets
- `workshop_synthetic_records.csv` -- flat CSV of all records

**Parameters:**

| Parameter | Value | Location |
|-----------|-------|----------|
| `n_reeps` | 480 | `generate_records()` call, line 250 |
| `n_bycatch` | 80 | `generate_records()` call, line 250 |
| Random seed | `None` (truly random) | line 14 |
| Lat center | -6.955 | line 53 |
| Lon center | 107.225 | line 54 |
| Lat spread | 0.025 | line 53 |
| Lon spread | 0.045 | line 54 |
| Hotspot probability | 70% | `random_location()`, line 79 |
| Lat clamp | -6.985 to -6.925 | line 89 |
| Lon clamp | 107.170 to 107.280 | line 90 |

**REEPS species list (11 species):**

| Scientific Name | Common Name | IUCN | Weight |
|-----------------|-------------|------|--------|
| Panthera pardus melas | Javan Leopard | CR | 0.13 |
| Nycticebus javanicus | Javan Slow Loris | CR | 0.19 |
| Manis javanica | Sunda Pangolin | CR | 0.10 |
| Hylobates moloch | Javan Gibbon | EN | 0.12 |
| Presbytis comata | Grizzled Langur | EN | 0.08 |
| Trachypithecus auratus | Javan Langur | VU | 0.12 |
| Nisaetus bartelsi | Bartels's Hawk-eagle | VU | 0.01 |
| Aonyx cinereus | Asian Small-clawed Otter | VU | 0.09 |
| Tragulus kanchil | Lesser Mouse-deer | LC | 0.02 |
| Prionailurus bengalensis | Leopard Cat | LC | 0.10 |
| Paradoxurus hermaphroditus | Common Palm Civet | LC | 0.04 |

**Non-REEPS bycatch species (4 species):**

| Scientific Name | Common Name |
|-----------------|-------------|
| Macaca fascicularis | Long-tailed Macaque |
| Sus scrofa | Wild Boar |
| Hystrix javanica | Javan Porcupine |
| Paguma larvata | Masked Palm Civet |

**Spatial hotspots (3 clusters):**

| Center Lat | Center Lon | Spread (sigma) | Description |
|-----------|-----------|----------------|-------------|
| -6.945 | 107.222 | 0.008 | Central-eastern richness hotspot |
| -6.955 | 107.210 | 0.012 | Western corridor |
| -6.965 | 107.235 | 0.010 | Southern patch |

**Survey years and weights:**

| Year | Weight | Source Label |
|------|--------|-------------|
| 2009 | 0.002 | Satwa Target 2020 |
| 2012 | 0.02 | Satwa Target 2020 |
| 2014 | 0.06 | Temuan Langsung |
| 2017 | 0.25 | Satwa Target 2020 |
| 2018 | 0.005 | Temuan Langsung |
| 2020 | 0.25 | Data Gabungan |
| 2022 | 0.20 | Species_coord_2022 |
| 2024 | 0.10 | reeps_record_24 |
| 2025 | 0.07 | BMP 2025-2026 |
| 2026 | 0.04 | BMP 2025-2026 |

**Survey methods and weights:**

| Method | Weight |
|--------|--------|
| Observation | 0.30 |
| Camera Trap | 0.10 |
| Sign (Feces) | 0.15 |
| Sign (Track) | 0.10 |
| Interview | 0.15 |
| Sign (Nest) | 0.05 |
| Unknown | 0.15 |

**Excel structure:**
- Sheet "Master Database": header row at row 4, data starts row 5
- Sheet "REEPS Database": filtered to Status='REEPS' only
- Sheet "Temporal Summary": empty placeholder

---

### workshop_stage1_master_gpkg.py

**Purpose:** Read the Excel database, harmonize species names, assign H3 cell indices, and write a GeoPackage with occurrence points and H3 cell summaries.

**Inputs:**
- `Workshop_Biodiversity_Database.xlsx` -- synthetic Excel database
- `aoi.gpkg` -- Area of Interest boundary polygon

**Outputs:**
- `Workshop_Master_Database.gpkg` with 3 layers

**Key parameters:**

| Parameter | Value | Line |
|-----------|-------|------|
| H3 resolution | 8 | 73 |
| Grid expansion | 2-ring buffer | 129 |
| Excel sheet | 'Master Database' | 78 |
| Excel header row | 3 (0-indexed) | 78 |

**Species harmonization map:** 50+ entries mapping variant names to canonical scientific names. Full map at lines 18-71. Categories of corrections:
- Misspellings (e.g., "Presbytys" -> "Presbytis")
- Local/Indonesian names (e.g., "macan" -> "Panthera pardus melas")
- Outdated taxonomy (e.g., "Nycticebus coucang" -> "Nycticebus javanicus")
- Case variants (e.g., "Lutung" and "lutung")

**H3 cell assignment:** Uses `h3.latlng_to_cell(lat, lon, resolution)` to assign each observation to a hexagonal cell. The grid is then expanded by 2 rings (`h3.grid_disk(cell, 2)`) around each data cell, filtered to cells that intersect the AOI.

**Output layers:**

#### reeps_occurrences
| Column | Type | Description |
|--------|------|-------------|
| No | int | Sequential record number |
| Source | str | Data source identifier |
| Species | str | Canonical scientific name (post-harmonization) |
| Common_Name | str | English common name |
| Status | str | "REEPS" or null |
| Year | float | Survey year |
| Month | float | Survey month (may be null) |
| Month_Year | str | Formatted date string |
| Survey_Method | str | Observation method |
| Location | str | Named location (usually null for synthetic) |
| Latitude | float | WGS84 latitude |
| Longitude | float | WGS84 longitude |
| h3_index | str | H3 cell identifier |
| geometry | Point | WGS84 point geometry |

#### reeps_h3_cells
| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| Lat | float | Cell centroid latitude |
| Lon | float | Cell centroid longitude |
| Total_Records | int | Observation count |
| Species_Richness | int | Unique species count |
| Species_List | str | Comma-separated species |
| Trend_Direction | str | "Increasing"/"Decreasing"/"Stable"/"No Data" |
| Trend_Slope | float | Linear regression slope |
| First_Year | float | Earliest survey year |
| Last_Year | float | Most recent survey year |
| Years_w__Data | int | Number of years with observations |
| Records_YYYY | int | Per-year record count (one column per year) |
| geometry | Polygon | H3 hexagon boundary |

#### aoi_boundary
Pass-through of the input AOI polygon.

---

### workshop_stage2_grid_analyses.py

**Purpose:** Compute all spatial analysis layers from the master GeoPackage.

**Inputs:**
- `Workshop_Master_Database.gpkg` (layers: reeps_occurrences, reeps_h3_cells, aoi_boundary)

**Outputs:**
- `Workshop_GridAnalyses.gpkg` with 13 layers

**Species filtering:**
- Only REEPS species are used (set of 15 canonical names, line 22-29)
- Prionodon linsang is explicitly excluded (line 20)
- 7 species are classified as threatened (line 31-36): P. pardus melas, N. javanicus, M. javanica, H. moloch, P. comata, N. bartelsi, P. bengalensis

**Output layers:**

#### h3_all_cells
Base grid with connectivity metrics for all cells.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| lat, lon | float | Cell centroid |
| Cell_Type | str | "Occupied" or "Empty" |
| Occupied | int | 1 or 0 |
| Patch_ID | int | Connected component ID (0 for empty) |
| Occ__Nbrs | int | Occupied K1 (immediate) neighbors |
| K2_Occ__Nbrs | int | Occupied cells within 2 rings |
| Sp__Reachable_K2 | int | Unique species reachable within K2 |
| Eco__Score | int | Connectivity score: K1*2 + K2 |

#### h3_richness_summary
Per-cell richness with temporal record breakdown.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| lat, lon | float | Cell centroid |
| Total_Records | int | Total observation count |
| Species_Richness | int | Unique species count |
| Species_List | str | Comma-separated species names |
| Trend_Direction | str | Richness trend over time |
| Trend_Slope | float | Linear regression slope of richness vs year |
| First_Year, Last_Year | float | Temporal bounds |
| Years_w__Data | int | Count of years with observations |
| Records_YYYY | int | Per-year record count |

#### h3_diversity
Diversity indices for occupied cells only.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| lat, lon | float | Cell centroid |
| Richness__S | int | Species count (S) |
| Records__N | int | Total records (N) |
| Shannon__H | float | Shannon entropy H' |
| Simpson__D | float | Simpson diversity index D |
| Pielou__J | float | Pielou evenness index J |
| Berger_Parker__BP | float | Berger-Parker dominance |
| Dominant_Species | str | Most abundant species |
| Common_Name | str | Common name of dominant |
| Diversity_Trend | str | Trend direction |
| Trend_Slope | float | Slope of richness over time |

#### h3_priority
Composite Priority Index (CPI) -- 6 components.

| Column | Type | Description |
|--------|------|-------------|
| Rank | int | Ordinal rank (1 = highest priority) |
| h3_index | str | H3 cell identifier |
| Tier | str | CRITICAL / HIGH / MEDIUM / LOW |
| Richness | int | Raw species count |
| Diversity | float | Shannon H' value |
| Connectivity | int | Occupied K1 neighbors |
| Co_occurrence | float | Species pair count: S*(S-1)/2 |
| Threatened | int | Count of CR+EN+VU species |
| Permeability | float | Habitat permeability (100 - mean resistance) |
| Priority_Index | float | Weighted composite score (0-1) |
| lat, lon | float | Cell centroid |

**CPI weights (line 262-265):**

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Richness | 0.177 | Species count per cell |
| Connectivity | 0.181 | Number of occupied H3 neighbors |
| Threatened | 0.169 | IUCN CR+EN+VU species count |
| Diversity | 0.180 | Shannon entropy (evenness) |
| Co_occurrence | 0.180 | Number of species pairs |
| Permeability | 0.113 | Habitat permeability from land cover classification |

**Priority tier thresholds:**

| Tier | CPI Range |
|------|-----------|
| CRITICAL | >= 0.70 |
| HIGH | >= 0.50 |
| MEDIUM | >= 0.30 |
| LOW | < 0.30 |

#### h3_temporal_traj
Temporal trajectory analysis.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| Status | str | "Persistent" (>=3 yrs) / "Recent" (last >=2020) / "Historical" / "Never Detected" |
| First_Detected | float | Earliest detection year |
| Last_Detected | float | Most recent detection year |
| Years_Detected | int | Number of years with detections |
| Total_Species_Ever | int | All species ever recorded |
| Total_Colonisations | int | Species appearances between consecutive surveys |
| Total_Extinctions | int | Species disappearances between consecutive surveys |
| Mean_Beta_Temporal | float | Average temporal turnover (Jaccard-based) |
| Richness_Trend | str | "Increasing"/"Decreasing"/"Stable"/"No Data" |
| lat, lon | float | Cell centroid |
| Rich_YYYY | int | Species richness per year |

#### h3_temporal_matrix
Detailed per-year records and species matrix.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| lat, lon | float | Cell centroid |
| YYYY_Records | int | Record count for year YYYY |
| YYYY_Species | int | Species count for year YYYY |
| Total_Records | int | Sum of all records |
| Years_With_Data | int | Number of years with any records |
| Status | str | "Occupied" or "Empty" |

#### h3_chao1
Chao1 nonparametric richness estimator.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| Total_Records | int | N (total individuals/records) |
| S_obs | int | Observed species count |
| Singletons_f1 | int | Species seen exactly once |
| Doubletons_f2 | int | Species seen exactly twice |
| Chao1_Estimate | float | Estimated true richness |
| Completeness_pct | float | S_obs / Chao1 * 100 |
| Undetected_est | float | Chao1 - S_obs |
| Years_Surveyed | int | Number of survey years |
| Survey_Years | str | Comma-separated year list |
| lat, lon | float | Cell centroid |

#### h3_corridor_gaps
Stepping stone / corridor analysis for unoccupied cells.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| lat, lon | float | Cell centroid |
| Betweenness | float | Graph betweenness centrality |
| K1_Occ_Neighbors | int | Adjacent occupied cells |
| K2_Richness_Reachable | int | Sum of richness in K2 occupied neighbors |
| Dist_to_Nearest_Occ | int | Ring distance to nearest occupied cell (-1 if none within 3 rings) |
| Is_Corridor_Candidate | int | 1 if >= 2 occupied K1 neighbors |
| Connects_Priority_Cells | int | Reserved (currently 0) |
| Stepping_Stone_Score | float | Composite: K1*3 + K2_richness*0.5 + betweenness*10 |
| Corridor_Tier | str | "High" (>5) / "Medium" (>2) / "Low" |

#### h3_isolation_risk
Isolation risk assessment for occupied cells.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| Priority_Index | float | CPI score |
| Priority_Tier | str | CPI tier |
| Direct_Occ_Neighbors | int | Adjacent occupied cells |
| Gap_Neighbors | int | Adjacent empty cells |
| Extra_Occ_via_Gaps | int | Occupied cells reachable through one gap cell |
| Total_Connectivity | int | Direct + extra via gaps |
| Species_Richness | int | Species in this cell |
| lat, lon | float | Cell centroid |

#### h3_survey_gaps
Survey priority ranking for unoccupied cells.

| Column | Type | Description |
|--------|------|-------------|
| h3_index | str | H3 cell identifier |
| lat, lon | float | Cell centroid |
| K1_Occ_Neighbors | int | Adjacent occupied cells |
| K2_Occ_Neighbors | int | Occupied cells within 2 rings |
| Dist_to_Nearest_Occ_km | int | Ring distance to nearest occupied cell |
| K1_Nbr_Richness | int | Sum of K1 neighbor richness |
| K2_Nbr_Richness | int | Sum of K2 neighbor richness |
| Corridor_Candidate | int | 1 if >= 2 occupied K1 neighbors |
| Gap_Priority_Score | float | K1_rich*2 + K2_rich*0.5 + corridor_bonus(10) |
| Survey_Priority | str | "High" (>10) / "Medium" (>3) / "Low" |

#### h3_master_occupied
Merged view of all metrics for occupied cells. Combines columns from h3_richness_summary, h3_diversity, h3_priority, h3_temporal_traj, h3_chao1, and h3_isolation_risk via left joins on h3_index. Columns with name collisions receive suffixes (`_pri`, `_temp`, `_chao`, `_iso`).

#### reeps_occurrences
Copy of REEPS-filtered occurrence records from the master GPKG (excluding Prionodon linsang).

#### aoi_boundary
Pass-through of the AOI polygon.

---

### workshop_stage3_maps_validation.py

**Purpose:** Generate interactive Folium HTML maps and validation CSV files.

**Inputs:**
- `Workshop_GridAnalyses.gpkg`
- `Workshop_Master_Database.gpkg`

**Outputs:**

#### HTML Maps (7 files)

| File | Layers | Color Scheme |
|------|--------|-------------|
| `Workshop_Diversity_Map.html` | Species richness (h3_diversity), empty cells | Yellow-orange-red ramp by richness |
| `Workshop_Priority_SurveyGap_Map.html` | Priority tiers (h3_priority), survey gaps (h3_survey_gaps, hidden by default) | Red/orange/yellow/pale by tier; red/orange/gray by gap priority |
| `Workshop_Connectivity_Map.html` | Eco score (h3_all_cells) | Green gradient by Eco__Score |
| `Workshop_Corridor_Map.html` | Corridor gaps (h3_corridor_gaps), isolation risk (h3_isolation_risk, hidden) | Red/orange/gray by corridor tier; blue by connectivity |
| `Workshop_Temporal_Turnover_Map.html` | Temporal trajectory (h3_temporal_traj), undetected (hidden) | Green/yellow/red by richness trend |
| `Workshop_H3_Map.html` | Record density (h3_richness_summary) | Blue gradient by Total_Records |
| `Workshop_CoOccurrence_Map.html` | Species pair count (derived from occurrences) | Orange-red gradient by pair count |

All maps use CartoDB positron basemap, zoom level 13, with `folium.LayerControl()` for toggling layers.

#### Validation CSVs (5 files)

| File | Content |
|------|---------|
| `chao1_per_period.csv` | Chao1 estimate per survey year |
| `chao1_comparison.csv` | Full dataset vs. recent (2020+) comparison |
| `spearman_trends.csv` | Spearman rho trend per species (requires >= 3 years) |
| `effort_per_period.csv` | Records, species, cells, methods per year |
| `cpi_correlation_matrix.csv` | Spearman correlation matrix of 6 CPI components |

---

### workshop_stage4_figures.py

**Purpose:** Generate static matplotlib figures at 300 DPI for publication.

**Inputs:**
- `Workshop_GridAnalyses.gpkg` (layers: h3_all_cells, h3_diversity, h3_priority)
- `Workshop_Master_Database.gpkg` (layer: aoi_boundary)

**Outputs (in `figures/` subdirectory):**

| File | Content | Colormap |
|------|---------|----------|
| `workshop_species_richness.png` | H3 cells colored by Richness__S | YlOrRd |
| `workshop_shannon_entropy.png` | H3 cells colored by Shannon__H | viridis |
| `workshop_priority_tiers.png` | H3 cells colored by Tier (categorical) | Custom 4-color |

All figures: 10x10 inches, 300 DPI, white background, AOI boundary overlay, axis labels.

**Priority tier color scheme:**
- LOW: #ffffb2 (pale yellow)
- MEDIUM: #fecc5c (yellow-orange)
- HIGH: #fd8d3c (orange)
- CRITICAL: #bd0026 (dark red)

---

### run_workshop.py

**Purpose:** Sequential runner for all 5 pipeline stages.

**Execution order:**
1. `generate_synthetic_data.py` -- create Excel + CSV
2. `workshop_stage1_master_gpkg.py` -- build master GeoPackage
3. `workshop_stage2_grid_analyses.py` -- compute analysis layers
4. `workshop_stage3_maps_validation.py` -- generate maps + CSVs
5. `workshop_stage4_figures.py` -- generate static figures

Stops on first failure. Reports per-step and total elapsed time.

---

## Formulas

### Shannon Entropy

```
H' = -SUM(p_i * ln(p_i))  for i = 1..S
```

Where `p_i = n_i / N`, `n_i` is the count of species i, `N` is total records, `S` is species count. Higher H' = more diverse. Maximum H' = ln(S) when all species equally abundant.

### Simpson Diversity Index

```
D = 1 - SUM(p_i^2)  for i = 1..S
```

Probability that two randomly chosen individuals belong to different species. Range: 0 (one species dominates) to ~1 (many equally abundant species).

### Pielou Evenness

```
J = H' / ln(S)
```

How evenly individuals are distributed among species. Range: 0 (one species dominates) to 1 (perfectly even). Undefined when S = 1 (set to 0 in implementation).

### Berger-Parker Dominance

```
d = max(n_i) / N
```

Proportion of the most abundant species. Higher = more dominated by one species.

### Chao1 Richness Estimator

```
Standard case (f2 > 0):
  Chao1 = S_obs + f1^2 / (2 * f2)

Bias-corrected case (f2 = 0):
  Chao1 = S_obs + f1 * (f1 - 1) / 2

Complete case (f1 = 0):
  Chao1 = S_obs
```

Where `S_obs` = observed species count, `f1` = singletons (species seen once), `f2` = doubletons (species seen twice).

Completeness: `S_obs / Chao1 * 100` (percent of estimated true richness observed).

### Composite Priority Index (CPI)

```
CPI = SUM(w_k * x_k_norm)  for k = 1..6
```

Each component is min-max normalized to [0, 1]:
```
x_norm = (x - x_min) / (x_max - x_min)
```

**Component weights (PCA-derived):**

| Component | Symbol | Weight | Raw metric |
|-----------|--------|--------|------------|
| Richness | w_rich | 0.177 | Species count per cell |
| Connectivity | w_conn | 0.181 | Occupied K1 neighbors |
| Threatened | w_thr | 0.169 | Count of CR+EN+VU species |
| Diversity | w_div | 0.180 | Shannon H' |
| Co-occurrence | w_cooc | 0.180 | S*(S-1)/2 species pairs |
| Permeability | w_perm | 0.113 | 100 - mean resistance from land cover |

**Permeability computation:**

Permeability per H3 cell is computed as `100 - mean_resistance`, where mean resistance is the zonal mean of the resistance surface derived from PlanetScope 3m land cover classification (`PS_Final_11class_Hierarchical.tif`, 11 classes):

| Land Cover Class | Resistance Value |
|-----------------|-----------------|
| Natural Forest | 1 |
| Production Forest | 5 |
| Agroforest | 5 |
| Young Regeneration | 10 |
| Sparse Vegetation | 30 |
| Crop Cultivation | 40 |
| Paddy | 50 |
| Bareland | 80 |
| Waterbody | 90 |
| Built-up | 100 |

Higher permeability values indicate cells dominated by natural or production forest (low resistance), while lower values indicate cells with built-up areas or bareland (high resistance).

**Priority tiers:**

| Tier | CPI Range | Interpretation |
|------|-----------|----------------|
| CRITICAL | >= 0.70 | Highest conservation value |
| HIGH | >= 0.50 | Important habitat |
| MEDIUM | >= 0.30 | Moderate value |
| LOW | < 0.30 | Lower priority |

### Temporal Beta-Diversity (Turnover)

```
beta = (gained + lost) / |union|
```

Computed between consecutive survey years for each cell. `gained` = species in current year not in previous; `lost` = species in previous not in current; `union` = all species in both years combined.

### Stepping Stone Score

```
score = K1_occ * 3 + K2_richness * 0.5 + betweenness * 10
```

Where `K1_occ` = number of occupied immediate neighbors, `K2_richness` = sum of species richness in K2 occupied cells, `betweenness` = graph betweenness centrality of the cell in the H3 adjacency network.

### Survey Gap Priority Score

```
score = K1_richness * 2 + K2_richness * 0.5 + corridor_bonus
```

Where `corridor_bonus` = 10 if the cell has >= 2 occupied K1 neighbors, else 0.

### Ecological Connectivity Score

```
Eco_Score = K1_occupied * 2 + K2_occupied
```

Simple weighted sum of occupied neighbors at K1 (immediate) and K2 (two-ring) distances.

### Richness Trend

Linear regression (numpy `polyfit` degree 1) of richness vs. year, using only years with data:
- Slope > 0.01: "Increasing"
- Slope < -0.01: "Decreasing"
- Otherwise: "Stable"

### Spearman Rank Correlation (Validation)

```
rho, p = spearmanr(years, counts)
```

Computed per species across survey years (minimum 3 years required). Used in `spearman_trends.csv`. Trend classification: rho > 0.3 = "Increasing", rho < -0.3 = "Decreasing", else "Stable".

---

## Output Files: Complete Inventory

### GeoPackages

| File | Layers | Source Stage |
|------|--------|-------------|
| `Workshop_Master_Database.gpkg` | reeps_occurrences, reeps_h3_cells, aoi_boundary | Stage 1 |
| `Workshop_GridAnalyses.gpkg` | 13 layers (see above) | Stage 2 |

### HTML Maps

| File | Source Stage |
|------|-------------|
| `Workshop_Diversity_Map.html` | Stage 3 |
| `Workshop_Priority_SurveyGap_Map.html` | Stage 3 |
| `Workshop_Connectivity_Map.html` | Stage 3 |
| `Workshop_Corridor_Map.html` | Stage 3 |
| `Workshop_Temporal_Turnover_Map.html` | Stage 3 |
| `Workshop_H3_Map.html` | Stage 3 |
| `Workshop_CoOccurrence_Map.html` | Stage 3 |

### Validation CSVs

| File | Source Stage |
|------|-------------|
| `chao1_per_period.csv` | Stage 3 |
| `chao1_comparison.csv` | Stage 3 |
| `spearman_trends.csv` | Stage 3 |
| `effort_per_period.csv` | Stage 3 |
| `cpi_correlation_matrix.csv` | Stage 3 |

### Static Figures

| File | Source Stage |
|------|-------------|
| `figures/workshop_species_richness.png` | Stage 4 |
| `figures/workshop_shannon_entropy.png` | Stage 4 |
| `figures/workshop_priority_tiers.png` | Stage 4 |

### Intermediate Data Files

| File | Source Stage |
|------|-------------|
| `Workshop_Biodiversity_Database.xlsx` | Data generation |
| `workshop_synthetic_records.csv` | Data generation |

---

## Coordinate Reference System

All spatial data uses **EPSG:4326** (WGS84 geographic coordinates). The AOI is reprojected to WGS84 if it has a different CRS.

## H3 Indexing

- Library: `h3` (Uber's H3 geospatial indexing system)
- Default resolution: 8
- H3 res-8 cell properties: ~460 m average edge length, ~0.74 km2 area
- Cell assignment: `h3.latlng_to_cell(lat, lon, 8)`
- Cell boundary: `h3.cell_to_boundary(cell)` returns list of (lat, lng) vertices
- Neighbors: `h3.grid_disk(cell, k)` returns all cells within k rings
- Distance: `h3.grid_distance(cell_a, cell_b)` returns ring distance
- Connected components: computed via NetworkX on the H3 adjacency graph
