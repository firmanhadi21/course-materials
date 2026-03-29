# Biodiversity Spatial Analysis Workshop

## A Step-by-Step Tutorial Using Synthetic Data

This tutorial walks you through the complete REEPS/UCPS biodiversity spatial analysis pipeline using randomized synthetic data. By the end, you will have generated a spatial database, computed diversity and priority indices, created interactive maps, and produced publication-quality figures -- all using the same code that processes real survey data.

**Total time: approximately 80 minutes**

---

## Prerequisites

### Software

- Python 3.10 or later
- The following pip packages:

```bash
pip install geopandas h3 networkx scipy folium openpyxl matplotlib shapely pyproj rasterio pandas numpy branca
```

### Files you need

All scripts must be in the `workshop/` directory:

| File | Purpose |
|------|---------|
| `generate_synthetic_data.py` | Creates ~560 random observation records |
| `workshop_stage1_master_gpkg.py` | Excel to GeoPackage with H3 indexing |
| `workshop_stage2_grid_analyses.py` | Diversity, CPI, corridors, temporal analysis |
| `workshop_stage3_maps_validation.py` | Folium HTML maps + validation CSVs |
| `workshop_stage4_figures.py` | Static matplotlib figures at 300 DPI |
| `run_workshop.py` | Master runner (all stages sequentially) |
| `aoi.gpkg` | Area of Interest boundary polygon |

### Verify your environment

```bash
cd /Users/macbook/Dropbox/Works/Cisokan/2026/Feb26/data/biodiv/workshop
python -c "import geopandas, h3, networkx, scipy, folium, openpyxl, matplotlib; print('All packages OK')"
```

You should see: `All packages OK`

---

## Part 1: Generate Your Dataset (10 min)

### What this step does

The `generate_synthetic_data.py` script creates a fully randomized biodiversity dataset that mirrors the structure of the real REEPS database. It generates approximately 560 records: 480 REEPS target species observations and 80 non-REEPS bycatch records.

### Run it

```bash
python generate_synthetic_data.py
```

### What you should see

```
============================================================
Generating Synthetic REEPS Dataset for Workshop
============================================================

Total records: 560
REEPS records: 480
Non-REEPS: 80

Species breakdown:
  Aonyx cinereus: 43
  Hylobates moloch: 58
  Manis javanica: 48
  ...

Year breakdown:
  2009: 1
  2012: 11
  2014: 34
  ...

Output files:
  .../Workshop_Biodiversity_Database.xlsx (560 total, 480 REEPS)
  .../workshop_synthetic_records.csv
```

The exact numbers will vary because the data is randomized each run.

### Understand the output

Open `workshop_synthetic_records.csv` in a spreadsheet or text editor. Each row is one observation with these columns:

| Column | Meaning | Example |
|--------|---------|---------|
| `No` | Sequential record number | 1, 2, 3... |
| `Source` | Data source identifier | "Satwa Target 2020", "reeps_record_24" |
| `Species` | Scientific name | "Panthera pardus melas" |
| `Common_Name` | English common name | "Javan Leopard" |
| `Status` | REEPS target or not | "REEPS" or blank |
| `Year` | Survey year | 2017 |
| `Month` | Survey month (if known) | 7 or blank |
| `Month_Year` | Formatted date string | "Jul-2025" |
| `Survey_Method` | How the observation was made | "Camera Trap", "Observation" |
| `Location` | Named location (unused in synthetic data) | blank |
| `Latitude` | WGS84 latitude | -6.955123 |
| `Longitude` | WGS84 longitude | 107.225456 |

The Excel file (`Workshop_Biodiversity_Database.xlsx`) contains two sheets:
- **Master Database** -- all 560 records (header on row 4, data from row 5)
- **REEPS Database** -- filtered to the 480 REEPS-only records

### Key things to notice

- There are 11 REEPS target species and 4 non-REEPS bycatch species
- IUCN threat levels: 3 CR, 2 EN, 3 VU, 3 LC
- Observations cluster around 3 spatial hotspots (70% chance) with the rest scattered across the AOI
- Recent years (2017, 2020, 2022) have more records, reflecting increasing survey effort
- Coordinates are bounded to approximately -6.925 to -6.985 latitude, 107.170 to 107.280 longitude

---

## Part 2: Build the Spatial Database (15 min)

### What this step does

The `workshop_stage1_master_gpkg.py` script reads the Excel file, harmonizes species names (correcting misspellings and local name variants), assigns each observation to an H3 hexagonal grid cell at resolution 8, and writes everything to a GeoPackage.

### Run it

```bash
python workshop_stage1_master_gpkg.py
```

### What you should see

```
============================================================
Workshop Stage 1: Build Workshop_Master_Database.gpkg
============================================================
Loaded AOI from .../aoi.gpkg
Loaded 560 records from Excel
Harmonized 0 species name variants
Unique species: 15
Assigned H3 res-8 indices to 560 records
Unique H3 cells with data: 45
Total H3 cells in AOI: 120 (data cells: 45, expanded via 2-ring)
Occupied cells: 45/120

Writing .../Workshop_Master_Database.gpkg...

Done! Workshop_Master_Database.gpkg built:
  reeps_occurrences: 560 records
  reeps_h3_cells: 120 cells
  aoi_boundary: 1 polygon(s)
```

Your exact cell counts will differ because locations are random.

### Understand the output

The GeoPackage `Workshop_Master_Database.gpkg` contains three layers:

**reeps_occurrences** -- Point geometries, one per observation record. Each point has the original fields plus `h3_index` (the H3 cell it falls in).

**reeps_h3_cells** -- Hexagonal polygon geometries covering the AOI. Columns include:
- `h3_index` -- unique H3 cell identifier
- `Total_Records` -- number of observations in this cell
- `Species_Richness` -- number of unique species
- `Trend_Direction` -- "Increasing", "Decreasing", "Stable", or "No Data"
- `Records_YYYY` -- record count per survey year

**aoi_boundary** -- The Area of Interest polygon.

### Verify in QGIS (optional)

Open `Workshop_Master_Database.gpkg` in QGIS. You should see:
1. The AOI boundary polygon
2. Hexagonal grid cells covering the AOI (some colored, some empty)
3. Point observations scattered within the hexagons

The H3 cell expansion uses a 2-ring buffer around data cells, then clips to the AOI boundary. This ensures analysis coverage extends slightly beyond the observed data.

### How species harmonization works

The script applies a mapping dictionary (line 18-71 of `workshop_stage1_master_gpkg.py`) that corrects:
- Misspellings: "Presbytys comata" becomes "Presbytis comata"
- Local names: "macan" becomes "Panthera pardus melas"
- Outdated taxonomy: "Nycticebus coucang" becomes "Nycticebus javanicus"
- Case variants: "Lutung" and "lutung" both become "Trachypithecus auratus"

With synthetic data, harmonization changes 0 names because the generator already uses canonical names. With real data, this step is critical.

---

## Part 3: Analyze Biodiversity Patterns (20 min)

### What this step does

The `workshop_stage2_grid_analyses.py` script computes all spatial analysis layers: diversity indices, the Composite Priority Index (CPI), temporal trajectories, Chao1 richness estimation, corridor/stepping stone analysis, and isolation risk.

### Run it

```bash
python workshop_stage2_grid_analyses.py
```

### What you should see

```
============================================================
Workshop Stage 2: Build Workshop_GridAnalyses.gpkg
============================================================
REEPS occurrences (excl. linsang): 480
All occurrences: 560

--- Building h3_all_cells ---
  120 cells

--- Building h3_richness_summary ---
  120 cells

--- Building h3_diversity ---
  45 occupied cells with diversity

--- Building h3_priority ---
  45 ranked cells
  Tiers: {'HIGH': 12, 'MEDIUM': 18, 'LOW': 10, 'CRITICAL': 5}

--- Building h3_temporal_traj ---
  120 cells

--- Building h3_temporal_matrix ---
  120 cells

--- Building h3_chao1 ---
  45 occupied cells

--- Building h3_corridor_gaps ---
  75 gap cells

--- Building h3_isolation_risk ---
  45 occupied cells

--- Building h3_survey_gaps ---
  75 gap cells

--- Building h3_master_occupied ---
  45 occupied cells

Writing .../Workshop_GridAnalyses.gpkg...
  Wrote h3_all_cells: 120 rows
  Wrote h3_richness_summary: 120 rows
  ...
```

### Understand each output layer

The GeoPackage `Workshop_GridAnalyses.gpkg` contains 13 layers. Here is what each one provides:

#### h3_all_cells
The base grid with connectivity metrics for every cell (occupied and empty).
- `Cell_Type`: "Occupied" or "Empty"
- `Occ__Nbrs`: number of occupied immediate neighbors (K1)
- `K2_Occ__Nbrs`: occupied cells within 2 rings
- `Sp__Reachable_K2`: species reachable within 2 hops
- `Eco__Score`: ecological connectivity score (K1*2 + K2)
- `Patch_ID`: connected component ID (0 for empty cells)

#### h3_richness_summary
Per-cell richness and temporal record counts for all cells.
- `Species_Richness`: number of unique species
- `Species_List`: comma-separated species names
- `Trend_Direction`: richness trend over time
- `Records_YYYY`: observation count per survey year

#### h3_diversity
Shannon, Simpson, Pielou, and Berger-Parker indices for occupied cells only.
- `Shannon__H`: Shannon entropy H' = -sum(pi * ln(pi))
- `Simpson__D`: Simpson diversity D = 1 - sum(pi^2)
- `Pielou__J`: Pielou evenness J = H' / ln(S)
- `Berger_Parker__BP`: dominance = max(ni) / N
- `Dominant_Species`: most abundant species in cell

#### h3_priority
The Composite Priority Index (CPI) with 6 components. This is the core conservation prioritization layer.
- `Richness`: raw species count
- `Diversity`: Shannon H' value
- `Connectivity`: occupied K1 neighbors
- `Co_occurrence`: number of species pairs (S*(S-1)/2)
- `Threatened`: count of CR+EN+VU species
- `Permeability`: habitat permeability (100 minus mean resistance from land cover)
- `Priority_Index`: weighted composite (0-1 scale)
- `Tier`: CRITICAL (>=0.70), HIGH (>=0.50), MEDIUM (>=0.30), LOW (<0.30)
- `Rank`: ordinal rank (1 = highest priority)

The CPI weights are: Richness 17.7%, Connectivity 18.1%, Threatened 16.9%, Diversity 18.0%, Co-occurrence 18.0%, Permeability 11.3%.

#### h3_temporal_traj
Temporal trajectory analysis per cell.
- `Status`: "Persistent" (>=3 years), "Recent" (last detected >=2020), "Historical", or "Never Detected"
- `Total_Colonisations`: species gained between consecutive surveys
- `Total_Extinctions`: species lost between consecutive surveys
- `Mean_Beta_Temporal`: average temporal beta-diversity (turnover)
- `Rich_YYYY`: richness per survey year

#### h3_temporal_matrix
Detailed records-and-species matrix per year per cell.
- `YYYY_Records`: record count for each year
- `YYYY_Species`: species count for each year
- `Years_With_Data`: how many years have any data

#### h3_chao1
Chao1 nonparametric richness estimator per occupied cell.
- `S_obs`: observed species count
- `Singletons_f1`: species seen exactly once
- `Doubletons_f2`: species seen exactly twice
- `Chao1_Estimate`: estimated true richness
- `Completeness_pct`: S_obs / Chao1 * 100
- `Undetected_est`: estimated missing species

#### h3_corridor_gaps
Stepping stone analysis for empty (unoccupied) cells.
- `Betweenness`: graph betweenness centrality
- `K1_Occ_Neighbors`: adjacent occupied cells
- `Is_Corridor_Candidate`: 1 if >= 2 occupied neighbors
- `Stepping_Stone_Score`: composite corridor value
- `Corridor_Tier`: "High", "Medium", or "Low"

#### h3_isolation_risk
Occupied cells at risk of ecological isolation.
- `Direct_Occ_Neighbors`: immediately adjacent occupied cells
- `Gap_Neighbors`: adjacent empty cells
- `Extra_Occ_via_Gaps`: occupied cells reachable through one gap
- `Total_Connectivity`: direct + gap-bridged connections

#### h3_survey_gaps
Unoccupied cells ranked by survey priority.
- `K1_Nbr_Richness`: total richness of adjacent occupied cells
- `Corridor_Candidate`: whether this gap connects occupied patches
- `Gap_Priority_Score`: composite survey priority
- `Survey_Priority`: "High", "Medium", or "Low"

#### h3_master_occupied
A merged view joining richness, diversity, priority, temporal, Chao1, and isolation data for all occupied cells. This is the "one-stop" layer for occupied cell analysis.

#### reeps_occurrences
Copy of the filtered REEPS occurrence points (excluding Prionodon linsang).

#### aoi_boundary
The Area of Interest polygon.

---

## Part 4: Create Interactive Maps (10 min)

### What this step does

The `workshop_stage3_maps_validation.py` script generates 7 interactive HTML maps using Folium and 5 validation CSV files.

### Run it

```bash
python workshop_stage3_maps_validation.py
```

### What you should see

```
============================================================
Workshop Stage 3: Regenerate HTML maps and validation
============================================================

--- HTML Maps ---
  Saved .../Workshop_Diversity_Map.html
  Saved .../Workshop_Priority_SurveyGap_Map.html
  Saved .../Workshop_Connectivity_Map.html
  Saved .../Workshop_Corridor_Map.html
  Saved .../Workshop_Temporal_Turnover_Map.html
  Saved .../Workshop_H3_Map.html
  Saved .../Workshop_CoOccurrence_Map.html

--- Validation CSVs ---
  Saved chao1_per_period.csv
  Saved chao1_comparison.csv
  Saved spearman_trends.csv
  Saved effort_per_period.csv
  Saved cpi_correlation_matrix.csv
```

### Explore the maps

Open each HTML file in a web browser. They are fully interactive -- you can zoom, pan, click cells, and toggle layers.

**Workshop_Diversity_Map.html** -- Species richness with a yellow-to-red color ramp. Hover over cells to see Shannon H', Simpson D, and the dominant species. Toggle the empty cells layer on/off.

**Workshop_Priority_SurveyGap_Map.html** -- CPI priority tiers (red = CRITICAL, orange = HIGH, yellow = MEDIUM, pale = LOW). Toggle the "Survey Gaps" layer to see which unoccupied cells should be surveyed next.

**Workshop_Connectivity_Map.html** -- Ecological connectivity score with a green gradient. Darker cells have more occupied neighbors. Isolated cells appear pale.

**Workshop_Corridor_Map.html** -- Stepping stone candidates (red = High priority corridor gaps). Toggle the "Isolation Risk" layer to see which occupied cells are poorly connected.

**Workshop_Temporal_Turnover_Map.html** -- Richness trend per cell: green = increasing, yellow = stable, red = decreasing. Toggle "Undetected" to see cells never surveyed.

**Workshop_H3_Map.html** -- Raw record density (total observation count per cell) with a blue gradient. Useful for identifying survey effort bias.

**Workshop_CoOccurrence_Map.html** -- Species pair count per cell. Cells with many co-occurring species appear dark red.

### Explore the validation CSVs

**chao1_per_period.csv** -- Chao1 richness estimate per survey year. Look for years where completeness is below 80% -- those years may have inadequate sampling.

**chao1_comparison.csv** -- Compares the full dataset against recent data (2020+). If recent completeness is lower, there may be species that have not been re-detected.

**spearman_trends.csv** -- Spearman rank correlation for each species' detection count over time. Positive rho = increasing detections; negative = decreasing.

**effort_per_period.csv** -- Records, species, cells surveyed, and methods per year. Watch for years with very few records -- trends from those years are unreliable.

**cpi_correlation_matrix.csv** -- Spearman correlation between the 6 CPI components. High correlation between Richness and Diversity is expected; low correlation between components validates that each captures distinct information.

---

## Part 5: Generate Publication Figures (10 min)

### What this step does

The `workshop_stage4_figures.py` script produces 3 static matplotlib figures at 300 DPI, suitable for reports and publications.

### Run it

```bash
python workshop_stage4_figures.py
```

### What you should see

```
============================================================
Workshop Stage 4: Generate static figures
============================================================

--- Generating figures ---
  Saved .../figures/workshop_species_richness.png
  Saved .../figures/workshop_shannon_entropy.png
  Saved .../figures/workshop_priority_tiers.png

Done! All figures saved to .../figures/
```

### Review the figures

Open the `figures/` directory. You will find three PNG files:

**workshop_species_richness.png** -- Hexagonal grid colored by species richness (S) using a YlOrRd (yellow-orange-red) colormap. Empty cells are light gray. The AOI boundary is drawn in dark gray.

**workshop_shannon_entropy.png** -- Same grid colored by Shannon entropy (H') using the viridis colormap. Higher entropy = more even species distribution.

**workshop_priority_tiers.png** -- Categorical map with four colors: pale yellow (LOW), yellow-orange (MEDIUM), orange (HIGH), dark red (CRITICAL). Includes a legend showing tier labels.

All figures include axis labels (Longitude, Latitude) and a title. They use a white background instead of a satellite basemap for clarity.

---

## Part 6: Interpret Results (15 min)

### Reading the Priority Map

Look at the priority tier map. Ask yourself:
- **Where are the CRITICAL cells?** These have the highest combination of richness, connectivity, threatened species, diversity, co-occurrence, and permeability.
- **Are CRITICAL cells clustered or scattered?** Clustered = core habitat; scattered = fragmented.
- **Do HIGH cells form a buffer around CRITICAL cells?** If so, the gradient suggests a functional core-buffer structure.

### Reading the Chao1 Results

Open `chao1_per_period.csv`. For each year:
- `Completeness_pct` close to 100% means you probably found most species present.
- Below 70% means many species were likely missed.
- `f1` (singletons) being high relative to `S_obs` suggests undersampling.

Open `chao1_comparison.csv`. Compare the Full dataset to Recent (2020+):
- If the recent completeness is much lower, species detected historically may have declined or the recent surveys are less thorough.

### Reading the Effort Table

Open `effort_per_period.csv`. Key questions:
- Which years had the most records? Years with few records (< 10) should be interpreted with caution.
- How many cells were covered per year? Low cell counts mean spatial conclusions are weak for that period.
- What methods were used? Camera traps detect different species than direct observation. Method changes over time can create apparent trends.

### Reading Spearman Trends

Open `spearman_trends.csv`:
- Species with positive `Spearman_rho` and `p_value < 0.05` show statistically significant increases.
- Negative rho with low p-value = significant decline.
- **Caution**: with synthetic data, trends are artifacts of the random generation. With real data, always consider whether apparent trends reflect actual population change or just changes in survey effort.

### Reading the Connectivity and Corridor Maps

In the corridor map:
- **High-tier corridor gaps** are empty cells that sit between occupied patches. Protecting these cells could maintain landscape connectivity.
- **Isolated occupied cells** (few neighbors) are vulnerable to local extinction if the surrounding habitat degrades.

### Quick sanity checks

1. Do the number of occupied cells in the grid match across layers? (h3_all_cells Occupied count should equal h3_diversity row count)
2. Does the sum of per-year records in `effort_per_period.csv` equal the total REEPS records?
3. Are any CRITICAL-tier cells also flagged as high isolation risk? Those are the most urgent conservation targets.

---

## Running the Full Pipeline in One Command

If you want to skip running each stage individually:

```bash
python run_workshop.py
```

This runs all 5 stages sequentially (generate data, build master GPKG, build grid analyses, create maps, create figures) and reports timing for each step. Total runtime is typically 30-90 seconds depending on your machine.

---

## Next Steps

- Read `HOWTO.md` for task-specific recipes (changing weights, adding species, etc.)
- Read `REFERENCE.md` for complete technical documentation of every layer and formula
- Read `EXPLANATION.md` for conceptual discussions on why these methods were chosen
- Try modifying CPI weights in `workshop_stage2_grid_analyses.py` (line 263) and rerunning stages 2-4 to see how priorities shift
