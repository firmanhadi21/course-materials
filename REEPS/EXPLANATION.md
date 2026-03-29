# Conceptual Guide

This document explains the reasoning behind the methods used in the biodiversity spatial analysis pipeline. It is meant to be read alongside the code, not as a substitute for it. If you want to know *what* the pipeline does, read `REFERENCE.md`. This document explains *why*.

---

## Why hexagonal grids?

Biodiversity data arrives as point observations: a GPS coordinate where someone saw an animal. To analyze spatial patterns, we need to aggregate these points into area units. The two common choices are square grids and hexagonal grids.

**Equal-area property.** H3 hexagons at a given resolution are approximately equal in area (within ~1% at resolution 8). Square grids are also equal-area when using a projected coordinate system, but H3 cells maintain this property on the globe without needing a local projection.

**Consistent neighbor relationships.** Every hexagon has exactly 6 neighbors at ring distance 1, and all 6 share an edge of equal length. In a square grid, a cell has 4 edge-neighbors and 4 corner-neighbors, and the corner distance is sqrt(2) times the edge distance. This ambiguity complicates connectivity analysis. With hexagons, "neighbor" has one unambiguous meaning.

**Reduced edge bias.** The perimeter-to-area ratio of a hexagon is lower than that of a square with the same area. This means fewer observations fall near cell boundaries, reducing edge effects when aggregating.

**Why H3 resolution 8?** At resolution 8, each cell has an average area of approximately 0.74 km2 (edge length ~460 m). This is appropriate for the UCPS study area because:
- The AOI spans roughly 6 x 10 km, yielding ~120 cells -- enough for meaningful spatial analysis without excessive computation.
- Camera trap spacing is typically 500-1000 m, so resolution 8 cells roughly correspond to one trap's detection zone.
- At resolution 7 (~5.2 km2), too many distinct habitats would merge into single cells. At resolution 9 (~0.10 km2), most cells would have zero or one observation, making diversity indices meaningless.

The choice of resolution is a trade-off between spatial detail and statistical power per cell. Resolution 8 balances these for the scale of the UCPS landscape.

---

## Why six CPI components?

The Composite Priority Index combines six dimensions of conservation value. Each component captures something that the others do not. The weights are derived from Principal Component Analysis (PCA) of REEPS monitoring data.

### Richness (17.7%)

The most intuitive measure: how many species live here. A cell with 8 species is more important than one with 2 species, all else being equal. However, richness alone is insufficient -- a cell with 8 common, widespread species may be less important than one with 3 threatened species.

### Connectivity (18.1%)

Counts the number of occupied immediate neighbors. A species-rich cell surrounded by other occupied cells is part of a functional habitat network. An equally rich but isolated cell is ecologically fragile -- if its population declines, there are no adjacent populations to provide recolonization. Connectivity reflects landscape-level viability, which richness alone cannot capture.

### Threatened species (16.9%)

Counts species classified as Critically Endangered, Endangered, or Vulnerable by IUCN. A cell hosting a Javan Leopard (CR) or Sunda Pangolin (CR) has outsized conservation importance regardless of its total richness. Threatened species receive their own weight because they are often rare (few records), which means they contribute little to richness-based metrics but represent disproportionate conservation value.

### Shannon diversity (18.0%)

Shannon entropy captures evenness -- not just how many species are present, but how evenly distributed their observations are. A cell with 5 species and one dominant (90% of records) is less functionally diverse than a cell with 5 equally abundant species. Evenness is partly an artifact of sampling: a cell surveyed intensively may appear more even simply because rare species had more chances to be detected.

### Co-occurrence (18.0%)

The number of species pairs present in a cell: S*(S-1)/2. This measures interaction potential -- cells where many species coexist may support ecological processes (pollination, seed dispersal, predator-prey regulation) that depend on multi-species presence. It amplifies the importance of species-rich cells non-linearly: going from 3 to 5 species doubles the pair count from 3 to 10.

### Why permeability as the 6th component? (11.3%)

Permeability measures how easily animals can move through a cell based on land cover. It is computed as `100 - mean_resistance` per H3 cell, where resistance values are derived from a PlanetScope 3m land cover classification (`PS_Final_11class_Hierarchical.tif`) with 11 classes:

| Land Cover Class | Resistance |
|-----------------|-----------|
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

PCA showed that permeability is uncorrelated with the 5 biodiversity metrics (|r| < 0.16), loading primarily on PC2 (which explains 18.1% of total variance). This confirms that permeability captures an independent habitat-quality dimension that the biodiversity survey metrics do not measure. A cell may have high richness but low permeability (e.g., a forest fragment surrounded by agriculture), or low richness but high permeability (e.g., intact forest that has not been surveyed). Including permeability in the CPI ensures that habitat structural quality influences prioritization alongside species-based metrics.

Permeability receives the lowest weight (11.3%) because it is a remotely-sensed physical attribute rather than a direct biodiversity observation. It complements rather than dominates the species-based components. Resistance values are also used separately in corridor gap analysis (`h3_corridor_gaps` layer) for stepping stone identification.

### Weight derivation

The weights were derived from Principal Component Analysis of REEPS monitoring data. The PCA identified how much variance each component explains independently. The 6 components load across multiple principal components, and the weights reflect their relative contributions to the total explained variance.

---

## Why Chao1 for completeness?

Every biodiversity survey misses species. The question is: how many did we miss? Chao1 is a nonparametric estimator that answers this using only the observed data -- no assumptions about species distributions are needed.

### The singleton/doubleton logic

Chao1 relies on a simple insight: the number of undetected species is related to the number of barely-detected species.

- **Singletons (f1):** Species seen exactly once. These are the "tip of the iceberg" -- if you only saw them once, there are probably others you never saw at all.
- **Doubletons (f2):** Species seen exactly twice. These anchor the estimate -- they tell you how many species are "hard to find but findable."

The formula:

```
Chao1 = S_obs + f1^2 / (2 * f2)
```

If there are many singletons and few doubletons, the estimate inflates dramatically -- this is correct, because many singletons with few doubletons means sampling was shallow. If there are no singletons (f1 = 0), then Chao1 = S_obs, meaning the survey likely found everything.

### What 100% completeness means and does not mean

`Completeness_pct = S_obs / Chao1 * 100`

- **100% completeness** means every species was seen at least twice. It does NOT mean every species in the landscape was found -- it means the estimator has no statistical evidence of missing species. With very small sample sizes, Chao1 can be artificially optimistic.
- **80% completeness** means the estimator predicts ~20% more species exist than were observed. This is a lower bound -- the true number of undetected species could be higher.
- **Completeness below 50%** indicates severe undersampling. The Chao1 estimate itself becomes unreliable because it is a lower-bound estimator and needs reasonable sample sizes to work.

### Per-cell vs. per-period estimates

The pipeline computes Chao1 at two levels:
- **Per cell (h3_chao1 layer):** Useful for identifying which cells need more sampling. But cells with < 5 records produce unreliable estimates.
- **Per year (chao1_per_period.csv):** Shows whether survey effort improved over time. Early years with few records will show low completeness regardless of actual species presence.

---

## Why species harmonization matters

In the real REEPS database, the same animal can appear under multiple names:

- **Scientific misspellings:** "Presbytys comata" vs. the correct "Presbytis comata"
- **Local/Indonesian names:** "macan" (Sundanese for leopard), "owa" (Sundanese for gibbon)
- **Outdated taxonomy:** "Nycticebus coucang" was split; Javan populations are now "Nycticebus javanicus"
- **Case and formatting:** "Lutung" vs. "lutung", "Babi Hutan" vs. "babi hutan"

Without harmonization, the same species gets counted as multiple species, inflating richness and confounding all downstream analyses. A cell that actually has 5 species might appear to have 7 due to name variants.

The harmonization map in `workshop_stage1_master_gpkg.py` (50+ entries) maps every known variant to a single canonical scientific name. This is a one-way mapping: the original name is preserved in `Species_Original` so the correction is auditable.

For synthetic workshop data, harmonization changes nothing because the generator uses canonical names directly. For real data, it is one of the most important data quality steps.

---

## Why stepping stones matter

Conservation typically focuses on occupied habitat: where are the animals now? But landscape ecology shows that the spaces *between* occupied patches are critical for long-term persistence.

### The stepping stone concept

If two occupied patches are separated by 3 empty hexagonal cells, animals must cross those cells to move between patches. If one of those empty cells is destroyed (e.g., converted to agriculture), the movement path becomes longer or impossible. The occupied patches become isolated, leading to:

- **Reduced gene flow:** Small populations without immigration lose genetic diversity.
- **No recolonization:** If a local population goes extinct, it cannot be re-established from a neighboring patch.
- **Edge effects:** Isolated patches have more edge relative to interior, increasing predation and invasive species pressure.

### Betweenness centrality intuition

The pipeline uses graph betweenness centrality to identify which empty cells are most important as stepping stones. Betweenness measures how many shortest paths between all pairs of nodes pass through a given node. In the H3 adjacency graph:

- A cell with high betweenness sits on many shortest paths between occupied cells. It is a bottleneck -- if it becomes impassable, many paths between patches become longer.
- A cell with low betweenness is bypassed by most paths. Losing it has little effect on overall connectivity.

### The scoring system

The stepping stone score combines three signals:

1. **K1 occupied neighbors (weight 3):** A cell adjacent to multiple occupied cells directly connects populations.
2. **K2 richness (weight 0.5):** Species diversity in the neighborhood indicates ecological value of maintaining the connection.
3. **Betweenness (weight 10):** Graph centrality identifies structural bottlenecks even when local neighbor counts are low.

A cell classified as a "corridor candidate" (K1 occupied neighbors >= 2) fills a gap between two occupied cells. These are the most urgent candidates for habitat protection or restoration.

---

## Why interactive AND static maps?

The pipeline produces both Folium HTML maps and matplotlib PNG figures. This is not redundant -- they serve different audiences and purposes.

### Interactive HTML maps

- **Exploration:** Click cells, toggle layers, zoom in/out. Essential for understanding spatial patterns when you do not know what to look for yet.
- **Data inspection:** Tooltips show exact values (CPI score, species list, Chao1 completeness) without needing to query the GeoPackage.
- **Stakeholder presentations:** Non-technical audiences can interact with the data directly in a browser.
- **Limitations:** Cannot be embedded in PDF reports. File sizes are large (5-20 MB). Rendering depends on browser JavaScript.

### Static PNG figures

- **Publication:** Journals and reports require static images at specific DPI (300 DPI standard).
- **Reproducibility:** A PNG is identical every time. An interactive map renders differently on different screens and zoom levels.
- **Print:** Static maps can be printed at precise scales.
- **Limitations:** No interactivity. Choosing what to show requires pre-made design decisions (colormap, zoom level, which layer).

Both are generated from the same underlying data, so they are always consistent.

---

## Understanding temporal turnover

The `h3_temporal_traj` layer tracks which species appear and disappear in each cell over time. The key metric is `Mean_Beta_Temporal`, a measure of species turnover between consecutive survey years.

### What beta-diversity means here

Temporal beta-diversity asks: "How different is the species composition of this cell in year t compared to year t-1?"

```
beta = (gained + lost) / |union|
```

- `gained`: species present in the current year but not the previous year
- `lost`: species present in the previous year but not the current year
- `union`: all species present in either year

A beta of 0 means the species composition is identical. A beta of 1 means complete turnover (no species in common).

### Why high turnover does not equal ecological crisis

Observing high temporal beta-diversity in biodiversity survey data is common, but its interpretation requires caution:

**True ecological turnover** occurs when species actually colonize or go locally extinct. This happens over ecological timescales (decades) and reflects habitat change, climate shifts, or competitive exclusion.

**Apparent turnover (survey artifact)** occurs when species are present but undetected. With small sample sizes, a species may be detected in one year and missed in the next -- not because it left, but because it was simply not encountered. This is especially common for:
- Rare species (Javan Leopard, Sunda Pangolin) with low detection probability
- Years with few survey days or limited spatial coverage
- Cells with very few total records

For the UCPS dataset, most observed turnover is likely a survey artifact. The `Colonisations` and `Extinctions` columns should be interpreted as "first/last detected" rather than "arrived/departed" unless detection probability is known to be high.

### The persistence classification

The pipeline classifies cells into temporal categories:
- **Persistent (>= 3 years with data):** Most reliable. Species trends from these cells carry more weight.
- **Recent (last detected >= 2020):** Data exists but may be too sparse for trend analysis.
- **Historical (last detected before 2020):** These cells may still be occupied but have not been resurveyed. High priority for new surveys.
- **Never Detected:** No observations in any year. Could be unsuitable habitat, or simply never surveyed.

---

## The survey effort confound

The single most important caveat in biodiversity analysis: **more records does not mean more species.** Survey effort -- how many person-hours, camera trap nights, or transect walks were conducted -- directly determines how many species can be detected.

### The problem

Consider two cells:
- Cell A: 50 records from 6 survey years using camera traps, observation, and sign surveys. 8 species detected.
- Cell B: 3 records from 1 survey year via a single interview. 2 species detected.

Does Cell A really have more species than Cell B? Almost certainly -- but we cannot know whether Cell B hosts 2 species or 8, because it was barely surveyed.

### How this affects the pipeline

**Species richness** is biased toward well-surveyed cells. Cells with more records tend to show higher richness, but this may reflect effort rather than ecological reality. The `effort_per_period.csv` output helps identify years and cells where effort was low.

**Chao1 partially corrects for this** by estimating undetected species from singleton/doubleton ratios. But Chao1 itself needs reasonable sample sizes (>10 records) to be reliable.

**CPI priority** inherits the richness bias. A cell may rank LOW not because it has low biodiversity, but because it was poorly surveyed. This is why the survey gap analysis (`h3_survey_gaps`) exists -- it identifies unoccupied cells that might actually be occupied if surveyed.

**Temporal trends** are especially vulnerable. If 2017 had 100 records and 2024 had 20 records, an apparent species decline may just be an effort decline. The `effort_per_period.csv` file should always be consulted alongside trend outputs.

### What to do about it

1. **Report effort alongside results.** Never present richness maps without noting survey effort per cell and per period.
2. **Use Chao1 completeness as a confidence indicator.** Cells with < 70% completeness should be flagged as uncertain.
3. **Prioritize survey gaps.** The `h3_survey_gaps` layer identifies where additional surveys would most improve knowledge.
4. **Be skeptical of temporal trends.** Only trust trends from cells/species with data in >= 3 years AND from `spearman_trends.csv` where p-value < 0.05.

---

## Why the pipeline uses synthetic data for the workshop

The workshop uses randomized synthetic data rather than real REEPS data for several reasons:

1. **Reproducibility without data sharing.** Real biodiversity data for threatened species is often sensitive -- sharing exact GPS locations of Javan Leopard dens could enable poaching. Synthetic data eliminates this risk while preserving the pipeline structure.

2. **Fresh data every run.** Because `generate_synthetic_data.py` uses a random seed of `None`, each run produces different data. This means participants cannot memorize answers -- they must actually interpret their unique results.

3. **Same pipeline, same code.** The workshop scripts are adapted from the production pipeline (`recalc_stage1_master_gpkg.py`, etc.). The only differences are file paths and the data source. Everything a participant learns applies directly to real data processing.

4. **Known ground truth.** With synthetic data, we know exactly what was generated. This makes it possible to verify that the pipeline works correctly (e.g., checking that Chao1 estimates are reasonable given the known species pool of 11 REEPS species).

The synthetic data is intentionally realistic: it includes spatial clustering (hotspots), uneven temporal effort, and the same species/method distributions as the real data. It does NOT include realistic ecological processes (dispersal, competition, habitat selection), so the spatial patterns are random rather than ecologically meaningful.
