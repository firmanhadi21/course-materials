# Biodiversity Spatial Analysis Workshop

Interactive workshop for conservation priority analysis using H3 hexagonal grids, diversity indices, and spatial connectivity modelling. Uses **synthetic (randomized) data** that follows the same workflow as real REEPS fauna monitoring in the Upper Cisokan landscape, West Java, Indonesia.

## Quick Start

```bash
# Install dependencies
pip install geopandas h3 networkx scipy folium openpyxl matplotlib shapely pyproj rasterio pandas numpy

# Run the entire pipeline
python run_workshop.py
```

This generates a synthetic dataset (~560 records, 11 REEPS species), processes it through the full analysis pipeline, and produces interactive maps, static figures, and validation tables in ~30 seconds.

## Workshop Structure (Diátaxis Framework)

| Document | Type | Purpose |
|---|---|---|
| [TUTORIAL.md](TUTORIAL.md) | Tutorial | Step-by-step walkthrough (~80 min) |
| [HOWTO.md](HOWTO.md) | How-to | 11 task-oriented recipes |
| [REFERENCE.md](REFERENCE.md) | Reference | Full technical documentation |
| [EXPLANATION.md](EXPLANATION.md) | Explanation | Conceptual discussions |

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| 0 | `generate_synthetic_data.py` | Synthetic Excel + CSV |
| 1 | `workshop_stage1_master_gpkg.py` | Master GeoPackage (H3 assignment) |
| 2 | `workshop_stage2_grid_analyses.py` | Grid analyses (13 layers) |
| 3 | `workshop_stage3_maps_validation.py` | 7 interactive HTML maps + CSVs |
| 4 | `workshop_stage4_figures.py` | Publication figures (300 DPI) |

## Analysis Components

- **Diversity indices**: Shannon H', Simpson D, Pielou J, Berger-Parker
- **Chao1** richness estimation with sampling completeness
- **6-component Conservation Priority Index** (PCA-weighted):
  - Richness (17.7%), Connectivity (18.1%), Threatened species (16.9%)
  - Shannon diversity (18.0%), Co-occurrence (18.0%), Habitat permeability (11.3%)
- **Spatial connectivity** via NetworkX graph analysis
- **Corridor analysis** with resistance-weighted stepping stones
- **Temporal turnover** (Sorensen-Dice beta-diversity)

## Data Files

| File | Description | Size |
|---|---|---|
| `aoi.gpkg` | Study area boundary (real) | 96 KB |
| `Sentinel2_S2DR3_30Sep24_RGB_lowres.tif` | Basemap for figures (real, downsampled) | 11 MB |
| `PS_Final_11class_Hierarchical.tif` | Land cover for permeability (real, referenced from parent) | 1.6 MB |

The synthetic observation data is generated fresh each run with randomized locations and counts.

## Requirements

- Python 3.10+
- Key packages: `geopandas`, `h3`, `networkx`, `scipy`, `folium`, `openpyxl`, `matplotlib`, `rasterio`, `pyproj`
