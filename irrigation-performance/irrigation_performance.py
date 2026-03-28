#!/usr/bin/env python3
"""
Irrigation Performance Analysis for Daerah Irigasi (DI) Klambu

Local Python conversion of PUPR_IrrigationPerformance_29Jan26.ipynb (Colab).
Computes Satisfaction Index, Uniformity, and Reliability from satellite-derived
ET0 (ERA5-Land) and ETa (SSEBop) for tertiary irrigation blocks.

Pipeline:
    1. GEE authentication (service account)
    2. Load tertiary block boundaries from GEE asset
    3. Satellite processing: Landsat + Sentinel-2 → NDVI/NDWI max dates → transplanting dates
    4. ET0 from ERA5-Land (Penman-Monteith)
    5. ETa from SSEBop/MODIS
    6. Rice Kc curve (110-day variety)
    7. CWR = ET0 * Kc
    8. Satisfaction Index = (1.2 * ETa) / CWR
    9. Weekly SI aggregation
    10. Uniformity & Reliability calculation
    11. Mapping and export

Usage:
    # Full pipeline
    python irrigation_performance.py \
        --study-area projects/ee-ozdogan05/assets/indonesia/klambu_nonsaluran_tertiary \
        --year 2023 \
        --season-start 10 \
        --output-dir irrigation_results/klambu_2023

    # Resume from existing intermediate files
    python irrigation_performance.py \
        --study-area projects/ee-ozdogan05/assets/indonesia/klambu_nonsaluran_tertiary \
        --year 2023 \
        --season-start 10 \
        --output-dir irrigation_results/klambu_2023 \
        --skip-gee

Requirements:
    pip install earthengine-api google-auth geopandas pandas numpy matplotlib openpyxl
"""

import os
import sys
import argparse
import math
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: GEE Authentication & Setup
# ============================================================================

def initialize_gee(project_id=None, service_account_email=None, key_file=None):
    """Initialize Google Earth Engine with service account or default credentials."""
    import ee

    if key_file and os.path.exists(key_file):
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(
            key_file, scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials=credentials, project=project_id)
        logger.info(f"GEE initialized with service account (project: {project_id})")
    else:
        try:
            ee.Initialize(project=project_id)
            logger.info(f"GEE initialized with default credentials (project: {project_id})")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            logger.info("GEE initialized after authentication")

    # Verify
    info = ee.String('Earth Engine is ready!').getInfo()
    logger.info(info)
    return ee


# ============================================================================
# SECTION 2: GEE Helper Functions
# ============================================================================

def get_gee_functions(ee):
    """Return all GEE processing functions."""

    # --- Cloud masking ---
    def getQABits(image, start, end, newName):
        pattern = 0
        for i in range(start, end + 1):
            pattern += int(math.pow(2, i))
        return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

    def maskClouds(image):
        pixelQA = image.select('QA_PIXEL')
        cloud = getQABits(pixelQA, 3, 3, 'cloud')
        cldShadow = getQABits(pixelQA, 4, 4, 'cloud_shadow')
        return (image.updateMask(cloud.eq(0))
                     .updateMask(cldShadow.eq(0))
                     .copyProperties(image)
                     .set('system:time_start', image.get('system:time_start')))

    def scaleC2(image):
        bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        scaled = []
        for b in bands:
            s = image.select(b).multiply(0.0000275).add(-0.2).rename(b)
            s = s.updateMask(s.gte(0).Or(s.lte(1)))
            scaled.append(s)
        result = scaled[0]
        for s in scaled[1:]:
            result = result.addBands(s)
        return (result.copyProperties(image)
                      .set('system:time_start', image.get('system:time_start')))

    # --- Sentinel-2 cloud masking ---
    CLD_PRB_THRESH = ee.Number(40)
    NIR_DRK_THRESH = ee.Number(0.15)
    CLD_PRJ_DIST = ee.Number(2)
    BUFFER = ee.Number(50)

    def add_cloud_bands(img):
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
        return ee.Image(img).addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(img):
        not_water = ee.Image(img).select('SCL').neq(6)
        SR_BAND_SCALE = 1e4
        dark_pixels = (ee.Image(img).select('B8')
                         .lt(NIR_DRK_THRESH.multiply(SR_BAND_SCALE))
                         .multiply(not_water).rename('dark_pixels'))
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(ee.Image(img).get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        cld_proj = (ee.Image(img).select('clouds')
                      .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST.multiply(10))
                      .reproject(crs=ee.Image(img).select(0).projection(), scale=100)
                      .select('distance').mask().rename('cloud_transform'))
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(img):
        img_cloud = add_cloud_bands(img)
        img_cloud_shadow = add_shadow_bands(img_cloud)
        is_cld_shdw = (img_cloud_shadow.select('clouds')
                         .add(img_cloud_shadow.select('shadows')).gt(0))
        is_cld_shdw = (is_cld_shdw.focalMin(2)
                         .focalMax(BUFFER.multiply(2).divide(20))
                         .reproject(crs=ee.Image(img).select([0]).projection(), scale=20)
                         .rename('cloudmask'))
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask(img):
        not_cld_shdw = ee.Image(img).select('cloudmask').Not()
        return ee.Image(img).select('B.*').updateMask(not_cld_shdw)

    # --- Vegetation indices ---
    def addNDVI(image):
        ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI')
        ndvi = ndvi.updateMask(ndvi.gt(-0.1))
        return (image.addBands(ndvi).copyProperties(image)
                     .set('system:time_start', image.get('system:time_start')))

    def addNDWI(image):
        ndwi = image.normalizedDifference(['green', 'nir']).rename('NDWI')
        ndwi = ndwi.updateMask(ndwi.lt(0.1))
        return (image.addBands(ndwi).copyProperties(image)
                     .set('system:time_start', image.get('system:time_start')))

    def addDate(image):
        year = ee.Date(image.get('system:time_start')).get('year')
        month = ee.Date(image.get('system:time_start')).get('month')
        day = ee.Date(image.get('system:time_start')).get('day')
        date = ee.Date.fromYMD(year, month, day).format("yyyy-MM-dd")
        return (image.set('DATE_ACQUIRED', date).copyProperties(image)
                     .set('system:time_start', image.get('system:time_start')))

    def addDOY(image):
        doy = ee.Number(ee.Date(image.get('system:time_start')).getRelative('day', 'year')).add(1)
        return (image.addBands(ee.Image(doy).rename('DOY')).copyProperties(image)
                     .set('system:time_start', image.get('system:time_start')))

    def makeFloat(image):
        return (image.toFloat().copyProperties(image)
                     .set('system:time_start', image.get('system:time_start')))

    # --- ET0 from ERA5-Land ---
    def addET0(image):
        tmean = image.select('temperature_2m').subtract(273.15).rename('Tmean')
        tdew = image.select('dewpoint_temperature_2m').subtract(273.15).rename('Tdew')
        ea = tdew.expression(
            '6.11 * 10 ** ((7.5 * tdew) / (237.3 + tdew))',
            {'tdew': tdew}).divide(10.0).rename('ea')
        es = tmean.expression(
            '0.6108 * exp(17.27 * tmean / (tmean + 237.3))',
            {'tmean': tmean}).rename('es')
        u = image.select('u_component_of_wind_10m').pow(2)
        v = image.select('v_component_of_wind_10m').pow(2)
        ws = u.add(v).sqrt().rename('wind')
        SWnet = image.select('surface_net_solar_radiation_sum').divide(1000000).rename('SWnet')
        LWnet = image.select('surface_net_thermal_radiation_sum').divide(1000000).rename('LWnet')
        Rnet = SWnet.subtract(LWnet).rename('Rnet')
        slope = tmean.expression(
            '(4090 * (0.6108 * exp((17.27 * tmean)/(tmean + 237.3)))) / pow((tmean + 237.3), 2)',
            {'tmean': tmean}).rename('slope')
        psy = image.select('surface_pressure').multiply(0.001).multiply(0.000665).rename('psy')
        et0 = tmean.expression(
            '(0.408 * slope * (Rnet - 0) + (psy * (900 / (tmean + 273))) * ws * (es - ea)) / '
            '(slope + psy * (0.34 * ws + 1))',
            {'slope': slope, 'Rnet': Rnet, 'psy': psy, 'tmean': tmean,
             'ws': ws, 'es': es, 'ea': ea}).rename('ET0')
        return (image.addBands(et0).copyProperties(image)
                     .set('system:time_start', image.get('system:time_start'))
                     .set('system:time_end', image.get('system:time_end')))

    # --- Linear interpolation ---
    def LinearResampling(collection, date_attribute, date_interval, region):
        bandNames = collection.first().bandNames()
        minDate = ee.Date(ee.Number(collection.first().get(date_attribute)))

        def add_timestep(image):
            return (image.set("timestep",
                             ee.Number(ee.Date(ee.Number(image.get(date_attribute)))
                                     .difference(minDate, "day")).float())
                         .copyProperties(image)
                         .set('system:time_start', image.get('system:time_start')))

        collection = collection.map(add_timestep).sort("timestep")
        dateLimit = (ee.Number(ee.Image(collection.toList(1, collection.size().add(-1)).get(0))
                              .get("timestep")).ceil().int())

        def add_constant(image):
            constant = (ee.Image.constant(ee.Number(image.get("timestep")).float())
                          .float().mask(ee.Image(image).mask().float().select([0]))
                          .set("timestep", image.get("timestep")).rename("timestep"))
            return (image.addBands(constant).copyProperties(image)
                         .set('system:time_start', image.get('system:time_start')))

        indexes = collection.map(add_constant).sort("timestep", False)
        dateSequence = ee.List.sequence(0, dateLimit.add(-1), date_interval)

        def mosaic_upper_iteration(dateVal, acc):
            acc = ee.List(acc)
            latest_image = (indexes.filter(ee.Filter.gte('timestep', ee.Number(dateVal).float()))
                                 .mosaic())
            return acc.add(latest_image)

        mosaicUpperVal = ee.List(dateSequence.iterate(mosaic_upper_iteration, ee.List([])))

        def calc_date_diff(dateVal):
            idx = ee.Number(dateVal).float().divide(date_interval).round().int()
            return (ee.Image(mosaicUpperVal.get(idx))
                      .subtract(ee.Image(mosaicUpperVal.get(idx.subtract(1)))).neq(0))

        dateDiff = dateSequence.slice(1, dateSequence.size()).map(calc_date_diff)
        dateDiff = ee.List([dateDiff.get(0)]).cat(dateDiff)

        def mosaic_lower_iteration(dateVal, acc):
            idx = ee.Number(dateVal).float().divide(date_interval).round().int()
            acc = ee.List(acc)
            min_img = ee.Image(acc.get(0))
            list_imgs = ee.List(acc.get(1))
            diff = ee.Image(dateDiff.get(idx))
            new_img = ee.Image(mosaicUpperVal.get(idx.subtract(1)))
            min_img = min_img.where(diff, new_img)
            list_imgs = list_imgs.add(min_img)
            return [min_img, list_imgs]

        mosaicLowerVal = ee.List(ee.List(dateSequence.iterate(
            mosaic_lower_iteration,
            ee.List([ee.Image(mosaicUpperVal.get(0)), []]))).get(1))

        def interpolation_iteration(dateVal, acc):
            acc = ee.List(acc)
            idx = ee.Number(dateVal).float().divide(date_interval).round().int()
            minFrame = ee.Image(mosaicLowerVal.get(idx)).float()
            maxFrame = ee.Image(mosaicUpperVal.get(idx)).float()
            minDate_img = minFrame.select("timestep")
            maxDate_img = maxFrame.select("timestep")
            minVal = minFrame.select(bandNames)
            maxVal = maxFrame.select(bandNames)
            constantIdx = (ee.Image.constant(ee.Number(dateVal)).float()
                             .clip(region).rename("timestep"))
            out = minVal.add((maxVal.subtract(minVal))
                            .multiply(constantIdx.subtract(minDate_img))
                            .divide(maxDate_img.subtract(minDate_img)))
            out = ee.Algorithms.If(
                acc.size().gt(0),
                out.unmask(0).clip(region).where(out.mask().Not(), acc.get(acc.size().add(-1))),
                out)
            return acc.add(out)

        interpolated = ee.List(dateSequence.iterate(interpolation_iteration, ee.List([])))
        return ee.ImageCollection(interpolated)

    return {
        'maskClouds': maskClouds,
        'scaleC2': scaleC2,
        'add_cld_shdw_mask': add_cld_shdw_mask,
        'apply_cld_shdw_mask': apply_cld_shdw_mask,
        'addNDVI': addNDVI,
        'addNDWI': addNDWI,
        'addDate': addDate,
        'addDOY': addDOY,
        'makeFloat': makeFloat,
        'addET0': addET0,
        'LinearResampling': LinearResampling,
    }


# ============================================================================
# SECTION 3: GEE Data Extraction
# ============================================================================

def extract_transplanting_dates(ee, funcs, study_area_fc, box, sdate, edate, output_dir):
    """
    Extract NDVI/NDWI max dates per tertiary block to estimate transplanting dates.
    Merges Landsat 5/7/8/9 + Sentinel-2, interpolates, computes vegetation indices.
    """
    logger.info("Extracting transplanting dates from satellite imagery...")
    CLOUD_FILTER = ee.Number(60)

    # Sentinel-2
    s2coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(box).filterDate(sdate, edate)
                .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    s2cloudless = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(box).filterDate(sdate, edate))
    filter_join = ee.Filter.equals(leftField='system:index', rightField='system:index')
    saveFirstJoin = ee.Join.saveFirst(matchKey='s2cloudless')
    joined = saveFirstJoin.apply(s2coll, s2cloudless, filter_join)
    s2 = (ee.ImageCollection(joined.map(funcs['add_cld_shdw_mask'])
                                   .map(funcs['apply_cld_shdw_mask']))
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
                   ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
            .map(funcs['addDate']).map(funcs['makeFloat']))

    # Landsat collections
    def get_landsat(collection_id, band_names):
        return (ee.ImageCollection(collection_id)
                  .filterBounds(box).filterDate(sdate, edate)
                  .map(funcs['maskClouds'])
                  .select(band_names, ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
                  .map(funcs['scaleC2']).map(funcs['makeFloat']))

    L9 = get_landsat('LANDSAT/LC09/C02/T1_L2',
                     ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    L8 = get_landsat('LANDSAT/LC08/C02/T1_L2',
                     ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    L7 = get_landsat('LANDSAT/LE07/C02/T1_L2',
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
    L5 = get_landsat('LANDSAT/LT05/C02/T1_L2',
                     ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])

    # Merge and interpolate
    merged = s2.merge(L9).merge(L8).merge(L7).merge(L5)
    merged = merged.map(funcs['addDate']).map(funcs['addDOY']).sort('system:time_start')

    logger.info(f"Merged collection size: {merged.size().getInfo()}")

    # Interpolate to 8-day intervals
    interpolated = funcs['LinearResampling'](merged, 'system:time_start', 8, box)

    # Add NDVI and NDWI
    interpolated_vi = interpolated.map(funcs['addNDVI']).map(funcs['addNDWI'])

    # Compute max NDVI/NDWI dates per block
    ndvi_max = interpolated_vi.select('NDVI').reduce(ee.Reducer.max())
    ndwi_max = interpolated_vi.select('NDWI').reduce(ee.Reducer.max())

    # Get DOY of max NDVI and NDWI per block
    def get_max_dates(feature):
        geom = feature.geometry()
        ndvi_stats = ndvi_max.reduceRegion(
            reducer=ee.Reducer.max(), geometry=geom, scale=30, maxPixels=1e9)
        ndwi_stats = ndwi_max.reduceRegion(
            reducer=ee.Reducer.max(), geometry=geom, scale=30, maxPixels=1e9)
        return feature.set({
            'NDVImaxValue': ndvi_stats.get('NDVI_max'),
            'NDWImaxValue': ndwi_stats.get('NDWI_max'),
        })

    klambu_with_dates = study_area_fc.map(get_max_dates)

    # Export to local
    logger.info("Downloading transplanting dates...")
    features = klambu_with_dates.getInfo()['features']
    records = [f['properties'] for f in features]
    df = pd.DataFrame(records)
    df['norec'] = pd.to_numeric(df['norec'], errors='coerce').fillna(0).astype(int)

    output_file = Path(output_dir) / 'klambu_new_dates.xlsx'
    df.to_excel(output_file, index=False)
    logger.info(f"Transplanting dates saved: {output_file} ({len(df)} records)")
    return df


def extract_daily_et0(ee, funcs, study_area_fc, box, sdate, edate, output_dir):
    """Extract daily ET0 from ERA5-Land for each tertiary block."""
    logger.info("Extracting daily ET0 from ERA5-Land...")

    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
              .filterDate(sdate, edate)
              .select(['temperature_2m', 'dewpoint_temperature_2m',
                       'u_component_of_wind_10m', 'v_component_of_wind_10m',
                       'surface_net_solar_radiation_sum',
                       'surface_net_thermal_radiation_sum',
                       'surface_pressure'])
              .map(funcs['addET0']))

    # Extract ET0 per block per day using reduceRegions
    def extract_et0_for_image(image):
        date = ee.Date(image.get('system:time_start'))
        year = date.get('year')
        month = date.get('month')
        day = date.get('day')
        doy = date.getRelative('day', 'year').add(1)

        reduced = image.select('ET0').reduceRegions(
            collection=study_area_fc,
            reducer=ee.Reducer.mean(),
            scale=11132
        )

        def add_date_props(feature):
            return feature.set({
                'year': year, 'month': month, 'day': day,
                'date': date.format('yyyy-MM-dd'), 'DOY': doy,
                'ET0': feature.get('mean')
            })

        return reduced.map(add_date_props)

    # Process in batches to avoid GEE limits
    era5_list = era5.toList(era5.size())
    n_images = era5.size().getInfo()
    logger.info(f"Processing {n_images} ERA5-Land daily images...")

    all_records = []
    batch_size = 30  # Process 30 days at a time

    for start_idx in range(0, n_images, batch_size):
        end_idx = min(start_idx + batch_size, n_images)
        logger.info(f"  Processing days {start_idx+1}-{end_idx}...")

        batch = ee.ImageCollection(era5_list.slice(start_idx, end_idx))
        batch_results = batch.map(extract_et0_for_image).flatten()

        try:
            features = batch_results.getInfo()['features']
            records = [f['properties'] for f in features]
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"  Batch {start_idx}-{end_idx} failed: {e}, trying individually...")
            for i in range(start_idx, end_idx):
                try:
                    img = ee.Image(era5_list.get(i))
                    result = extract_et0_for_image(img)
                    features = result.getInfo()['features']
                    records = [f['properties'] for f in features]
                    all_records.extend(records)
                except Exception as e2:
                    logger.warning(f"    Day {i} failed: {e2}")

    df = pd.DataFrame(all_records)
    output_file = Path(output_dir) / 'klambu_daily_ET0.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Daily ET0 saved: {output_file} ({len(df)} records)")
    return df


def extract_daily_eta(ee, funcs, study_area_fc, box, sdate, edate, output_dir):
    """Extract daily ETa from SSEBop for each tertiary block."""
    logger.info("Extracting daily ETa from SSEBop...")

    # Use SSEBop (or MODIS ET)
    eta_collection = (ee.ImageCollection("MODIS/061/MOD16A2GF")
                        .filterDate(sdate, edate)
                        .filterBounds(box)
                        .select('ET'))

    def extract_eta_for_image(image):
        date = ee.Date(image.get('system:time_start'))
        year = date.get('year')
        month = date.get('month')
        day = date.get('day')
        doy = date.getRelative('day', 'year').add(1)

        # MODIS ET is in kg/m2/8day, convert to mm/day
        daily_et = image.select('ET').multiply(0.1).divide(8)

        reduced = daily_et.reduceRegions(
            collection=study_area_fc,
            reducer=ee.Reducer.mean(),
            scale=500
        )

        def add_date_props(feature):
            return feature.set({
                'year': year, 'month': month, 'day': day,
                'date': date.format('yyyy-MM-dd'), 'DOY': doy,
                'ETa': feature.get('mean')
            })

        return reduced.map(add_date_props)

    eta_list = eta_collection.toList(eta_collection.size())
    n_images = eta_collection.size().getInfo()
    logger.info(f"Processing {n_images} MODIS ET images...")

    all_records = []
    for i in range(n_images):
        try:
            img = ee.Image(eta_list.get(i))
            result = extract_eta_for_image(img)
            features = result.getInfo()['features']
            records = [f['properties'] for f in features]
            all_records.extend(records)
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{n_images} images")
        except Exception as e:
            logger.warning(f"  Image {i} failed: {e}")

    df = pd.DataFrame(all_records)
    output_file = Path(output_dir) / 'klambu_daily_ETa.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Daily ETa saved: {output_file} ({len(df)} records)")
    return df


# ============================================================================
# SECTION 4: Local Processing (no GEE needed)
# ============================================================================

def create_rice_kc_110(output_dir):
    """Create 110-day rice Kc curve."""
    logger.info("Creating rice Kc curve (110-day variety)...")

    days = list(range(1, 111))
    kc_values = []
    for day in days:
        if day <= 10:
            kc = 1.05  # Initial stage
        elif day <= 40:
            kc = 1.05 + (1.15 - 1.05) * (day - 10) / 30  # Development
        elif day <= 80:
            kc = 1.15 + (1.2 - 1.15) * (day - 40) / 40  # Mid-season
        else:
            kc = 1.2 + (0.95 - 1.2) * (day - 80) / 30  # Late season
        kc_values.append(round(kc, 3))

    rice_kc_df = pd.DataFrame({'Day': days, 'Kc': kc_values})
    output_file = Path(output_dir) / 'rice_kc_110.xlsx'
    rice_kc_df.to_excel(output_file, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rice_kc_df['Day'], rice_kc_df['Kc'], 'r-', linewidth=2, label='Kc')
    ax.axvspan(1, 10, alpha=0.2, color='blue', label='Initial')
    ax.axvspan(11, 40, alpha=0.2, color='green', label='Development')
    ax.axvspan(41, 80, alpha=0.2, color='yellow', label='Mid-season')
    ax.axvspan(81, 110, alpha=0.2, color='orange', label='Late season')
    ax.set_xlabel('Day after transplanting')
    ax.set_ylabel('Crop coefficient (Kc)')
    ax.set_title('Rice Crop Coefficient (Kc) - 110 Day Variety')
    ax.set_ylim(0.9, 1.25)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'rice_kc_110.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Kc curve saved: {output_file}")
    return rice_kc_df


def join_et0_with_dates(output_dir):
    """Join daily ET0 with transplanting dates."""
    logger.info("Joining ET0 with transplanting dates...")

    et0_file = Path(output_dir) / 'klambu_daily_ET0.csv'
    dates_file = Path(output_dir) / 'klambu_new_dates.xlsx'

    df1 = pd.read_csv(et0_file)
    df2 = pd.read_excel(dates_file)

    # Set start/end dates from NDWI/NDVI max dates
    if 'NDWImaxDate' in df2.columns and 'NDVImaxDate' in df2.columns:
        df2['startDate'] = df2['NDWImaxDate'].astype(int)
        df2['endDate'] = df2['NDVImaxDate'].astype(int)

    if 'diff' not in df2.columns:
        df2['diff'] = df2['endDate'] - df2['startDate']

    if 'mean' in df1.columns:
        df1 = df1.rename(columns={'mean': 'ET0'})

    output = pd.merge(df1, df2[['norec', 'startDate', 'endDate', 'diff']],
                       on=['norec'], how='outer')
    output_file = Path(output_dir) / 'klambu_daily_ET0_with_startdates.csv'
    output.to_csv(output_file, index=False)
    logger.info(f"ET0+dates saved: {output_file} ({len(output)} records)")
    return output


def calculate_cwr(output_dir, n_blocks=749):
    """Calculate Crop Water Requirements = ET0 * Kc (time-aligned to transplanting)."""
    logger.info("Calculating crop water requirements (CWR)...")

    df = pd.read_csv(Path(output_dir) / 'klambu_daily_ET0_with_startdates.csv')
    kc = pd.read_excel(Path(output_dir) / 'rice_kc_110.xlsx')

    # Find transplanting start index (DOY == startDate)
    df['target'] = (df['DOY'] - df['startDate']).abs()
    idx = df.index[df['target'] == 0].to_numpy()

    # Expand: 110 days from each transplanting start
    inc = 1
    ninc = 110
    newidx = pd.Index(np.add.outer(idx, inc * np.arange(ninc)).ravel())

    # Filter valid indices
    newidx = newidx[newidx < len(df)]
    newdf = df.iloc[newidx]

    # Repeat Kc for all blocks
    n = len(idx)
    kc_blocks = pd.concat([kc] * n, ignore_index=True)

    # Align lengths
    min_len = min(len(newdf), len(kc_blocks))
    newdf = newdf.iloc[:min_len].reset_index(drop=True)
    kc_blocks = kc_blocks.iloc[:min_len].reset_index(drop=True)

    combined = pd.concat([newdf, kc_blocks], axis=1)
    combined['CWR'] = combined['ET0'] * combined['Kc']

    output_file = Path(output_dir) / 'cwr.csv'
    combined.to_csv(output_file, index=False)
    logger.info(f"CWR saved: {output_file} ({len(combined)} records)")
    return combined


def join_eta_with_dates(output_dir):
    """Join daily ETa with transplanting dates."""
    logger.info("Joining ETa with transplanting dates...")

    df1 = pd.read_csv(Path(output_dir) / 'klambu_daily_ETa.csv')
    df2 = pd.read_excel(Path(output_dir) / 'klambu_new_dates.xlsx')

    if 'NDWImaxDate' in df2.columns:
        df2['startDate'] = df2['NDWImaxDate'].astype(int)
        df2['endDate'] = df2['NDVImaxDate'].astype(int)
    if 'diff' not in df2.columns:
        df2['diff'] = df2['endDate'] - df2['startDate']

    output = pd.merge(df1, df2[['norec', 'startDate', 'endDate', 'diff']],
                       on=['norec'], how='outer')
    output_file = Path(output_dir) / 'klambu_daily_ETa_with_startdates.csv'
    output.to_csv(output_file, index=False)
    logger.info(f"ETa+dates saved: {output_file} ({len(output)} records)")
    return output


def calculate_satisfaction_index(output_dir):
    """Calculate daily Satisfaction Index = (1.2 * ETa) / CWR."""
    logger.info("Calculating Satisfaction Index...")

    df = pd.read_csv(Path(output_dir) / 'klambu_daily_ETa_with_startdates.csv')
    cwr = pd.read_csv(Path(output_dir) / 'cwr.csv')

    # Find transplanting start indices
    df['target'] = (df['DOY'] - df['startDate']).abs()
    idx = df.index[df['target'] == 0].to_numpy()

    inc = 1
    ninc = 110
    newidx = pd.Index(np.add.outer(idx, inc * np.arange(ninc)).ravel())
    newidx = newidx[newidx < len(df)]
    newdf = df.iloc[newidx]

    # Join with CWR
    output = pd.merge(newdf, cwr, on=['norec', 'year', 'month', 'day'], how='inner')

    # SI = (1.2 * ETa) / CWR, capped at 1.0
    output['satIndex'] = (1.2 * output['ETa']) / output['CWR']
    output.loc[output['satIndex'] > 1, 'satIndex'] = 1.0
    output.loc[output['satIndex'] < 0, 'satIndex'] = 0.0

    # Select columns
    col_mapping = {
        'nama': ['nama', 'nama_x', 'nama_y'],
        'norec': ['norec'], 'year': ['year'], 'month': ['month'], 'day': ['day'],
        'date': ['date', 'date_x'], 'DOY': ['DOY', 'DOY_x'],
        'startDate': ['startDate', 'startDate_x'],
        'endDate': ['endDate', 'endDate_x'],
        'ET0': ['ET0'], 'Kc': ['Kc'], 'CWR': ['CWR'],
        'ETa': ['ETa'], 'satIndex': ['satIndex']
    }

    final_columns = []
    for desired, possible in col_mapping.items():
        for p in possible:
            if p in output.columns:
                final_columns.append(p)
                break

    output_file = Path(output_dir) / 'klambu_daily_SI.csv'
    output[final_columns].to_csv(output_file, index=False)
    logger.info(f"Daily SI saved: {output_file} ({len(output)} records)")
    return output[final_columns]


def aggregate_weekly_si(output_dir):
    """Aggregate daily SI to weekly SI."""
    logger.info("Aggregating daily SI to weekly SI...")

    df = pd.read_csv(Path(output_dir) / 'klambu_daily_SI.csv')

    # Create date column
    if 'date' not in df.columns:
        if all(c in df.columns for c in ['year', 'month', 'day']):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        else:
            df['date'] = pd.to_datetime('2023-10-01') + pd.to_timedelta(df.index, unit='D')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Year-week grouping
    df['year_week'] = df['date'].dt.strftime('%Y-W%U')

    # Weekly aggregation per block
    weekly = df.groupby(['norec', 'year_week']).agg({
        'satIndex': 'mean',
        'date': 'first'
    }).reset_index()

    weekly.columns = ['norec', 'year_week', 'satIndex', 'date']

    output_file = Path(output_dir) / 'klambu_weekly_SI.xlsx'
    weekly.to_excel(output_file, index=False)
    logger.info(f"Weekly SI saved: {output_file} ({len(weekly)} records)")
    return weekly


def calculate_uniformity_reliability(output_dir):
    """Calculate Uniformity and Reliability from weekly SI."""
    logger.info("Calculating Uniformity and Reliability...")

    df = pd.read_excel(Path(output_dir) / 'klambu_weekly_SI.xlsx')

    # 1. WEEKLY UNIFORMITY: spatial variation within each week
    weekly_uniformity = df.groupby('year_week').agg(
        mean_SI=('satIndex', 'mean'),
        std_SI=('satIndex', 'std'),
        block_count=('satIndex', 'count')
    ).reset_index().round(4)

    weekly_uniformity['coefficient_variation'] = (
        weekly_uniformity['std_SI'] / weekly_uniformity['mean_SI'] * 100
    )
    weekly_uniformity['uniformity'] = (
        1 - weekly_uniformity['std_SI'] / weekly_uniformity['mean_SI']
    ).fillna(1).clip(0, 1)

    # 2. BLOCK-LEVEL UNIFORMITY
    overall_mean_SI = df['satIndex'].mean()
    block_uniformity = df.groupby('norec').agg(
        mean_SI=('satIndex', 'mean'),
        std_SI=('satIndex', 'std'),
        week_count=('satIndex', 'count'),
        min_SI=('satIndex', 'min'),
        max_SI=('satIndex', 'max')
    ).reset_index().round(4)

    block_uniformity['deviation_from_overall'] = abs(block_uniformity['mean_SI'] - overall_mean_SI)
    block_uniformity['uniformity'] = (
        1 - block_uniformity['deviation_from_overall'] / overall_mean_SI
    ).fillna(1).clip(0, 1)

    # 3. RELIABILITY: temporal consistency per block
    si_threshold = 0.5
    reliability = df.groupby('norec').agg(
        mean_SI=('satIndex', 'mean'),
        std_SI=('satIndex', 'std'),
        total_weeks=('satIndex', 'count'),
        min_SI=('satIndex', 'min'),
        max_SI=('satIndex', 'max')
    ).reset_index().round(4)

    # Count weeks above threshold per block
    weeks_above = df[df['satIndex'] > si_threshold].groupby('norec').size().reset_index(name='weeks_above')
    reliability = reliability.merge(weeks_above, on='norec', how='left')
    reliability['weeks_above'] = reliability['weeks_above'].fillna(0).astype(int)
    reliability['reliability'] = reliability['weeks_above'] / reliability['total_weeks']
    reliability['reliability'] = reliability['reliability'].clip(0, 1)

    # Save results
    weekly_uniformity.to_csv(Path(output_dir) / 'uniformity_weekly_results.csv', index=False)
    block_uniformity.to_csv(Path(output_dir) / 'uniformity_block_results.csv', index=False)
    reliability.to_csv(Path(output_dir) / 'reliability_block_results.csv', index=False)

    # Summary
    with open(Path(output_dir) / 'performance_summary.txt', 'w') as f:
        f.write("Irrigation Performance Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall mean SI: {overall_mean_SI:.4f}\n\n")
        f.write(f"Weekly Uniformity:\n")
        f.write(f"  Mean: {weekly_uniformity['uniformity'].mean():.4f}\n")
        f.write(f"  Range: {weekly_uniformity['uniformity'].min():.4f} - {weekly_uniformity['uniformity'].max():.4f}\n")
        f.write(f"  Weeks with CU > 0.85: {(weekly_uniformity['uniformity'] > 0.85).sum()} / {len(weekly_uniformity)}\n\n")
        f.write(f"Block-level Uniformity:\n")
        f.write(f"  Mean: {block_uniformity['uniformity'].mean():.4f}\n")
        f.write(f"  Range: {block_uniformity['uniformity'].min():.4f} - {block_uniformity['uniformity'].max():.4f}\n\n")
        f.write(f"Reliability (SI threshold = {si_threshold}):\n")
        f.write(f"  Mean: {reliability['reliability'].mean():.4f}\n")
        f.write(f"  Range: {reliability['reliability'].min():.4f} - {reliability['reliability'].max():.4f}\n")
        f.write(f"  Blocks with RI > 0.75: {(reliability['reliability'] > 0.75).sum()} / {len(reliability)}\n")

    logger.info(f"Uniformity mean: {weekly_uniformity['uniformity'].mean():.4f}")
    logger.info(f"Reliability mean: {reliability['reliability'].mean():.4f}")

    # Plots
    plot_performance(weekly_uniformity, block_uniformity, reliability, output_dir)

    return weekly_uniformity, block_uniformity, reliability


def plot_performance(weekly_uniformity, block_uniformity, reliability, output_dir):
    """Create performance visualization plots."""
    output_dir = Path(output_dir)

    # 1. Weekly Uniformity time series
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(weekly_uniformity)), weekly_uniformity['uniformity'], 'g-o', markersize=4)
    ax.axhline(y=0.85, color='r', linestyle='--', alpha=0.7, label='Target CU = 0.85')
    ax.set_xlabel('Week')
    ax.set_ylabel('Uniformity Coefficient')
    ax.set_title("Christiansen's Uniformity Coefficient Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'uniformity_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Weekly Mean SI time series
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(weekly_uniformity)), weekly_uniformity['mean_SI'], 'b-o', markersize=4)
    ax.fill_between(range(len(weekly_uniformity)),
                    weekly_uniformity['mean_SI'] - weekly_uniformity['std_SI'],
                    weekly_uniformity['mean_SI'] + weekly_uniformity['std_SI'],
                    alpha=0.2, color='blue')
    ax.set_xlabel('Week')
    ax.set_ylabel('Mean Satisfaction Index')
    ax.set_title('Mean Satisfaction Index Over Time')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'si_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Reliability histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(reliability['reliability'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0.75, color='r', linestyle='--', label='Target RI = 0.75')
    ax.set_xlabel('Reliability Index')
    ax.set_ylabel('Number of Blocks')
    ax.set_title('Distribution of Reliability Index Across Tertiary Blocks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'reliability_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Block uniformity histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(block_uniformity['uniformity'], bins=20, color='forestgreen', edgecolor='white', alpha=0.8)
    ax.axvline(x=0.85, color='r', linestyle='--', label='Target CU = 0.85')
    ax.set_xlabel('Uniformity Index')
    ax.set_ylabel('Number of Blocks')
    ax.set_title('Distribution of Uniformity Index Across Tertiary Blocks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'uniformity_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Performance plots saved")


# ============================================================================
# SECTION 5: Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Irrigation Performance Analysis for Daerah Irigasi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (GEE + local processing)
    python irrigation_performance.py \\
        --study-area projects/ee-ozdogan05/assets/indonesia/klambu_nonsaluran_tertiary \\
        --year 2023 --season-start 10 \\
        --output-dir irrigation_results/klambu_2023

    # Skip GEE extraction (use existing CSVs)
    python irrigation_performance.py \\
        --study-area projects/ee-ozdogan05/assets/indonesia/klambu_nonsaluran_tertiary \\
        --year 2023 --season-start 10 \\
        --output-dir irrigation_results/klambu_2023 \\
        --skip-gee

    # Local processing only (all CSVs already exist)
    python irrigation_performance.py \\
        --output-dir irrigation_results/klambu_2023 \\
        --local-only
        """)
    parser.add_argument('--study-area', default='projects/ee-ozdogan05/assets/indonesia/klambu_nonsaluran_tertiary',
                        help='GEE asset path for study area FeatureCollection')
    parser.add_argument('--year', type=int, default=2023, help='Analysis year')
    parser.add_argument('--season-start', type=int, default=10,
                        help='Season start month (default: 10 = October)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--skip-gee', action='store_true',
                        help='Skip GEE extraction (use existing intermediate files)')
    parser.add_argument('--local-only', action='store_true',
                        help='Local processing only (all intermediate CSVs must exist)')
    parser.add_argument('--project-id', default='ee-geodeticengineeringundip',
                        help='GEE project ID')
    parser.add_argument('--key-file', default='ee-geodetic.json',
                        help='GEE service account key file')
    parser.add_argument('--n-blocks', type=int, default=749,
                        help='Number of tertiary blocks (default: 749 for Klambu)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logger.info(f"Irrigation Performance Analysis - {args.year}")
    logger.info(f"Output: {output_dir}")

    # ---- GEE Extraction Phase ----
    if not args.local_only:
        import ee
        ee_module = initialize_gee(args.project_id, key_file=args.key_file)
        funcs = get_gee_functions(ee)

        study_area_fc = ee.FeatureCollection(args.study_area)
        box = study_area_fc.geometry().bounds()

        year = ee.Number(args.year)
        sdate = ee.Date.fromYMD(year, args.season_start, 1)
        edate = sdate.advance(6, 'month')
        logger.info(f"Season: {sdate.getInfo()['value']} to {edate.getInfo()['value']}")

        if not args.skip_gee:
            # Step 1: Transplanting dates
            extract_transplanting_dates(ee, funcs, study_area_fc, box, sdate, edate, output_dir)

            # Step 2: Daily ET0
            extract_daily_et0(ee, funcs, study_area_fc, box, sdate, edate, output_dir)

            # Step 3: Daily ETa
            extract_daily_eta(ee, funcs, study_area_fc, box, sdate, edate, output_dir)

    # ---- Local Processing Phase ----
    logger.info("\n" + "=" * 60)
    logger.info("LOCAL PROCESSING")
    logger.info("=" * 60)

    # Step 4: Rice Kc curve
    create_rice_kc_110(output_dir)

    # Step 5: Join ET0 with transplanting dates
    et0_joined = Path(output_dir) / 'klambu_daily_ET0_with_startdates.csv'
    if et0_joined.exists():
        logger.info(f"Using existing {et0_joined}")
    else:
        join_et0_with_dates(output_dir)

    # Step 6: Calculate CWR
    cwr_file = Path(output_dir) / 'cwr.csv'
    if cwr_file.exists():
        logger.info(f"Using existing {cwr_file}")
    else:
        calculate_cwr(output_dir, args.n_blocks)

    # Step 7: Join ETa with transplanting dates
    eta_joined = Path(output_dir) / 'klambu_daily_ETa_with_startdates.csv'
    if eta_joined.exists():
        logger.info(f"Using existing {eta_joined}")
    else:
        join_eta_with_dates(output_dir)

    # Step 8: Calculate daily SI
    si_file = Path(output_dir) / 'klambu_daily_SI.csv'
    if si_file.exists():
        logger.info(f"Using existing {si_file}")
    else:
        calculate_satisfaction_index(output_dir)

    # Step 9: Weekly aggregation
    weekly_file = Path(output_dir) / 'klambu_weekly_SI.xlsx'
    if weekly_file.exists():
        logger.info(f"Using existing {weekly_file}")
    else:
        aggregate_weekly_si(output_dir)

    # Step 10: Uniformity and Reliability
    calculate_uniformity_reliability(output_dir)

    elapsed = time.time() - start_time
    logger.info(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Results in: {output_dir}")


if __name__ == '__main__':
    main()
