# Import Libraries
import os
import re
import sys
import glob
import logging
import rasterio
import rioxarray
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import xrspatial as xrs
from pathlib import Path
from rasterio.merge import merge
from rasterstats import zonal_stats

from src.data_ingestion.spatial_transformations import easting_northing_to_lat_long, \
    find_catchment_boundary
    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

# TODO: Make resolution dynamic for all? Maybe not priority for this pipeline if certain.

# --- Land Cover ---

def load_land_cover_data(tif_path: str, csv_path: str, catchment: str, shape_filepath: str):
    """
    Loads land cover data from GeoTIFF file using xarray, flattens it to a DataFrame,
    converts x, y coordinates to lat/lon, and saves to CSV.
    """
    logger.info(f"Loading land cover data from {tif_path}...")
    
    # Load land cover data from GeoTIFF using xarray and select band
    land_cover_ds = rioxarray.open_rasterio(tif_path, masked=True)
    if 'band' in land_cover_ds.dims and len(land_cover_ds.coords['band']) > 1:
        land_cover_ds = land_cover_ds.sel(band=land_cover_ds.coords['band'].values[0])
        logger.info(f"Multiple land cover bands found, selecting first.")
    
    # Remove the band dimension if it's still there and store the original data crs
    land_cover_ds = land_cover_ds.squeeze()
    
    # --- Trim to catchment geometry or bounding box ---
    
    # Derive full path
    temp_geojson_path = f"{catchment}_combined_boundary.geojson"
    path = shape_filepath + temp_geojson_path
    
    _, _, minx, miny, maxx, maxy = find_catchment_boundary(
        catchment=catchment,
        shape_filepath=path,
        required_crs=27700
    )

    land_cover_ds_clipped = land_cover_ds.rio.clip_box(minx, miny, maxx, maxy)
    logger.info("Land cover data clipped to catchment bounding box.\n")
    
    # --- Convert ds to dataframe and clean up ---
    
    # Convert the xarray DataArray to a pandas DataFrame
    land_cover_df = land_cover_ds_clipped.to_dataframe(name='land_cover_code').reset_index()
    
    # Rename x and y for clarity then reproject to lat/lon
    land_cover_df = land_cover_df.rename(columns={'x': 'easting', 'y': 'northing'})
    land_cover_df = easting_northing_to_lat_long(input_df=land_cover_df)
    
    # Drop unneeded columns and check data type
    land_cover_df = land_cover_df.drop(['band', 'spatial_ref'], axis=1, errors='ignore')
    land_cover_df['land_cover_code'] = land_cover_df['land_cover_code'].astype('Int64')
    
    # Check for NaN
    NaN_count = land_cover_df['land_cover_code'].isna().sum()
    if NaN_count > 0:
        logger.info(f"Total missing land cover codes: {NaN_count}")
        
    # Aggregate categories to reduce dimensionality
    agg_land_cover_df = aggregate_land_cover_categories(land_cover_df)
    
    # Save to csv for preprocessing
    agg_land_cover_df.to_csv(csv_path, index=False)
    logger.info(f"Land Cover data succesfully saved to {csv_path}.")
    
    return agg_land_cover_df

def aggregate_land_cover_categories(land_use_df: pd.DataFrame):
    """
    Map land use categories from full detail to primary categories to reduce dimensionality
    """
    # Confirm data type
    land_use_df['land_cover_code'] = land_use_df['land_cover_code'].astype(int)
    
    # Define mapping
    mapping = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4,
               6: 5, 7: 6, 8: 6, 9: 6, 10: 7}
    
    # Map all categories to new combined categories
    land_use_df['land_cover_code'] = land_use_df['land_cover_code'].replace(mapping)

    # Convert to categorical dtype
    land_use_df['land_cover_code'] = land_use_df['land_cover_code'].astype('category')
        
    return land_use_df

# --- Elevation ---

def _load_mosaic_elevation(dir_path: str, mesh_cells_gdf_polygons: gpd.GeoDataFrame, csv_path: str):

    logger.info(f"Loading Elevation Data: Starting DTM processing from {dir_path} directory...")
    
    # Find all DTM .asc files recursively
    dtm_tiles_paths = glob.glob(os.path.join(dir_path, '**', '*.asc'), recursive=True)

    # If no DTM data then log error, add a column of NaNs and return the original mesh GDF
    if not dtm_tiles_paths:
        logger.error(f"No .asc files found in {dir_path}. Please check the path and file extensions.")
        mesh_cells_gdf_polygons['mean_elevation'] = np.nan
        return mesh_cells_gdf_polygons

    logger.info(f"Found {len(dtm_tiles_paths)} DTM .asc tiles.")

    # --- Mosaic (Merge) the DTM Tiles into a single GeoTIFF using rasterio ---
    
    dtm_files_to_mosaic = []
    
    try:
        for filepath in dtm_tiles_paths:
            data = rasterio.open(filepath)
            dtm_files_to_mosaic.append(data)
        
        mosaic, out_transform = merge(dtm_files_to_mosaic)
        out_meta = dtm_files_to_mosaic[0].meta.copy()
        
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "crs": dtm_files_to_mosaic[0].crs,
            "nodata": dtm_files_to_mosaic[0].nodata
        })

        with rasterio.open(csv_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        logger.info(f"Merged DTM saved to {csv_path}")
    
    except Exception as e:
        logger.error(f"Error merging DTM tiles: {e}. Cannot proceed with elevation data.", exc_info=True)
        mesh_cells_gdf_polygons['mean_elevation'] = np.nan
        return mesh_cells_gdf_polygons
    
    finally:
        # Close connection to file on disk
        for data in dtm_files_to_mosaic:
            data.close()
    
    return mesh_cells_gdf_polygons, dtm_files_to_mosaic

def _calculate_polygon_zone_average(mesh_cells_gdf_polygons: gpd.GeoDataFrame, clipped_dtm: gpd.GeoDataFrame):
    """
    Perform Zonal Statistics (rasterstats) to calculate mean elevation for each 1km mesh polygon.
    
    Return
        mesh_cells_gdf_polygons (pd.GeoDataFrame): 1km average elevation statistics in catchment
    """
    if mesh_cells_gdf_polygons.geometry.iloc[0].geom_type != 'Polygon':
        # If wrong data type log error and return unaggregated catchment data
        logger.error("mesh_cells_gdf_polygons geometries are not polygons. Zonal statistics requires polygons.")
        mesh_cells_gdf_polygons['mean_elevation'] = np.nan
        return mesh_cells_gdf_polygons
    
    if str(mesh_cells_gdf_polygons.crs) != str(clipped_dtm.rio.crs):
        logger.warning(f"Mesh cells GeoDataFrame CRS ({mesh_cells_gdf_polygons.crs}) does not match clipped "
                        f"DTM CRS ({clipped_dtm.rio.crs}). Reprojecting mesh cells for zonal stats.")
        mesh_cells_gdf_polygons = mesh_cells_gdf_polygons.to_crs(clipped_dtm.rio.crs)

    stats = zonal_stats(
        mesh_cells_gdf_polygons,
        clipped_dtm.values,
        affine=clipped_dtm.rio.transform(),
        stats="mean",
        nodata=clipped_dtm.rio.nodata
    )

    mesh_cells_gdf_polygons['mean_elevation'] = [s['mean'] if s and 'mean' in s else np.nan for s in stats]
    logger.info("Mean elevation calculated for each mesh node using zonal statistics.")
    
    return mesh_cells_gdf_polygons

def _preprocess_elevation_data(mesh_cells_gdf_polygons: gpd.GeoDataFrame, elev_max: float, elev_min: float,
                              catchment: str):
    """
    Ensure there is no outlying data and all sit within known catchment bounds
    """
    # Check for NaNs
    NaN_count = mesh_cells_gdf_polygons['mean_elevation'].isna().sum()
    if NaN_count > 0:
        logging.warning(f"WARNING: {NaN_count} nodes with NaN value. Replacing with Median.")
        median = mesh_cells_gdf_polygons['mean_elevation'].median()
        mesh_cells_gdf_polygons['mean_elevation'] = mesh_cells_gdf_polygons['mean_elevation'].fillna(median)
    else:
        logger.info(f"No NaN 'mean_elevation' value found. Continuing with Processing.\n")
    
    # Cap values between known catchemnt elevation limits
    column = 'mean_elevation'
    mesh_cells_gdf_polygons = cap_data_between_limits(mesh_cells_gdf_polygons, elev_max, elev_min,
                                                      catchment, column)
    
    return mesh_cells_gdf_polygons

def load_process_elevation_data(dir_path: str, csv_path: str, catchment: str, catchment_gdf: gpd.GeoDataFrame,
                        mesh_cells_gdf_polygons: gpd.GeoDataFrame, elev_max: float, elev_min: float,
                        output_geojson_dir: str, elevation_geojson_path: str, grid_resolution: int = 1000):
    """
    Loads OS elevation DTM data from tile directory using rasterio, flattens it to a DataFrame,
    converts x, y coordinates to lat/lon, and saves to CSV.
    """
    
    # Mosaic (Merge) the DTM Tiles into a single GeoTIFF using rasterio
    mesh_cells_gdf_polygons, dtm_files_to_mosaic = _load_mosaic_elevation(dir_path, mesh_cells_gdf_polygons, csv_path)

    # Load merged DTM
    dtm_raster = rioxarray.open_rasterio(csv_path, masked=True).squeeze()

    # Reproject crs if needed
    if str(dtm_raster.rio.crs) != str(catchment_gdf.crs):
        dtm_raster = dtm_raster.rio.reproject(catchment_gdf.crs)

    # Clip to bounding box using rio.clip_box()
    minx, miny, maxx, maxy = catchment_gdf.total_bounds
    clipped_dtm = dtm_raster.rio.clip_box(minx, miny, maxx, maxy)
    logger.info("DTM raster clipped to catchment bounding box.")
    
    # Aggregate polygon data to averages using rasterstats lib  
    mesh_cells_gdf_polygons = _calculate_polygon_zone_average(mesh_cells_gdf_polygons, clipped_dtm)
    logger.info(f"Loading, merging and aggregating DTM elevation data complete "
                f"for {catchment} catchment.\n")
    
    # Check for outliers and inconsistencies
    mesh_cells_gdf_polygons = _preprocess_elevation_data(mesh_cells_gdf_polygons, elev_max,
                                                        elev_min, catchment)
    
    # Rename columns as needed for subsequent merge (geometry must be polygon not node)
    mesh_cells_gdf_polygons = mesh_cells_gdf_polygons.rename(columns={'geometry': 'polygon_geometry'})
    
    # --- Save polygons for future reference ---
    
    os.makedirs(output_geojson_dir, exist_ok=True)
    output_geojson_path = os.path.join(output_geojson_dir, f"{catchment}_mesh_cells_polygons.geojson")
    
    # Explicitly state which column is geometry (as it is renamed, causing a geojson issue)
    mesh_cells_gdf_polygons = mesh_cells_gdf_polygons.set_geometry('polygon_geometry')
    mesh_gdf_polygons = mesh_cells_gdf_polygons.drop(columns='mean_elevation')
    
     # Save polygons only
    try:
        mesh_gdf_polygons.to_file(output_geojson_path, driver='GeoJSON')
        logger.info(f"Mesh cell polygons saved to: {output_geojson_path}\n")
    except Exception as e:
        logger.error(f"Failed to save mesh cell polygons to GeoJSON: {e}")
        
    # Save polygons and elevation gdf to access for merge

    mesh_gdf_polygons_to_save = mesh_cells_gdf_polygons[['node_id', 'mean_elevation', 'polygon_geometry']].copy()
    
    try:
        mesh_gdf_polygons_to_save.to_file(elevation_geojson_path, driver='GeoJSON')
        logger.info(f"Mesh cell polygons saved to: {elevation_geojson_path}\n")
    except Exception as e:
        logger.error(f"Failed to save mesh cell polygons to GeoJSON: {e}")
    
    return mesh_cells_gdf_polygons, clipped_dtm

# --- Various ---

def cap_data_between_limits(gdf: gpd.GeoDataFrame, max_limit: float, min_limit: float, catchment: str,
                            column_name: str):
    
    # Define bounds
    logging.info(f"Check {catchment} catchment {column_name} sits in known bounds: "
                 f"{min_limit} - {max_limit}...")
    
    # Count if any averages sit outside bounds
    capped_lower_count = (gdf[column_name] < min_limit).sum()
    capped_upper_count = (gdf[column_name] > max_limit).sum()
    
    # Cap to catchment bounds
    if capped_lower_count > 0 or capped_upper_count > 0:
        logger.warning(f"WARNING: Capping {capped_lower_count} nodes below {min_limit:.2f} and "
                       f"{capped_upper_count} nodes above {max_limit:.2f}.")
        
        # Clipping data
        gdf[column_name] = np.clip(gdf[column_name], a_min=min_limit, a_max=max_limit)
        
    else:
        logger.info(f"All 'mean_elevation' values within known bounds ({min_limit:.2f} -"
                    f" {max_limit:.2f}). No capping performed.\n")
    
    return gdf

# --- Slope and Aspect ---

def preprocess_slope_data(slope_gdf: gpd.GeoDataFrame, catchment: str):
    """
    Preprocess slope degrees, sine component and cosine aspect component. Check and fill NaNs and verify
    all slope degrees are within logical bounds.
    """
    # Replace NaNs with average and return warning
    logging.info(f"Check {catchment} catchment slope for NaN values.")
    
    for column in ['mean_slope_degrees', 'mean_aspect_sin', 'mean_aspect_cos']:
        NaN_count = slope_gdf[column].isna().sum()
        if NaN_count > 0:
            logger.warning(F"WARNING: {NaN_count} rows found with NaN {column}. Replacing with Median.\n")
            median = slope_gdf[column].median()
            slope_gdf[column] = slope_gdf[column].fillna(median)
        else:
            logger.info(f"No NaN {column} value found. Continuing with Processing.\n")
    
    # Check all slope values sit in logical bounds (in degrees)
    lower_cap = 0.0  # Slope is always +ve, aspect defined direction
    upper_cap = 80.0
    column = 'mean_slope_degrees'
    
    slope_gdf = cap_data_between_limits(slope_gdf, upper_cap, lower_cap, catchment, column)

    return slope_gdf

def aggregate_slope_and_aspect(mesh_cells_gdf_polygons: gpd.GeoDataFrame, catchment: str,
                                slope_magnitude_deg: gpd.GeoDataFrame, aspect_sin: gpd.GeoDataFrame,
                                aspect_cos: gpd.GeoDataFrame):
    """
    Aggregate the derived slope (degrees) and sine and cosine aspects to 1km grid cell
    resolution using zonal stats to calculate means.
    """
    # Calculate mean slope magnitude
    stats_slope = zonal_stats(
        mesh_cells_gdf_polygons,
        slope_magnitude_deg.values,
        affine=slope_magnitude_deg.rio.transform(),
        stats="mean",
        nodata=slope_magnitude_deg.rio.nodata
    )
    
    mesh_cells_gdf_polygons['mean_slope_degrees'] = [
        slope['mean'] if slope and 'mean' in slope else np.nan for slope in stats_slope]
    logging.info(f"Mean slope (degrees) 1km grid cells calculated for {catchment} catchment.")
    
    # Calculate mean slope aspect sine component
    stats_sin = zonal_stats(
        mesh_cells_gdf_polygons,
        aspect_sin.values,
        affine=aspect_sin.rio.transform(),
        stats="mean",
        nodata=aspect_sin.rio.nodata
    )
    
    mesh_cells_gdf_polygons['mean_aspect_sin'] = [
        sin['mean'] if sin and 'mean' in sin else np.nan for sin in stats_sin]
    logging.info(f"Mean aspect (sine component) 1km grid cells calculated for {catchment} catchment.")
    
    # Calculate mean slope aspect cosine component
    stats_cos = zonal_stats(
        mesh_cells_gdf_polygons,
        aspect_cos.values,
        affine=aspect_cos.rio.transform(),
        stats="mean",
        nodata=aspect_cos.rio.nodata
    )
    
    mesh_cells_gdf_polygons['mean_aspect_cos'] = [
        cosine['mean'] if cosine and 'mean' in cosine else np.nan for cosine in stats_cos]
    logging.info(f"Mean aspect (cosine component) 1km grid cells calculated for {catchment} catchment.")
    
    return mesh_cells_gdf_polygons

def calculate_directional_edges(slope_magnitude_deg, aspect_radians, catchment, mesh_cells_gdf_polygons):
    slope_magnitude_rad = np.deg2rad(slope_magnitude_deg)
    slope_dx = np.sin(aspect_radians) * np.tan(slope_magnitude_rad)
    slope_dy = np.cos(aspect_radians) * np.tan(slope_magnitude_rad)
    
    # Assertion to Check Raster Shape Matches Affine Transform
    assert slope_magnitude_deg.shape == (slope_magnitude_deg.sizes['y'], slope_magnitude_deg.sizes['x']), \
        "Raster shape does not match expected dimensions."
    
    # Copy polygon geometry to create direction weight df
    directional_edge_weights = mesh_cells_gdf_polygons[['geometry']].copy()
    
    # Merge into directional edge reference df
    stats_dx = zonal_stats(
        directional_edge_weights,
        slope_dx.values,
        affine=slope_dx.rio.transform(),
        stats="mean",
        nodata=slope_dx.rio.nodata
    )

    stats_dy = zonal_stats(
        directional_edge_weights,
        slope_dy.values,
        affine=slope_dy.rio.transform(),
        stats="mean",
        nodata=slope_dy.rio.nodata
    )

    directional_edge_weights['mean_slope_dx'] = [
        dx['mean'] if dx and 'mean' in dx else np.nan for dx in stats_dx
    ]
    directional_edge_weights['mean_slope_dy'] = [
        dy['mean'] if dy and 'mean' in dy else np.nan for dy in stats_dy
    ]

    logger.info(f"Directional slope (dx, dy) aggregated to 1km mesh for {catchment}.")
    
    # Confirm crs
    if directional_edge_weights.crs.to_epsg() != 27700:
        directional_edge_weights = directional_edge_weights.to_crs(27700)
    
    # Add easting and northing from centroid
    directional_edge_weights['easting'] = directional_edge_weights.geometry.centroid.x
    directional_edge_weights['northing'] = directional_edge_weights.geometry.centroid.y
        
    # Find lat / lon
    directional_edge_weights = easting_northing_to_lat_long(directional_edge_weights)
            
    # final output: mean_elevation, mean_slope_deg, mean_aspect_sin, mean_aspect_cos, easting, northing
    logger.info(f"Slope and aspect derivation and preprocessing complete for {catchment} catchment.\n")
    
    return directional_edge_weights

def derive_slope_data(high_res_raster: xr.DataArray, mesh_cells_gdf_polygons: gpd.GeoDataFrame,
                      catchment: str, direction_output_path: str, slope_output_path: str):
    """
    Derives slope magnitude and aspect (transformed to sine/cosine components)
    from high-resolution DEM raster (used in elevation) and aggregate to 1km mesh cells.
    """
    logging.info(f"Deriving slope magnitude and direction data for {catchment} catchment...\n")
    
    print(type(high_res_raster))
    print(high_res_raster)
    
    # Ensure DataArray has no band dimension
    if 'band' in high_res_raster.dims:
        high_res_raster = high_res_raster.squeeze()
        
    # Derive slope and aspect using xarray-spatial
    slope_magnitude_deg = xrs.slope(high_res_raster)
    logging.info(f"Slope magnitude derived in degrees from DEM.")
    
    aspect_degrees = xrs.aspect(high_res_raster)
    logging.info(f"Slope aspect derived (0–360°) from DEM.\n")
    
    # Transform Aspect into radians then Circularity (with both Sine and Cosine Components)
    aspect_radians = np.deg2rad(aspect_degrees)
    aspect_sin = np.sin(aspect_radians).rename("aspect_sin")
    aspect_cos = np.cos(aspect_radians).rename("aspect_cos")
    logging.info("Aspect converted from degrees to radians for sin/cos component transform.")
    
    # Ensure CRS match before aggregation
    if str(mesh_cells_gdf_polygons.crs) != str(high_res_raster.rio.crs):
        logger.warning(f"Reprojecting mesh cells from ({mesh_cells_gdf_polygons.crs}) to"
                       f" ({high_res_raster.rio.crs}).\n")
        mesh_cells_gdf_polygons = mesh_cells_gdf_polygons.to_crs(high_res_raster.rio.crs)
    
    # --- Aggregate to 1km Mesh Cells using zonal statistics lib ---
    
    slope_gdf = aggregate_slope_and_aspect(mesh_cells_gdf_polygons, catchment,
                                           slope_magnitude_deg, aspect_sin, aspect_cos)

    # --- Slope and Aspect Preprocessing ---

    slope_gdf = preprocess_slope_data(slope_gdf, catchment)
    
    # --- Calculate Gradient Componenets for GNN Directional Edge Weights --
    
    directional_edge_weights = calculate_directional_edges(slope_magnitude_deg, aspect_radians,
                                                            catchment, mesh_cells_gdf_polygons)
    
    # --- Save files ---
    directional_edge_weights.to_csv(direction_output_path, index=False)
    logger.info(f"Direction Weights csv saved to {direction_output_path}.")
    
    slope_gdf.to_csv(slope_output_path, index=False)
    logger.info(f"Slope magnitue and aspect csv saved to {slope_output_path}.")
    
    return slope_gdf, directional_edge_weights

# --- Geology Feature Set ---

def _loop_geology_subdirectories(subdirs, columns_of_interest, mesh_crs, catchment):
    """
    Load simplified geometries from subdirectories
    """
    logger.info(f"Loading simplified geology geometries from {catchment} catchment...")
    
    layer_suffixes = {
        "bedrock": "_bedrock.shp",
        "superficial": "_superficial.shp"
    }
        
    simplified_geology = {layer: [] for layer in layer_suffixes}
    
    # Loop through subdirectories
    for subdir in subdirs:
        for layer, suffix in layer_suffixes.items():
            for shp in subdir.glob(f"*{suffix}"):
                try:
                    # Only read relevant columns + geometry (avoiding unnecessary compiutation)
                    wanted_cols = columns_of_interest.get(layer, []) + ["geometry"]
                    gdf = gpd.read_file(shp, dtype={"MAX_EPOCH": "str"})[wanted_cols]

                    # Reproject if necessary
                    if gdf.crs != mesh_crs:
                        gdf = gdf.to_crs(mesh_crs)

                    simplified_geology[layer].append(gdf)
                except Exception as e:
                    logging.warning(f"[ERROR] Failed to load {shp.name}: {e}")
    
    return simplified_geology

def _merge_geology_layers(simplified_geology, mesh_crs):
    """
    Merge all individual layers into one concatenated gdf
    """
    # Initialise feature layers dict
    final_layers = {}
    
    logging.info(f"Merging simlpified geology layers into concatenated groups...\n")
    for name, gdf_list in simplified_geology.items():
        if gdf_list:
            merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=mesh_crs)
            final_layers[name] = merged
            logging.info(f"[OK] Loaded and simplified '{name}': {len(merged)} features")
        else:
            logging.info(f"[SKIPPED] No features found for: {name}")

    return final_layers

def _map_geo_cols_to_cats(bedrock_gdf: gpd.GeoDataFrame, superficial_gdf: gpd.GeoDataFrame):
    """
    Map heterogeous and highly imbalanced categories to main categorisations
    while retaining hydrological meaning.
    """
    # Define bedrock gdf column mappings
    bedrock_map = {
        'SEDIMENTARY': 'sedimentary',
        'IGNEOUS': 'igneous',
        'SEDIMENTARY AND IGNEOUS': 'other',
        'METAMORPHIC': 'other'
    }

    # Apply bedrock mappings
    original_bed_cats = bedrock_gdf["RCS_ORIGIN"].nunique()
    bedrock_gdf["geo_bedrock_type"] = bedrock_gdf["RCS_ORIGIN"].map(bedrock_map).fillna("unknown")
    simplified_bed_cats = bedrock_gdf["geo_bedrock_type"].nunique()
    
    logging.info(f"[BEDROCK] Mapped RCS_ORIGIN from {original_bed_cats} to {simplified_bed_cats} categories.")
    logging.info(f'[BEDROCK] Category distribution:\n\n {bedrock_gdf["geo_bedrock_type"].value_counts()}\n')

    # Define superficial gdf column mappings
    superficial_map = {
        # main groups
        'PEAT': 'organic',
        'DIAMICTON': 'diamicton',
        'DIAMICTON, SAND AND GRAVEL': 'diamicton',

        # course
        'SAND': 'coarse',
        'GRAVEL': 'coarse',
        'SAND AND GRAVEL': 'coarse',
        'GRAVEL, SAND AND SILT': 'coarse',
        'SAND, GRAVEL AND BOULDERS': 'coarse',
        'GRAVEL, SAND, SILT AND CLAY': 'coarse',
        'ROCK FRAGMENTS, ANGULAR, UNDIFFERENTIATED SOURCE ROCK': 'coarse',
        'MUD, SANDY': 'coarse',

        # mixed_fines
        'CLAY, SILT, SAND AND GRAVEL': 'mixed_fines',
        'SILT, SAND AND GRAVEL': 'mixed_fines',
        'SAND, SILT AND CLAY': 'mixed_fines',
        'CLAY, SAND AND GRAVEL': 'mixed_fines',
        'CLAY, SILT AND SAND': 'mixed_fines',

        # Residual fine-grained / rare / uncertain
        'CLAY': 'other',
        'CLAY AND SILT': 'other',
        'SILT AND CLAY': 'other',
        'CLAY, SILTY': 'other',
        'SILT': 'other',
        'WATER, TYPE UNSPECIFIED': 'other',
        'UNKNOWN/UNCLASSIFIED ENTRY': 'other'
    }

    # Apply superficial mappings
    original_sup_cats = superficial_gdf["RCS_D"].nunique()
    superficial_gdf["geo_superficial_type"] = superficial_gdf["RCS_D"].map(superficial_map).fillna("unknown")
    simplified_sup_cats = superficial_gdf["geo_superficial_type"].nunique()

    logging.info(f"[SUPERFICIAL] Mapped RCS_D from {original_sup_cats} to {simplified_sup_cats} categories.")
    logging.info(f'[SUPERFICIAL] Category distribution:\n\n {superficial_gdf["geo_superficial_type"].value_counts()}\n')
    
    # Drop old columns
    bedrock_gdf = bedrock_gdf.drop(columns=['RCS_ORIGIN'])
    superficial_gdf = superficial_gdf.drop(columns=['RCS_D'])

    return bedrock_gdf, superficial_gdf

def _align_geology_with_mesh(mesh_cells_gdf_polygons, bedrock_gdf, superficial_gdf, catchment):
    """
    Merge spatial geolgoy with main model mesh using intersecting geology then aggregate to 1km values
    using modal average. Merge aggregated data back to single df.
    """
    logging.info(f"Merging geology data together for {catchment} catchment...\n")
    
    # Merge to mesh by intersecting points
    bedrock_joined = gpd.sjoin(mesh_cells_gdf_polygons, bedrock_gdf, how="left", predicate="intersects")
    superficial_joined = gpd.sjoin(mesh_cells_gdf_polygons, superficial_gdf, how="left", predicate="intersects")

    # Calc mode label per 1km cell for bedrock
    bedrock_mode = (
        bedrock_joined[["node_id", "geo_bedrock_type"]].dropna().groupby("node_id")
        .agg(lambda x: x.mode().iloc[0])
    )
    logging.info(f"Bedrock geology data merged to mesh.")

    # Calc mode label per 1km cell for superficial
    superficial_mode = (
        superficial_joined[["node_id", "geo_superficial_type"]].dropna().groupby("node_id")
        .agg(lambda x: x.mode().iloc[0])
    )
    logging.info(f"Superficial geology data merged to mesh.")

    # Merge both back to mesh
    mesh_geology_df = (
        mesh_cells_gdf_polygons[["node_id", "geometry"]]
        .merge(bedrock_mode, on="node_id", how="left")
        .merge(superficial_mode, on="node_id", how="left")
    )
    logging.info(f"Full geology df built for {catchment} catchment.\n")
    
    # Fill missing values
    mesh_geology_df['geo_superficial_type'] = mesh_geology_df['geo_superficial_type'].fillna("other")
    mesh_geology_df['geo_bedrock_type'] = mesh_geology_df['geo_bedrock_type'].fillna("other")

    return mesh_geology_df

def load_and_process_geology_layers(base_dir: str, mesh_crs: str, columns_of_interest: list,
                                    mesh_cells_gdf_polygons: gpd.GeoDataFrame, perm_dir: str,
                                    geo_output_dir: str, catchment: str):
    """
    Recursively loads and simplifies geological shapefiles (e.g., bedrock, superficial)
    from all 'ew*' or 'sc*' subdirectories, retaining only columns of interest.
    Automatically reprojects to match mesh CRS.
    """
    base_path = Path(base_dir)
    subdirs = [p for p in base_path.iterdir() if p.is_dir() and re.match(r'^(ew|sc)', p.name)]

    # --- Load data in from sub directories and merge together ---
    
    simplified_geology = _loop_geology_subdirectories(subdirs, columns_of_interest,
                                                      mesh_crs, catchment)
    
    final_layers = _merge_geology_layers(simplified_geology, mesh_crs)

    # --- Split into individual df's by type and map sparse categories to final ---
    
    bedrock_gdf = final_layers["bedrock"]
    superficial_gdf = final_layers["superficial"]
    
    bedrock_gdf, superficial_gdf = _map_geo_cols_to_cats(bedrock_gdf, superficial_gdf)
    
    # --- Merge with mesh by intersecting areas and aggregate to 1km modal values ---
    
    mesh_geology_df = _align_geology_with_mesh(mesh_cells_gdf_polygons, bedrock_gdf, superficial_gdf, catchment)
    
    # --- Pull in permeability data ---
    
    bedrock_perm_agg, superficial_perm_agg = ingest_and_process_permeability(perm_dir, mesh_cells_gdf_polygons)
    
    mesh_geology_df = (
        mesh_geology_df
        .merge(bedrock_perm_agg, on="node_id", how="left")
        .merge(superficial_perm_agg, on="node_id", how="left")
    )
    
    # --- Add lat and lon to plot ---
    
    # Project to lat/lon (EPSG:4326)
    centroids_latlon = mesh_geology_df.geometry.to_crs("EPSG:4326").centroid
    
    # Add as explicit lat and lon cols in df
    mesh_geology_df["lon"] = centroids_latlon.x
    mesh_geology_df["lat"] = centroids_latlon.y
    
    # --- Save df ---
    geology_filename = geo_output_dir + '/geology_df.csv'
    mesh_geology_df.to_csv(geology_filename)
    
    logger.info(f"Full geology dataframe saved to {geology_filename}.\n")

    return mesh_geology_df

def get_geo_feats():
    """
    Get interactive mapping features for geology map.
    """
    geo_superficial_labels = {
        'coarse':'Coarse (sand & gravel)',
        'mixed_fines':'Mixed fines',
        'organic':'Organic (peat)',
        'diamicton':'Diamicton',
        'superficial_other':'Other (Superficial)'
    }

    geo_bedrock_labels = {
        'sedimentary':'Sedimentary',
        'igneous':'Igneous',
        'bedrock_other':'Other (Bedrock)'
    }

    geo_bedrock_colors = {'sedimentary': '#8e412e', 'igneous': '#4d85ba', 'bedrock_other': '#999999'}

    geo_superficial_colors = {'coarse': '#8e412e', 'mixed_fines': '#4d85ba', 'organic': '#009E73',
                            'diamicton': "#DFD53E", 'superficial_other': '#999999'}

    feature_category_colors = {'geo_superficial_type': geo_superficial_colors, 'geo_bedrock_type': geo_bedrock_colors}
    feature_category_labels = {'geo_superficial_type': geo_superficial_labels, 'geo_bedrock_type': geo_bedrock_labels}
    layer_labels = {"geo_superficial_type": "Superficial Type", "geo_bedrock_type": "Bedrock Type"}
    
    return feature_category_colors, feature_category_labels, layer_labels

# --- Bedrock and Superficial Permeability ---

def _encode_ordinal_perm(row, perm_encoding):
    """
    Encode each row to ordinal mapping and return average if possible or
    available row if not.
    """
    max_ = perm_encoding.get(row["MAX_PERM"], None)
    min_ = perm_encoding.get(row["MIN_PERM"], None)
    
    if max_ is not None and min_ is not None:
        return (max_ + min_) / 2
    elif max_ is not None:
        return max_
    elif min_ is not None:
        return min_
    else:
        return None

def _get_mode_or_none(x):
    mode_vals = x.mode()
    return mode_vals.iloc[0] if not mode_vals.empty else None

def _aggregate_perm_by_node_id(bedrock_join, superficial_join):
    """
    Aggregate each gdf to 1km mesh
    """
    # Aggregate bedrock permeability by node_id
    bedrock_perm_agg = (
        bedrock_join.groupby("node_id")
        .agg({
            "FLOW_TYPE": _get_mode_or_none,
            "bedrock_perm_avg": "mean",
        }).rename(columns={"FLOW_TYPE": "bedrock_flow_type"})
    )
    logger.info(f"Aggregated bedrock permeability: {len(bedrock_perm_agg)} nodes.")

    # Aggregate superficial permeability by node_id
    superficial_perm_agg = (
        superficial_join.groupby("node_id")
        .agg({
            "FLOW_TYPE": _get_mode_or_none,
            "superficial_perm_avg": "mean",
        }).rename(columns={"FLOW_TYPE": "superficial_flow_type"})
    )
    logger.info(f"Aggregated superficial permeability: {len(superficial_perm_agg)} nodes.\n")
    
    return bedrock_perm_agg, superficial_perm_agg

def _encode_agg_clean_perm(bedrock_join, superficial_join):
    """
    Do full preprocessing of permeability data to build final dfs to merge
    into main geology df.
    """
    # --- Encode max and min to ordinal ---

    logger.info(f"Encoding categorical permeability to ordinal...")
    perm_encoding = {"Very Low": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}

    bedrock_join["bedrock_perm_avg"] = bedrock_join.apply(lambda r: _encode_ordinal_perm(r, perm_encoding), axis=1)
    superficial_join["superficial_perm_avg"] = superficial_join.apply(lambda r: _encode_ordinal_perm(r, perm_encoding), axis=1)

    logger.info("Encoding complete.\n")

    # --- Aggregate permeability by node_id ---

    bedrock_perm_agg, superficial_perm_agg = _aggregate_perm_by_node_id(bedrock_join, superficial_join)
    logger.info("Permeability aggregation complete.")

    # --- Clean up final dfs ---

    bedrock_med = bedrock_perm_agg["bedrock_perm_avg"].median()
    bedrock_perm_agg["bedrock_perm_avg"] = bedrock_perm_agg["bedrock_perm_avg"].fillna(bedrock_med).round(2)
    bedrock_perm_agg["bedrock_flow_type"] = bedrock_perm_agg["bedrock_flow_type"].fillna("Mixed")

    superficial_med = superficial_perm_agg["superficial_perm_avg"].median()
    superficial_perm_agg["superficial_perm_avg"] = superficial_perm_agg["superficial_perm_avg"].fillna(superficial_med).round(2)
    superficial_perm_agg["superficial_flow_type"] = superficial_perm_agg["superficial_flow_type"].fillna("Mixed")

    logger.info("Final flow type and permeability cleaning complete.\n")
    return bedrock_perm_agg, superficial_perm_agg

def ingest_and_process_permeability(perm_dir: str, mesh_cells_gdf_polygons: gpd.GeoDataFrame):
    """
    Read in BGS Permeability data that corresponds to the bedrock and superficial categorisations
    """
    # Get data paths to load in
    bedrock_path = perm_dir + "/BedrockPermeability_v8.shp"
    superficial_path = perm_dir + "/SuperficialPermeability_v8.shp"

    # --- Load permeability shapefiles ---

    # Define columns of interest
    cols = ['FLOW_TYPE', 'MAX_PERM', 'MIN_PERM', 'geometry']

    # Read in bedrock permeability data
    logger.info("Loading bedrock permeability shapefile...")
    bedrock_perm = gpd.read_file(bedrock_path)[cols]

    # Read in superficial permeability data
    logger.info("Loading superficial permeability shapefile...\n")
    superficial_perm = gpd.read_file(superficial_path)[cols]

    logger.info(f"Loaded bedrock permeability: {len(bedrock_perm)} features.")
    logger.info(f"Loaded superficial permeability: {len(superficial_perm)} features.\n")

    # --- Reproject crs if required ---

    mesh_crs = mesh_cells_gdf_polygons.crs 

    if bedrock_perm.crs != mesh_crs:
        logger.info("Reprojecting bedrock permeability to mesh CRS...\n")
        bedrock_perm = bedrock_perm.to_crs(mesh_crs)
        
    if superficial_perm.crs != mesh_crs:
        logger.info("Reprojecting superficial permeability to mesh CRS...\n")
        superficial_perm = superficial_perm.to_crs(mesh_crs)

    # --- Prepare mesh polygons for join ---

    polygon_gdf = mesh_cells_gdf_polygons.copy().drop(columns=['mean_elevation'])

    # --- Spatial joins by type ---

    bedrock_join = gpd.sjoin(
        polygon_gdf,
        bedrock_perm,
        how='left',
        predicate='intersects'
    )
    logger.info(f"Joined bedrock permeability: {len(bedrock_join)} rows.")

    superficial_join = gpd.sjoin(
        polygon_gdf,
        superficial_perm,
        how='left',
        predicate='intersects'
    )
    logger.info(f"Joined superficial permeability: {len(superficial_join)} rows.\n")

    # --- Encode, aggregate and clean dfs ---

    bedrock_perm_agg, superficial_perm_agg = _encode_agg_clean_perm(bedrock_join, superficial_join)

    return bedrock_perm_agg, superficial_perm_agg

# --- Save final static data csv ---

def save_final_static_data(static_features: pd.DataFrame, dir_path: str):
    # Define columns to drop
    if 'geometry_x' in static_features.columns:
        cols = ['geometry_x', 'geometry_y', 'easting', 'northing', 'lon', 'lat']
    else:
        cols = ['geometry', 'polygon_geometry', 'easting', 'northing', 'lon', 'lat']
    
    # Drop unneeded features and set index
    static_features = static_features.drop(columns=cols)
    static_features = static_features.set_index('node_id')
    
    # Save static df to csv
    save_path = dir_path + 'final_static_df.csv'
    static_features.to_csv(save_path)

    logger.info(f"Final merged static dataframe saved to {save_path}")

