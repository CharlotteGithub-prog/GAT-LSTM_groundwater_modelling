# Import Libraries
import os
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

# Land Cover
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
    _, _, minx, miny, maxx, maxy = find_catchment_boundary(
        catchment=catchment,
        shape_filepath=shape_filepath,
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

# Land Cover
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

# Elevation
def load_mosaic_elevation(dir_path: str, mesh_cells_gdf_polygons: gpd.GeoDataFrame, csv_path: str):

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

# Elevation
def calculate_polygon_zone_average(mesh_cells_gdf_polygons: gpd.GeoDataFrame, clipped_dtm: gpd.GeoDataFrame):
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

# Elevation
def preprocess_elevation_data(mesh_cells_gdf_polygons: gpd.GeoDataFrame, elev_max: float, elev_min: float,
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

# Elevation
def load_process_elevation_data(dir_path: str, csv_path: str, catchment: str, catchment_gdf: gpd.GeoDataFrame,
                        mesh_cells_gdf_polygons: gpd.GeoDataFrame, elev_max: float, elev_min: float,
                        grid_resolution: int = 1000):
    """
    Loads OS elevation DTM data from tile directory using rasterio, flattens it to a DataFrame,
    converts x, y coordinates to lat/lon, and saves to CSV.
    """
    
    # Mosaic (Merge) the DTM Tiles into a single GeoTIFF using rasterio
    mesh_cells_gdf_polygons, dtm_files_to_mosaic = load_mosaic_elevation(dir_path, mesh_cells_gdf_polygons, csv_path)

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
    mesh_cells_gdf_polygons = calculate_polygon_zone_average(mesh_cells_gdf_polygons, clipped_dtm)
    logger.info(f"Loading, merging and aggregating DTM elevation data complete "
                f"for {catchment} catchment.\n")
    
    # Check for outliers and inconsistencies
    mesh_cells_gdf_polygons = preprocess_elevation_data(mesh_cells_gdf_polygons, elev_max,
                                                        elev_min, catchment)
    
    # Rename columns as needed for subsequent merge (geometry must be polygon not node)
    mesh_cells_gdf_polygons = mesh_cells_gdf_polygons.rename(columns={'geometry': 'polygon_geometry'})
    
    return mesh_cells_gdf_polygons, clipped_dtm

# Various
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

# Slope and Aspect
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
    
# Slope and Aspect
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

# Slope and Aspect
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

# Slope and Aspect 
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
    