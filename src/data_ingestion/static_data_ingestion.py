# Import Libraries
import os
import sys
import glob
import logging
import rasterio
import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
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

# TODO: Make resolution dynamic for land cover

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
    # Define bounds
    logging.info(f"Check {catchment} catchment elevation sits in known bounds: "
                 f"{elev_min} mAOD - {elev_max} mAOD...")
    
    # Count if any averages sit outside bounds
    capped_lower_count = (mesh_cells_gdf_polygons['mean_elevation'] < elev_min).sum()
    capped_upper_count = (mesh_cells_gdf_polygons['mean_elevation'] > elev_max).sum()
    
    # Cap to catchment bounds
    if capped_lower_count > 0 or capped_upper_count > 0:
        logger.warning(f"WARNING: Capping {capped_lower_count} nodes below {elev_min:.2f} mAOD and "
                       f"{capped_upper_count} nodes above {elev_max:.2f} mAOD.")
        
        mesh_cells_gdf_polygons['mean_elevation'] = np.clip(
            mesh_cells_gdf_polygons['mean_elevation'],
            a_min=elev_min,
            a_max=elev_max
        )
        
    else:
        logger.info(f"All 'mean_elevation' values within known bounds ({elev_min:.2f} - {elev_max:.2f} mAOD). No capping performed.\n")
        
    # Check for NaNs
    NaN_count = mesh_cells_gdf_polygons['mean_elevation'].isna().sum()
    if NaN_count > 0:
        logging.warning(f"WARNING: {NaN_count} nodes with NaN value.")
    
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
    load_mosaic_elevation(dir_path, mesh_cells_gdf_polygons, csv_path)

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
    
    return mesh_cells_gdf_polygons
    
def load_slope_data():
    pass