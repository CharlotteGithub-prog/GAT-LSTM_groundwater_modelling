# Import Libraries
import os
import sys
import logging
import rasterio
import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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
    catchment_gdf, _, minx, miny, maxx, maxy = find_catchment_boundary(
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
    
    # Save to csv for preprocessing
    land_cover_df.to_csv(csv_path, index=False)
    logger.info(f"Land Cover data succesfully saved to {csv_path}.")
    
    return land_cover_df, catchment_gdf

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

def load_elevation_data():
    pass

def load_slope_data():
    pass