# Import Libraries
import os
import logging
import rasterio
import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_land_cover_data(tif_path: str, csv_path: str):
    """
    Loads land cover data from GeoTIFF file using xarray, flattens it to a DataFrame,
    converts x, y coordinates to lat/lon, and saves to CSV.
    """
    # Load land cover data from GeoTIFF using xarray and select band
    land_cover_ds = rioxarray.open_rasterio(tif_path, masked=True)
    
    if 'band' in land_cover_ds.dims and len(land_cover_ds.coords['band']) > 1:
        land_cover_ds = land_cover_ds.sel(band=land_cover_ds.coords['band'].values[0])
        logging.info(f"Multiple land cover bands found, selecting first.")
    
    # Remove the band dimension if it's still there
    land_cover_ds = land_cover_ds.squeeze()
    
    # Store original CRS (before any transformations are applied)
    source_crs = land_cover_ds.rio.crs
    
    # Convert the xarray DataArray to a pandas DataFrame and rename cols to lat/lon
    land_cover_df = land_cover_ds.to_dataframe(name='land_cover_code').reset_index()
    if 'lon' in land_cover_df.columns and 'lat' in land_cover_df.columns:
        land_cover_df.rename(columns={'lon': 'x', 'lat': 'y'}, inplace=True)

    # Convert x, y (BNG) to Lat/Lon (WGS84)
    geometry = [Point(xy) for xy in zip(land_cover_df['x'], land_cover_df['y'])]
    gdf = gpd.GeoDataFrame(land_cover_df, geometry=geometry, crs=source_crs)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    
    # Extract longitude (x) and latitude (y) from the transformed geometries
    land_cover_df['lon'] = gdf_wgs84.geometry.x
    land_cover_df['lat'] = gdf_wgs84.geometry.y
    
    # Drip unneeded columns and check data type
    land_cover_df = land_cover_df.drop(['band', 'spatial_ref'], axis=1)
    land_cover_df['land_cover_code'] = land_cover_df['land_cover_code'].astype(int)
    
    # Check for NaN
    NaN_count = land_cover_df['land_cover_code'].isna().sum()
    if NaN_count > 0:
        logging.info(f"Total missing land cover codes: {NaN_count}")
    
    # Save to csv for preprocessing
    land_cover_df.to_csv(csv_path, index=False)
    return land_cover_df

def load_elevation_data():
    pass

def load_slope_data():
    pass