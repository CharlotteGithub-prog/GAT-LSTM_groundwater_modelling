import sys
import logging
import pandas as pd
from pyproj import CRS
import geopandas as gpd
from typing import Union
from pyproj import Transformer
from shapely.geometry import Point

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

# Convert GWL alphanumeric OS grid ref to easting, northing, lat, lon
def grid_ref_to_coords(grid_ref: str, grid_letters: dict):
    """
    Convert OS Grid references into espg 27700 easting and northing values, then into
    espg 4326 coordinate ref system (longitude and latitude) for future visualisaations.
    """
    # Transformer to WGS84 (for lat/long coordinate transformations)
    transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
    
    # Clean grid references
    grid_ref = grid_ref.strip().upper()
    
    # Split to letter and numeric
    letter_only = grid_ref[:2]  # e.g. "NY" (from NY123456)
    numeric_only = grid_ref[2:]  # e.g. "123456" (from NY123456)
    
    # Check expected form (paired values)
    if len(numeric_only) % 2 != 0:
        raise ValueError(f"Invalid grid reference: {grid_ref}")
    
    easting_base, northing_base = grid_letters[letter_only]  # e.g. "300000, 500000" from NY (check ref file)
    half = len(numeric_only) // 2  # Seperate easting and northing (e.g. "123", "456")
    easting_offset = int(numeric_only[:half].ljust(5, '0'))  # e.g. "123" -> 12300 m
    northing_offset = int(numeric_only[half:].ljust(5, '0'))  # e.g. "456" -> 45600 m
    
    easting = int(easting_base + easting_offset)  # e.g. 300000 + 12300 = 312300 m
    northing = int(northing_base + northing_offset)  # e.g. 500000 + 45600 = 545600 m
    lon, lat = transformer.transform(easting, northing)  # Transform 312300, 545600 to lat, long (epsg:4326)
    
    return pd.Series([easting, northing, lat, lon])

# Reprojection helper
def _reproject_df_coords(df: pd.DataFrame, input_x_col: str, input_y_col: str,
                         source_crs: Union[str, int, CRS], target_crs_val: Union[str, int, CRS],
                         transformed_x_col: str, transformed_y_col: str):
    """
    Internal helper to reproject x, y coordinates in a DataFrame.
    Assumes input_x_col and input_y_col contain the coordinates in source_crs_val.
    Adds new columns with reprojected coordinates in target_crs_val.
    """
    # Check columns given as input for reprojection actually exist in df
    if input_x_col not in df.columns:
        raise ValueError(f"Input column '{input_x_col}' not found in the DataFrame.")
    elif input_y_col not in df.columns:
        raise ValueError(f"Input column '{input_y_col}' not found in the DataFrame.")
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df[input_x_col], df[input_y_col])]
    
    # Set the CRS of the input coordinates
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=source_crs)

    # Perform the transformation
    gdf_reproj = gdf.to_crs(target_crs_val)

    # Extract transformed coordinates and add to the original DataFrame
    df_result = df.copy()
    df_result.loc[:, transformed_x_col] = gdf_reproj.geometry.x
    df_result.loc[:, transformed_y_col] = gdf_reproj.geometry.y

    # Return the df with the new, requested columns
    return df_result

# General Conversion
def lat_long_to_easting_northing(input_df: pd.DataFrame, output_csv_path: str = None, lon_col_name: str = 'lon',
                                 lat_col_name: str = 'lat', output_easting_col: str = 'easting',
                                 output_northing_col: str = 'northing'):
    """
    Take an input df with spatial reference columns in latitude and longitute and derive
    easting and northing columns using reprojection helper.
    """
    logger.info(f"Converting '{lon_col_name}'/'{lat_col_name}' to '{output_easting_col}'/'{output_northing_col}'...")
    
    reprojected_df = _reproject_df_coords(
        df=input_df,   
        input_x_col=lon_col_name,
        input_y_col=lat_col_name,
        source_crs=4326,  # WGS84
        target_crs_val=27700, # British National Grid
        transformed_x_col=output_easting_col,
        transformed_y_col=output_northing_col
    )
    
    # If output path is given then save, otherwise just return df
    if output_csv_path:
        reprojected_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved to {output_csv_path}\n")
    
    return reprojected_df

# General Conversion
def easting_northing_to_lat_long(input_df: pd.DataFrame, output_csv_path: str = None, easting_col: str = 'easting',
                                 northing_col: str = 'northing', output_lon_col: str = 'lon',
                                 output_lat_col: str = 'lat'):
    """
    Take an input df with spatial reference columns in easting and northing and derive
    latitude and longitute columns using reprojection helper.
    """
    logger.info(f"Converting '{easting_col}'/'{northing_col}' to '{output_lon_col}'/'{output_lat_col}'...")
    
    reprojected_df = _reproject_df_coords(
        df=input_df,   
        input_x_col=easting_col,
        input_y_col=northing_col,
        source_crs=27700, # British National Grid
        target_crs_val=4326,  # WGS84
        transformed_x_col=output_lon_col,
        transformed_y_col=output_lat_col
    )
    
    # If output path is given then save, otherwise just return df
    if output_csv_path:
        reprojected_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved to {output_csv_path}\n")
    
    return reprojected_df

# Check shapefile bounds
def find_catchment_boundary(catchment:str, shape_filepath: str, required_crs: int = 27700):
    """
    Find the orthogonal catchment bounds to avoid excess data preprocessing at this stage. Return the
    min and max of the easting/northing (crs:27700) or lon/lat (crs:4326).
    """
    logger.info(f"Finding {catchment} catchment spatial boundaries...\n")
    
    # Load spatial boundary shape file
    logger.info(f"Loading  boundary from: {shape_filepath}")
    catchment_polygon = gpd.read_file(shape_filepath)
    catchment_polygon = catchment_polygon.to_crs(epsg=required_crs) # Ensure it is in target crs

    # Check polygon geometry -> if shapefile has multiple features then dissolve them
    if len(catchment_polygon) > 1:
        logger.info("Multiple polygons found in the catchment boundary. Merging into a single geometry.")
        catchment_geometry = catchment_polygon.unary_union
    else:
        logger.info("Single polygon found in the catchment boundary.")
        catchment_geometry = catchment_polygon.geometry.iloc[0]
    
    # Get the bounds of the catchment and set grid resolution (in km)
    minx, miny, maxx, maxy = catchment_polygon.total_bounds
    logger.info(f"Catchment bounding box: min_x={minx}, min_y={miny}, max_x={maxx}, max_y={maxy}\n")
    
    return catchment_polygon, catchment_geometry, minx, miny, maxx, maxy
