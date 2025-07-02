# Import Libraries
import os
import sys
import logging
import pandas as pd
import geopandas as gpd
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

def reorder_static_columns(df: pd.DataFrame):
    """ 
    Reorder columns in df to put geometry before features
    """
    desired_order = ['node_id', 'geometry', 'polygon_geometry', 'easting', 'northing',
                    'lon', 'lat', 'land_cover_code', 'mean_elevation',
                    'mean_slope_degrees', 'mean_aspect_sin', 'mean_aspect_cos']

    return df[desired_order]

def snap_stations_to_mesh(station_list_path: str, polygon_geometry_path: str, output_path: str,
                          mesh_nodes_gdf: gpd.GeoDataFrame, catchment: str):
    """
    tbd
    """
    logger.info(f"Snapping {catchment} catchment stations to mesh centroids...\n")
    
    # --- Read in data required for merges ---
    
    # Reading in polygon geometry data
    full_path = polygon_geometry_path + f"{catchment}_mesh_cells_polygons.geojson"
    polygon_geometry_mesh = gpd.read_file(full_path)
    
    # Reading in station list with coords to snap stations to mesh 
    station_list_with_coords = pd.read_csv(station_list_path)
    
    # Use .apply() to create Point objects for each row and convert df -> gdf
    geometry = [Point(xy) for xy in zip(station_list_with_coords['easting'], station_list_with_coords['northing'])]
    crs_nat_grid = "EPSG:27700"
    station_list_gdf = gpd.GeoDataFrame(station_list_with_coords, geometry=geometry, crs=crs_nat_grid)

    # --- Perform spatial merge for gdf ---
    
    logger.info(f"Spatial merging catchment stations within polygon geometry.")

    station_node_mapping_temp = gpd.sjoin(
        station_list_gdf,
        polygon_geometry_mesh[['geometry', 'node_id']],
        how='left',
        op='within'
    )

    # --- Perform regular merge to retain RHS gdf polygon geometry ---
    
    logger.info(f"Merging catchment stations to retain original polygon.")
    
    mesh_geometries = polygon_geometry_mesh[['node_id', 'geometry']].rename(columns={
        'geometry': 'polygon_geometry'})
    
    station_node_mapping = pd.merge(
        station_node_mapping_temp,
        mesh_geometries,
        on='node_id',
        how='left'
    )

    # --- Perform regular merge to merge in centroids
    
    logger.info(f"Merging catchment stations with centroid geometry.")
    
    centroid_gdf = mesh_nodes_gdf[['node_id', 'easting', 'northing']].rename(columns={
        'easting': 'easting_centroid', 'northing': 'northing_centroid'})

    station_node_mapping = pd.merge(
        station_node_mapping,
        centroid_gdf,
        on='node_id',
        how='left'
    )
    
    logger.info(f"Replacing original station geometry with nearest centroid geometry.")

    # Drop original columns and replace with merged centroid columns
    station_node_mapping = station_node_mapping.drop(columns=['easting', 'northing'])
    station_node_mapping = station_node_mapping.rename(columns={
        'easting_centroid': 'easting', 'northing_centroid': 'northing'})

    # Select which columns to keep for future mapping
    station_node_mapping = station_node_mapping[['node_id', 'station_id', 'station_name', 'easting',
                                                'northing']].drop_duplicates().dropna(subset=['node_id'])
    
    logger.info(f"Converting spatial mapping DataFrame to GeoDataFrame.\n")

    # Recreate the geometry column for column with updated easting and northing and convert back to gdf
    updated_geometry = [Point(xy) for xy in zip(station_node_mapping['easting'], station_node_mapping['northing'])]
    station_node_mapping = gpd.GeoDataFrame(station_node_mapping, geometry=updated_geometry, crs=crs_nat_grid)
    
    # Save snapped df
    logger.info(f"Saving snapped station list to {output_path}")
    station_node_mapping.to_csv(output_path, index=False)
    
    logger.info(f"All {catchment} catchment stations snapped to centroids.\n")

    return station_node_mapping