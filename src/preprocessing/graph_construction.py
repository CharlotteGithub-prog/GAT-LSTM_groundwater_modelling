import os
import ast
import folium
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pyproj import Transformer
from shapely.geometry import box
from datetime import datetime, timedelta

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

def build_mesh(shape_filepath: str, grid_resolution: int = 1000, output_paths: dict = None):
    """
    Builds a spatial mesh of nodes (centroids of grid cells) within the input catchment boundary
    and saves the generated mesh nodes to output paths specified in project config.

    Args:
        shape_filepath (str): Path to the catchment boundary shapefile.
        grid_resolution (int): Resolution of the grid (default 1 km resolution with EPSG:27700 in meters).
        output_paths (dict, optional): Dictionary containing output file paths.
                                       Expected keys: 'mesh_nodes_csv_output', 'mesh_nodes_gpkg_output', 'mesh_nodes_shp_output'.
                                       If None, files will not be saved.

    Returns:
        tuple: (mesh_nodes_table_df, mesh_nodes_gdf, catchment_polygon)
            - mesh_nodes_table_df (pd.DataFrame): Node ID and coordinates.
            - mesh_nodes_gdf (gpd.GeoDataFrame): Node ID, coordinates, and geometry (Point).
            - catchment_polygon (gpd.GeoDataFrame): The processed catchment boundary.
    """
    logger.info(f"BUILD_MESH: Starting mesh construction with input: {shape_filepath} and resolution: {grid_resolution}m\n")
    
    ## ---- Import single geometry spatial data ----
    
    # Load spatial boundary shape file
    logger.info(f"Loading catchment boundary from: {shape_filepath}")
    catchment_polygon = gpd.read_file(shape_filepath)
    catchment_polygon = catchment_polygon.to_crs(epsg=27700) # Convert to British National Grid

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

    ## ---- Set up coordinate node mesh ----

    # Generate bottom left coordinates of grid cells
    x_coordinates_bottomleft = np.arange(minx, maxx + grid_resolution, grid_resolution)
    y_coordinates_bottomleft = np.arange(miny, maxy + grid_resolution, grid_resolution)

    logger.info(f"Number of x-coordinates (bottom-left): {len(x_coordinates_bottomleft)}")
    logger.info(f"Number of y-coordinates (bottom-left): {len(y_coordinates_bottomleft)}\n")


    # Initialise grid cell list and set up regular grid of points within the bounding box
    grid_cells = []
    for x in x_coordinates_bottomleft:
        for y in y_coordinates_bottomleft:
            cell = box(x, y, x + grid_resolution, y + grid_resolution)
            grid_cells.append(cell)
    
    logger.info(f"Generated {len(grid_cells)} grid cells within bounding box (before filtering).")

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:27700")

    # Keep only grid cells that intersect the catchment
    grid_intersected = gpd.overlay(grid_gdf, gpd.GeoDataFrame(geometry=[catchment_geometry],
        crs="EPSG:27700"), how='intersection', keep_geom_type=True)
    mesh_nodes_gdf = grid_intersected.copy() # Using copy to avoid SettingWithCopyWarning
    logger.info(f"Filtered down to catchment boundary containing {len(grid_intersected)} nodes\n")

    ## ---- Convert to table ----

    # Calculate the centroid of each (potentially clipped) grid cell
    mesh_nodes_gdf['geometry'] = mesh_nodes_gdf.geometry.representative_point() # previously as .centroid

    # Add original Easting/Northing coordinates (as in EPSG:27700)
    mesh_nodes_gdf['easting'] = mesh_nodes_gdf.geometry.x
    mesh_nodes_gdf['northing'] = mesh_nodes_gdf.geometry.y
    mesh_nodes_gdf['node_id'] = range(len(mesh_nodes_gdf)) # UNID

    # Convert to WGS84 (EPSG:4326) to add lat/lon for visualisations
    mesh_nodes_4326 = mesh_nodes_gdf.to_crs(epsg=4326)
    mesh_nodes_gdf['lon'] = mesh_nodes_4326.geometry.x
    mesh_nodes_gdf['lat'] = mesh_nodes_4326.geometry.y

    # Select the columns needed for node table
    mesh_nodes_table = mesh_nodes_gdf[['node_id', 'easting', 'northing', 'lon', 'lat']]
    
    logger.info(f"First few mesh nodes (centroids with coordinates):\n\n{mesh_nodes_table.head().to_string()}\n")
    logger.info(f"Total number of mesh nodes (centroids) for the catchment: {len(mesh_nodes_table)}\n")
    
    # --- Saving the outputs ---
    
    if output_paths:
        csv_path = output_paths.get('mesh_nodes_csv_output')
        gpkg_path = output_paths.get('mesh_nodes_gpkg_output')
        shp_path = output_paths.get('mesh_nodes_shp_output')
    
        # Save the mesh nodes table and gdf to appropriate files
        if csv_path:
            mesh_nodes_table.to_csv(csv_path, index=False)
            logger.info(f"Saved mesh nodes CSV to: {csv_path}")
        if gpkg_path:
            mesh_nodes_gdf.to_file(gpkg_path, layer='mesh_nodes', driver='GPKG')  # GeoPackage
            logger.info(f"Saved mesh nodes GPKG to: {gpkg_path}")
        if shp_path:
            mesh_nodes_gdf.to_file(shp_path, driver='ESRI Shapefile')
            logger.info(f"Saved mesh nodes shp to: {shp_path}\n")
    
    return mesh_nodes_table, mesh_nodes_gdf, catchment_polygon