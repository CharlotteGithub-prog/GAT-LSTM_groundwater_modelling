import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point

from src.data_ingestion.spatial_transformations import find_catchment_boundary

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def build_mesh(shape_filepath: str, output_path: str, catchment: str, grid_resolution: int = 1000):
    """
    Builds a spatial mesh of nodes (centroids of grid cells) within the input catchment boundary
    and saves the generated mesh nodes to output paths specified in project config.

    Args:
        shape_filepath (str): Path to the catchment boundary shapefile.
        grid_resolution (int): Resolution of the grid (default 1 km resolution with EPSG:27700 in meters).
        output_paths (str): String containing output file path without extension

    Returns:
        tuple: (mesh_nodes_table_df, mesh_nodes_gdf, mesh_cells_gdf_polygons, catchment_polygon)
            - mesh_nodes_table_df (pd.DataFrame): Node ID and coordinates (non-spatial table).
            - mesh_nodes_gdf (gpd.GeoDataFrame): Node ID, coordinates, and geometry (Point).
            - mesh_cells_gdf_polygons (gpd.GeoDataFrame): Node ID, coordinates, and geometry (Polygon).
            - catchment_polygon (gpd.GeoDataFrame): The processed catchment boundary (GeoDataFrame).
    """
    # Derive full path
    temp_geojson_path = f"{catchment}_combined_boundary.geojson"
    path = shape_filepath + temp_geojson_path
    
    logging.info(f"BUILD_MESH: Starting mesh construction with input: {path} and resolution: {grid_resolution}m\n")
    
    ## ---- Import single geometry spatial data and find bounds ----
    
    catchment_polygon, catchment_geometry, minx, miny, maxx, maxy = find_catchment_boundary(
        catchment=catchment,
        shape_filepath=path,
        required_crs=27700  # Ensure the catchment boundary is in BNG
    )

    ## ---- Set up coordinate node mesh ----

    # Generate bounds of grid aligned to global crs informed 1km mesh (not aligned to geometry)
    minx_aligned = np.floor(minx / grid_resolution) * grid_resolution
    miny_aligned = np.floor(miny / grid_resolution) * grid_resolution
    maxx_aligned = np.ceil(maxx / grid_resolution) * grid_resolution
    maxy_aligned = np.ceil(maxy / grid_resolution) * grid_resolution

    x_coordinates_bottomleft = np.arange(minx_aligned, maxx_aligned + grid_resolution, grid_resolution)
    y_coordinates_bottomleft = np.arange(miny_aligned, maxy_aligned + grid_resolution, grid_resolution)

    logger.info(f"Aligned minx: {minx_aligned}, miny: {miny_aligned}, maxx: {maxx_aligned}, maxy: {maxy_aligned}")
    logger.info(f"Number of x-coordinates (bottom-left): {len(x_coordinates_bottomleft)}")
    logger.info(f"Number of y-coordinates (bottom-left): {len(y_coordinates_bottomleft)}\n")
    

    # Initialise grid cell list and set up regular grid of points within the bounding box
    all_mesh_nodes_points = []
    all_mesh_polygons = []
    
    for x_bl in x_coordinates_bottomleft:
        for y_bl in y_coordinates_bottomleft:
            # Calculate the centroid of each cell
            centroid_x = x_bl + (grid_resolution / 2)
            centroid_y = y_bl + (grid_resolution / 2)
            all_mesh_nodes_points.append(Point(centroid_x, centroid_y))
            
            # Create the polygon for this cell
            cell_polygon = box(x_bl, y_bl, x_bl + grid_resolution, y_bl + grid_resolution)
            all_mesh_polygons.append(cell_polygon)
    
    logger.info(f"Generated {len(all_mesh_nodes_points)} grid cells (centroids and polygons) within bounding "
                f"box (before filtering).")

    # Create GEOdf of all potential (bounding box) nodes
    all_mesh_nodes_gdf = gpd.GeoDataFrame(geometry=all_mesh_nodes_points, crs="EPSG:27700")
    all_mesh_polygons_gdf = gpd.GeoDataFrame(geometry=all_mesh_polygons, crs="EPSG:27700")
    
    # Create a single-row GEOdf from the geometry (boundary)
    catchment_gdf_for_sjoin = gpd.GeoDataFrame(geometry=[catchment_geometry], crs=catchment_polygon.crs)

    # Join dataframes and drop the right index column the sjoin adds
    mesh_nodes_gdf = gpd.sjoin(all_mesh_nodes_gdf, catchment_gdf_for_sjoin, how="inner", predicate='within')
    mesh_nodes_gdf = mesh_nodes_gdf.drop(columns=['index_right'])
    logger.info(f"Filtered down to catchment boundary containing {len(mesh_nodes_gdf)} nodes\n")

    ## ---- Convert to table ----
    
    # Assign unique node_ids (UNID) after filtering
    mesh_nodes_gdf['node_id'] = range(len(mesh_nodes_gdf))
    
    # Rename geometry column of polygons to avoid conflict during sjoin and use 'left_index' to match
    mesh_cells_gdf_polygons = gpd.sjoin(
        all_mesh_polygons_gdf.reset_index().rename(columns={'index': 'original_poly_index'}),
        mesh_nodes_gdf[['node_id', 'geometry']],
        how="inner",
        predicate='contains' # Ensure polygon always contains its centroid
    )
    
    mesh_cells_gdf_polygons = mesh_cells_gdf_polygons.drop(columns=['index_right']).set_index('original_poly_index')

    # Add original Easting/Northing coordinates (as in EPSG:27700)
    mesh_nodes_gdf['easting'] = mesh_nodes_gdf.geometry.x
    mesh_nodes_gdf['northing'] = mesh_nodes_gdf.geometry.y

    # Convert to WGS84 (EPSG:4326) to add lat/lon for visualisations
    mesh_nodes_4326 = mesh_nodes_gdf.to_crs(epsg=4326)
    mesh_nodes_gdf['lon'] = mesh_nodes_4326.geometry.x
    mesh_nodes_gdf['lat'] = mesh_nodes_4326.geometry.y

    # Select the columns needed for node table
    mesh_nodes_table = mesh_nodes_gdf[['node_id', 'easting', 'northing', 'lon', 'lat']]
    
    logger.info(f"First few mesh nodes (centroids with coordinates):\n\n{mesh_nodes_table.head().to_string()}\n")
    logger.info(f"Total number of mesh nodes (centroids) for the catchment: {len(mesh_nodes_table)}\n")
    
    # --- Saving the outputs ---
    
    csv_path = f"{output_path}_{grid_resolution}.csv"
    gpkg_path = f"{output_path}_{grid_resolution}.gpkg"
    shp_path = f"{output_path}_{grid_resolution}.shp"
    
    # Save the mesh nodes table and gdf to appropriate files
    mesh_nodes_table.to_csv(csv_path, index=False)
    logger.info(f"Saved mesh nodes CSV to: {csv_path}")
    
    mesh_nodes_gdf.to_file(gpkg_path, layer='mesh_nodes', driver='GPKG')  # GeoPackage
    logger.info(f"Saved mesh nodes GPKG to: {gpkg_path}")
    
    mesh_nodes_gdf.to_file(shp_path, driver='ESRI Shapefile')
    logger.info(f"Saved mesh nodes shp to: {shp_path}\n")
    
    return mesh_nodes_table, mesh_nodes_gdf, mesh_cells_gdf_polygons, catchment_polygon

def define_catchment_polygon(england_catchment_gdf_path: str, target_mncat: str, catchment: str,
                             polygon_output_path: str):
    
    # Read in and filter to desired catchment
    full_management_catchments_gdf = gpd.read_file(england_catchment_gdf_path)
    eden_esk_catchments = full_management_catchments_gdf[
        (full_management_catchments_gdf['mncat_name'] == target_mncat)
    ]
    
    logging.info(f"{target_mncat} boundary polygon(s) extracted from England data.")

    # Dissolve constituent parts into one polygon
    combined_eden_esk_polygon = eden_esk_catchments.dissolve()

    # Save as a geojson (checking crs first)
    if combined_eden_esk_polygon.crs is None or combined_eden_esk_polygon.crs != "EPSG:27700":
        combined_eden_esk_polygon = combined_eden_esk_polygon.to_crs(epsg=27700)
        
    temp_geojson_path = f"{catchment}_combined_boundary.geojson"
    path = polygon_output_path + temp_geojson_path
    combined_eden_esk_polygon.to_file(path, driver="GeoJSON")
    
    print(path)

    logging.info(f"Combined {target_mncat} boundary saved to: {path}")

def build_main_df(start_date: str, end_date: str, mesh_nodes_gdf: gpd.GeoDataFrame, catchment: str):
    """
    Build a dataframe for merging features into with pairs of timestep and node_id
    acting as multiindex pairs. Convert to df to return for easy merging.

    Returns:
        pd.DataFrame: A dataframe consisting of 'timestep' and 'node_id' for full
                        model period.
    """
    logger.info(f"Building main model input dataframe for {catchment} catchment...\n")

    # Get model date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    logger.info(f"Building timesteps from {start_date[0:10]} to {end_date[0:10]}")

    # Get unique node id's
    all_node_ids = mesh_nodes_gdf['node_id'].unique().tolist()

    logger.info(f"Building node ID's from 0 to {len(all_node_ids)-1}")

    # Create the Cartesian product of dates and node_ids (cross join for all combos)
    multi_index = pd.MultiIndex.from_product([date_range, all_node_ids],
                                            names=['timestep', 'node_id'])

    logger.info(f"Converting multi index to data frame for feature merging")

    # Convert multi index to df for merging
    main_df = multi_index.to_frame(index=False)

    print(f"\nTotal rows in main {catchment} catchment DataFrame: {len(main_df):.0e}\n")

    return main_df
