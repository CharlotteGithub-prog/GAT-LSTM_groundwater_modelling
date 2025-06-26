import sys
import logging
import numpy as np
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
        tuple: (mesh_nodes_table_df, mesh_nodes_gdf, catchment_polygon)
            - mesh_nodes_table_df (pd.DataFrame): Node ID and coordinates.
            - mesh_nodes_gdf (gpd.GeoDataFrame): Node ID, coordinates, and geometry (Point).
            - catchment_polygon (gpd.GeoDataFrame): The processed catchment boundary.
    """
    logging.info(f"BUILD_MESH: Starting mesh construction with input: {shape_filepath} and resolution: {grid_resolution}m\n")
    
    ## ---- Import single geometry spatial data and find bounds ----
    
    catchment_polygon, catchment_geometry, minx, miny, maxx, maxy = find_catchment_boundary(
        catchment=catchment,
        shape_filepath=shape_filepath,
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
    for x_bl in x_coordinates_bottomleft:
        for y_bl in y_coordinates_bottomleft:
            # Calculate the centroid of each cell
            centroid_x = x_bl + (grid_resolution / 2)
            centroid_y = y_bl + (grid_resolution / 2)
            all_mesh_nodes_points.append(Point(centroid_x, centroid_y))
    
    logger.info(f"Generated {len(all_mesh_nodes_points)} grid cells within bounding box (before filtering).")

    # Create GEOdf of all potential (bounding box) nodes
    all_mesh_nodes_gdf = gpd.GeoDataFrame(geometry=all_mesh_nodes_points, crs="EPSG:27700")
    
    # Create a single-row GEOdf from the geometry (boundary)
    catchment_gdf_for_sjoin = gpd.GeoDataFrame(geometry=[catchment_geometry], crs=catchment_polygon.crs)

    # Join dataframes and drop the right index column the sjoin adds
    mesh_nodes_gdf = gpd.sjoin(all_mesh_nodes_gdf, catchment_gdf_for_sjoin, how="inner", predicate='within')
    mesh_nodes_gdf = mesh_nodes_gdf.drop(columns=['index_right'])
    logger.info(f"Filtered down to catchment boundary containing {len(mesh_nodes_gdf)} nodes\n")

    ## ---- Convert to table ----
    
    # Assign unique node_ids (UNID) after filtering
    mesh_nodes_gdf['node_id'] = range(len(mesh_nodes_gdf))

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
    
    return mesh_nodes_table, mesh_nodes_gdf, catchment_polygon
