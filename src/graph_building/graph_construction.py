import os
import sys
import torch
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

# Build initial graph mesh
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

def build_main_df(start_date: str, end_date: str, mesh_nodes_gdf: gpd.GeoDataFrame, catchment: str,
                  pred_frequency: str = 'daily'):
    """
    Build a dataframe for merging features into with pairs of timestep and node_id
    acting as multiindex pairs. Convert to df to return for easy merging.

    Returns:
        pd.DataFrame: A dataframe consisting of 'timestep' and 'node_id' for full
                        model period.
    """
    logger.info(f"Building main model input dataframe for {catchment} catchment...\n")

    # Get model date range (assumes daily, weekly or monthly timestep)
    frequency_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}
    clean_pred_frequency = pred_frequency.lower().strip()
    
    frequency = frequency_map.get(clean_pred_frequency)
    if frequency is None:
        raise ValueError(f"Invalid prediction frequency: {pred_frequency}. Must be 'daily', "
                         f"'weekly', or 'monthly'.")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
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

# Define graph directional edge weightings
def _define_graph_adjacency_idx(mesh_cells_gdf_polygons: gpd.GeoDataFrame):
    # --- Define Graph Adjacency (Simple Grid Adjacency using spatial touches) ---
    
    logger.info("Building graph adjacency based on 'touches' spatial predicate...")
    mesh_cells_gdf_polygons.sindex  # Use explicit spatial index for efficiency.

    # Spatially join the df with itself to return new gdf with duplicated rows (one for each adjacency)
    adjacency_df = gpd.sjoin(
        mesh_cells_gdf_polygons,
        mesh_cells_gdf_polygons,
        how="inner",
        predicate="touches",
        lsuffix="source",
        rsuffix="destination"
    )

    # Filter out self-loops incase they were made during sjoin
    adjacency_df = adjacency_df[adjacency_df['node_id_source'] != adjacency_df['node_id_destination']]

    # Extract source and destination node IDs to create edge_index_df
    edge_index_df = pd.DataFrame({
        'source': adjacency_df['node_id_source'].values,
        'destination': adjacency_df['node_id_destination'].values
    })

    # Wrap value pairs in 2D NumPy array to reduce tensor computation reqs
    edge_index_np = np.array([
        edge_index_df['source'].values,
        edge_index_df['destination'].values
    ])

    # Convert to PyTorch tensor (directly as node_id's 0-indexed and sequential)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
    logger.info(f"{edge_index_tensor.shape[1]} edges generated for the graph.\n")
    
    return edge_index_df, edge_index_tensor

def _calculate_edge_attrs(source_data: pd.DataFrame, dest_data: pd.DataFrame, epsilon_scalar: str):
    """
    Calculate and return the attributes for this edge.
    """
    # Distance Attribute (pythagorean)
    dist_x = dest_data['easting'] - source_data['easting']
    dist_y = dest_data['northing'] - source_data['northing']
    distance = np.sqrt(dist_x**2 + dist_y**2)
    
    # Elevation Difference Attr (to indicate potential flow direction)
    elevation_difference = source_data['mean_elevation'] - dest_data['mean_elevation']
    
    # Source Node Slope Components (calculated during initial slope derivations)
    source_slope_dx = float(source_data['mean_slope_dx'])
    source_slope_dy = float(source_data['mean_slope_dy'])
    
    # Magnitude of source slope vector (adding epsilon to prevent div by zero err when flat)
    epsilon = float(epsilon_scalar)
    source_slope_magnitude = np.sqrt(source_slope_dx**2 + source_slope_dy**2) + epsilon
    
    # Directionality score: Cosine similarity between source's slope vector and vector from source to dest
    # Where: Cosine similarity = dot_product / (magnitude1 * magnitude2)
    dot_product = (source_slope_dx * dist_x) + (source_slope_dy * dist_y)
    directionality = dot_product / (source_slope_magnitude * distance)
    
    # Clip score to [-1, 1] range to ensure valid cosine similarity
    directional_score = np.clip(directionality, -1.0, 1.0)
    
    return distance, elevation_difference, source_slope_dx, source_slope_dy, directional_score
   
def _save_tensor_attrs(edge_index_tensor: torch, edge_attr_tensor: torch,
                       graph_output_dir: str, catchment: str):
    """
    Save tensors to access in subsequent notebooks
    """
    # Create save dir if it doesn't exist
    os.makedirs(graph_output_dir, exist_ok=True)

    edge_index_path = os.path.join(graph_output_dir, f"edge_index_tensor.pt")
    edge_attr_path = os.path.join(graph_output_dir, f"edge_attr_tensor.pt")

    torch.save(edge_index_tensor, edge_index_path)
    torch.save(edge_attr_tensor, edge_attr_path)

    logger.info(f"Graph components saved for {catchment} catchment:")
    logger.info(f"    - Edge Index: {edge_index_path}")
    logger.info(f"    - Edge Attributes: {edge_attr_path}\n")

def _build_station_radius_edges(node_static_features_df: pd.DataFrame, station_node_ids: np.ndarray, radius_m: float):
    """
    Create directed edges between all station nodes within radius_m. Returns a df with cols ['source',
    'destination'] (in both directions).
    """
    if station_node_ids is None or len(station_node_ids) == 0:
        return pd.DataFrame(columns=["source", "destination"])
    
    # Coords in m -> shape (S, 2)
    coords = node_static_features_df.loc[station_node_ids, ["easting", "northing"]].to_numpy(dtype=np.float64)
    
    # Pairwise Euclid. dist.
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2)) # Now (S,S)
    
    # Buidling i->j pairs with 0 < dist <= radius_m
    src_index, dest_index = np.where((D <= float(radius_m)) & (D > 0.0))  # D check fpr defensive but maybe throw error?
    src_nodes = station_node_ids[src_index]
    dest_nodes = station_node_ids[dest_index]
    
    return pd.DataFrame({"source": src_nodes, "destination": dest_nodes})

def define_graph_adjacency(directional_edge_weights: pd.DataFrame, elevation_geojson_path: str, graph_output_dir: str,
                           mesh_cells_gdf_polygons: gpd.GeoDataFrame, epsilon_path: str, station_node_ids: np.ndarray,
                           station_radius_m: float, catchment: str):
    
    # --- Load required data points and initialise static node feature df ---
    logging.info(f"Determining graph adjacency for {catchment} catchment...\n")
    
    # Create specific node_id column to merge and initialise combined edge feat df
    directional_edge_weights["node_id"] = range(0, len(directional_edge_weights))
    node_static_features_df = mesh_cells_gdf_polygons[['geometry', 'node_id']].copy()
    
    # Assign node_id to mean_elevation from elevation_gdf (assuming same order)
    elevation_gdf = gpd.read_file(elevation_geojson_path)
    if 'node_id' not in elevation_gdf.columns:
        elevation_gdf['node_id'] = mesh_cells_gdf_polygons['node_id']
        logger.warning("Added 'node_id' to elevation_gdf assuming positional alignment.")
    
    # Merge elevation feat into static features df using node_id col
    node_static_features_df = node_static_features_df.merge(
        elevation_gdf[['mean_elevation', 'node_id']],
        on='node_id',
        how='left'
    )

    # Join the slope components (dx, dy) and coordinates from directional_edge_weights_indexed
    node_static_features_df = node_static_features_df.merge(
        directional_edge_weights[['mean_slope_dx', 'mean_slope_dy', 'easting', 'northing', 'node_id']],
        on='node_id',
        how='left'
    )
    
    # Check no NaNs introduced from the join (meaning indexes correctly aligned)
    if node_static_features_df[['mean_slope_dx', 'mean_slope_dy', 'easting', 'northing']].isnull().any().any():
        logger.warning("NaNs found in joined static features. Check node_id alignment.")

    # Ensure index is node id
    node_static_features_df = node_static_features_df.set_index("node_id", drop=False).sort_index()
    
    # --- Get Adjacency's ---
    
    # Get base mesh adjacency (k edges)
    grid_edge_index_df, _ = _define_graph_adjacency_idx(mesh_cells_gdf_polygons)
    
    # Define station -> station edges within radius
    station_edges_df = _build_station_radius_edges(node_static_features_df, station_node_ids, station_radius_m)
    logger.info(f"Added {len(station_edges_df)} station <-> station edges within {station_radius_m/1000:.1f}km.")

    # combine k and station edges (deduplicating as needed)
    edge_index_df = pd.concat([grid_edge_index_df, station_edges_df], ignore_index=True)
    edge_index_df.drop_duplicates(subset=["source", "destination"], inplace=True)
    
    # Rebuild tensors so edge ordering and attrs are stable (for reproducibility)
    edge_index_df = edge_index_df.sort_values(["source", "destination"], kind="mergesort").reset_index(drop=True)

    # --- Define Edge Features (edge_attr) ---

    logger.info("Calculating edge features...")
    edge_features = []

    # Iterate through each edge to calculate its attributes
    for index, row in edge_index_df.iterrows():
        source_node_id = row['source']
        dest_node_id = row['destination']
        
        source_data = node_static_features_df.loc[source_node_id]
        dest_data = node_static_features_df.loc[dest_node_id]
    
        (distance, elevation_difference, source_slope_dx, source_slope_dy,
         directional_score) = _calculate_edge_attrs(source_data, dest_data, epsilon_path)
        
        # Group all features for this edge
        edge_features.append([
            distance,
            elevation_difference,
            source_slope_dx,
            source_slope_dy,
            directional_score
        ])
        
    # Convert features list to PyTorch tensor
    edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float)
    
    edge_index_np = np.vstack([edge_index_df["source"].to_numpy(dtype=np.int64),
                               edge_index_df["destination"].to_numpy(dtype=np.int64)])
    edge_index_tensor = torch.as_tensor(edge_index_np, dtype=torch.long)

    # Assert final attribut tensor is equal length to edge index tensor
    assert edge_attr_tensor.shape[0] == edge_index_tensor.shape[1], \
        f"Mismatch: {edge_attr_tensor.shape[0]} attrs vs {edge_index_tensor.shape[1]} edges"
    logger.info(f"Edge attributes tensor created with shape: {edge_attr_tensor.shape}\n")
    
    # Save tensors for direct access
    _save_tensor_attrs(edge_index_tensor, edge_attr_tensor, graph_output_dir, catchment)
    
    return edge_attr_tensor, edge_index_tensor
