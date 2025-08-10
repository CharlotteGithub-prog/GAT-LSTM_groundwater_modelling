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

# Reorder columns for more logical scanning
def reorder_static_columns(df: pd.DataFrame):
    """ 
    Reorder columns in df to put geometry before features
    """
    desired_order = ['node_id', 'geometry', 'polygon_geometry', 'easting', 'northing',
                    'lon', 'lat', 'land_cover_code', 'mean_elevation',
                    'mean_slope_degrees', 'mean_aspect_sin', 'mean_aspect_cos',
                    'geo_bedrock_type', 'geo_superficial_type', 'bedrock_flow_type',
                    'bedrock_perm_avg', 'superficial_flow_type', 'superficial_perm_avg',
                    'HOST_soil_class', 'aquifer_productivity', 'distance_to_river']

    # Keep only the columns from desired_order that are in df
    available = [col for col in desired_order if col in df.columns]

    # Add remaining columns not in desired_order
    remaining = [col for col in df.columns if col not in desired_order]

    return df[available + remaining]

# Snapping stations to centroid nodes
def _obtain_snapping_data(polygon_geometry_path: str, catchment: str, station_list_path: str):
    """
    Loads spatial polygon geometry data and station coordinates, converting them to GeoDataFrames.
    
    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            - station_list_gdf (gpd.GeoDataFrame): GeoDataFrame of stations with point geometry.
            - polygon_geometry_mesh (gpd.GeoDataFrame): GeoDataFrame of mesh cell polygons.
    """
    # Reading in polygon geometry data
    full_path = polygon_geometry_path + f"{catchment}_mesh_cells_polygons.geojson"
    polygon_geometry_mesh = gpd.read_file(full_path)
    
    # Reading in station list with coords to snap stations to mesh 
    station_list_with_coords = pd.read_csv(station_list_path)
    
    # Use .apply() to create Point objects for each row and convert df -> gdf
    geometry = [Point(xy) for xy in zip(station_list_with_coords['easting'], station_list_with_coords['northing'])]
    crs_nat_grid = "EPSG:27700"
    station_list_gdf = gpd.GeoDataFrame(station_list_with_coords, geometry=geometry, crs=crs_nat_grid)
    
    return station_list_gdf, polygon_geometry_mesh

def _perform_attribute_merges(polygon_geometry_mesh: gpd.GeoDataFrame, station_node_mapping_temp: pd.DataFrame,
                              mesh_nodes_gdf: gpd.GeoDataFrame):
    """
    Merges polygon geometry and centroid coordinates into the station-node mapping DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with original polygon geometry and centroid coordinates merged in.
    """
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

    # --- Perform regular merge to merge in centroids ---
    
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
    
    return station_node_mapping

def _finalise_mapping_gdf(station_node_mapping: pd.DataFrame, crs: str = "EPSG:27700"):
    """
    Finalises the station-node mapping DataFrame, selecting relevant columns and recreating geometry.
    
    Returns:
        gpd.GeoDataFrame: Final GeoDataFrame with stations mapped to mesh centroids.
    """
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
    station_node_mapping = gpd.GeoDataFrame(station_node_mapping, geometry=updated_geometry, crs=crs)
    
    return station_node_mapping

def snap_stations_to_mesh(station_list_path: str, polygon_geometry_path: str, output_path: str,
                          mesh_nodes_gdf: gpd.GeoDataFrame, catchment: str):
    """
    Snaps groundwater level stations to the nearest mesh centroid nodes within a catchment.
    Orchestrates the process of loading station and mesh data, performing spatial and attribute
    merges, and finalising the mapping.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame mapping each station to its snapped mesh node_id,
                          with updated centroid coordinates.
    """
    logger.info(f"Snapping {catchment} catchment stations to mesh centroids...\n")
    
    # Read in data required for merges
    station_list_gdf, polygon_geometry_mesh = _obtain_snapping_data(
        polygon_geometry_path,
        catchment,
        station_list_path
    )

    # Perform spatial merge for gdf
    logger.info(f"Spatial merging catchment stations within polygon geometry.")

    station_node_mapping_temp = gpd.sjoin(
        station_list_gdf,
        polygon_geometry_mesh[['geometry', 'node_id']],
        how='left',
        op='within'
    )

    # Perform attribute merges
    station_node_mapping = _perform_attribute_merges(
        polygon_geometry_mesh,
        station_node_mapping_temp,
        mesh_nodes_gdf
    )

    # Converting snapped station DataFrame to mapping GeoDataFrame
    station_node_mapping = _finalise_mapping_gdf(station_node_mapping)
    
    # Save snapped df to 02_processed dir
    logger.info(f"Saving snapped station list to {output_path}")
    station_node_mapping.to_csv(output_path, index=False)
    
    logger.info(f"All {catchment} catchment stations snapped to centroids.\n")
    
    return station_node_mapping

# Merging timeseries data into full timeseries df
def merge_timeseries_data_to_df(model_start_date: str, model_end_date: str, feature_csv: str, csv_name: str,
                                feature: str, pred_frequency: str = 'daily', timeseries_df: pd.DataFrame = None):
    # Log feature being processed
    logging.info(f"Merging {feature} data into main timeseries dataframe...")
    
    # Get model frequency
    frequency_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}
    clean_pred_frequency = pred_frequency.lower().strip()
    
    frequency = frequency_map.get(clean_pred_frequency)
    if frequency is None:
        raise ValueError(f"Invalid prediction frequency: {pred_frequency}. Must be 'daily', "
                         f"'weekly', or 'monthly'.")

    # Initialise timeseries df on first run
    if timeseries_df is None:
        date_range = pd.date_range(start=model_start_date, end=model_end_date, freq=frequency)
        timeseries_df = pd.DataFrame(index=date_range)
        timeseries_df.index.name = 'time'

    # Read in timeseries feature data (parse date objs to convert to datetime as they are read in)
    file_path = os.path.join(feature_csv, csv_name)
    feature_df = pd.read_csv(file_path, parse_dates=['time'])

    # Normalise the 'time' column to remove raw time components (which would cause mismatch with main df)
    feature_df['time'] = feature_df['time'].dt.normalize() 

    # Merge timeseries feature data into timeseries df by timestep
    merged_ts_df = timeseries_df.merge(
        feature_df,
        left_index=True,  # Use the index of timeseries_df
        right_on='time',  # Match it to the 'time' column of precipitation_df
        how='left'
    )
    
    # Ensure the 'time' column becomes the index after merge and is named correctly
    if 'time' in merged_ts_df.columns:
        merged_ts_df = merged_ts_df.set_index('time') 
    merged_ts_df.index.name = 'time'
    
    return merged_ts_df

# Merge GWL data into full time series df
def _move_masked_cols_to_end(gwl_data):
    
    # Drop unneeded columns and rename as needed for merging clarity
    gwl_data = gwl_data.rename(columns={
        "data_type": "gwl_data_type",
        "quality": "gwl_data_quality",
        "masked": "gwl_masked",
        "value": "gwl_value"
    })
    
    # Define remaining column order
    cols_order = ['timestep', 'season_sin',  'season_cos', 'gwl_mean', 'gwl_dip', 'station_name',
                  'gwl_value', 'gwl_data_quality', 'gwl_data_type', 'gwl_masked', 'gwl_lag1',
                  'gwl_lag2', 'gwl_lag3', 'gwl_lag4', 'gwl_lag5', 'gwl_lag6', 'gwl_lag7']

    # Reindex the DataFrame with the new column order
    gwl_data = gwl_data[cols_order]
    
    return gwl_data

def _assign_station_node_ids(gwl_data, node_mapping_dir):
    logger.info(f"Mapping node ID's to station data.")
    node_mapping = pd.read_csv(node_mapping_dir)
    
    # Ensure station name formatting map in both dataframe
    node_mapping['station_name'] = node_mapping['station_name'].astype(str).str.lower().str.replace(" ", "_")
    gwl_data['station_name'] = gwl_data['station_name'].astype(str).str.lower().str.replace(" ", "_")
    node_mapping = node_mapping[node_mapping['station_name'] != 'cliburn_town_bridge_1']
    
    # Ensure all cols that will require gwl are together at end of main_df
    gwl_data = _move_masked_cols_to_end(gwl_data)
    
    # Merge mappings into gwl df
    mapped_gwl_data = gwl_data.merge(
        node_mapping,
        on='station_name',
        how='left'
    )
    
    return mapped_gwl_data

def load_gwl_data_for_merge(station_dir, node_mapping_dir):
    station_data = []
    station_files = os.listdir(station_dir)

    # Loop through files and append to main list
    logger.info(f"Reading in final trimmed gwl dataframe by station...")
    for filename in station_files:
        if filename.endswith("trimmed.csv"):
            logger.info(f"    Processing {filename}")
            file_path = os.path.join(station_dir, filename)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True, encoding='latin1').reset_index()
            df = df.rename(columns={df.index.name or 'index': 'timestep'})
            station_data.append(df)

    # Combine all station data wiht concat and confirm timestep dtype
    logger.info(f"Combining station list into final df for merge.")
    gwl_data = pd.concat(station_data, ignore_index=True)
    gwl_data['timestep'] = pd.to_datetime(gwl_data['timestep'])
    
    # Map station data to node_id's to facilitate main df merge
    gwl_data = _assign_station_node_ids(gwl_data, node_mapping_dir)
    gwl_data = gwl_data.drop(columns=['station_id', 'easting', 'northing', 'geometry'])
    
    return gwl_data
