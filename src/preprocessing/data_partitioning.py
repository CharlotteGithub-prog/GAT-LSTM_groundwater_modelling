# Import Libraries
import os
import sys
import torch
import logging
import datetime
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from torch_geometric.data import Data
    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def load_graph_tensors(graph_output_dir, catchment):
    """
    Loads pre-generated edge_index and edge_attr tensors from catchment dir.
    """
    logging.info(f"Loading edge index and attribute tensors for {catchment} catchment...")
    try:
        edge_index_path = os.path.join(graph_output_dir, f"edge_index_tensor.pt")
        edge_attr_path = os.path.join(graph_output_dir, f"edge_attr_tensor.pt")

        edge_index_tensor = torch.load(edge_index_path)
        edge_attr_tensor = torch.load(edge_attr_path)
    
        logging.info(f"{catchment} catchment edge index and attribute tensors loaded successfully.\n")
        return edge_index_tensor, edge_attr_tensor
    
    except FileNotFoundError:
        logging.warning(f"Graph tensors not found at {graph_output_dir}. Ensure Section 5 has been run.")

# Process selected stations into train/val/tests splits
def _get_unique_node_ids_from_names(names_list, name_to_id_map, all_observed_names_set, split_name):
    """
    Helper function to convert a list of station names to their corresponding unique node_ids.
    Logs a warning if a specified station name is not found in observed stations.
    """
    # Initialise set (not list to ensure unique IDs)
    found_ids = set()
    for name in names_list:
        if name in all_observed_names_set:
            found_ids.add(name_to_id_map[name])
        else:
            logging.warning(f"Warning: Station '{name}' specified for {split_name}"
                            f" split not found in observed stations. Ignoring station...")
    
    # Should be sorted already but sort return for confirmation
    return sorted(list(found_ids))

def _get_split_count(main_df_full, perc_train, perc_val, perc_test):
    """
    Calculate total station count by type and define station and nodes lists from count.
    """
    # Get list of observed nodes (mesh nodes containing a gwl monitoring station)
    observed_nodes_df = main_df_full[main_df_full['station_name'].notna()][['station_name', 'node_id']].drop_duplicates()
    station_name_to_id = dict(zip(observed_nodes_df['station_name'], observed_nodes_df['node_id']))
    
    # Store all observed IDs as a set for efficient lookups and set operations
    all_observed_node_ids_set = set(observed_nodes_df['node_id'].tolist())
    all_observed_station_names_set = set(observed_nodes_df['station_name'].tolist())
    total_observed_stations_count = len(all_observed_node_ids_set)
    
    # Check station split is as expected based on function call
    percentage_total = perc_train + perc_val + perc_test
    if percentage_total != 100:
        logger.error(f"Error: Requested Train/Test/Val station percentages ({perc_train}+{perc_val}+{perc_test}"
                     f"={percentage_total}%) do not sum to 100%.")
        raise ValueError("Percentage split must sum to 100%.")

    # Calculate station number split
    num_val_target = round(total_observed_stations_count * (perc_val / 100))
    num_test_target = round(total_observed_stations_count * (perc_test / 100))
    num_train_target = total_observed_stations_count - num_val_target - num_test_target

    # Assert that calculated station split matches station list length
    assert (num_train_target + num_val_target + num_test_target) == total_observed_stations_count, \
        "Internal calculation error: Total stations in station count does not match observed count."
    
    return (num_train_target, num_val_target, num_test_target, station_name_to_id,all_observed_node_ids_set,
            all_observed_station_names_set)

def _check_split_success(train_station_ids, val_station_ids, test_station_ids, all_observed_node_ids_set):
    """
    Verify split results add up to total station ID list length and asset that no station
    IDs are overlapping (causing leakage and bloating model performance).
    """
    train_ids_set = set(train_station_ids)
    val_ids_set = set(val_station_ids)
    test_ids_set = set(test_station_ids)

    # Final validation of disjoint sets (these asserts are great)
    if len(train_ids_set.intersection(val_ids_set)) > 0:
        logger.error(f"ERROR: Train and Val station sets overlap! Overlapping IDs: {sorted(list(train_ids_set.intersection(val_ids_set)))}")
        raise ValueError("Overlapping station assignments detected.")
    if len(train_ids_set.intersection(test_ids_set)) > 0:
        logger.error(f"ERROR: Train and Test station sets overlap! Overlapping IDs: {sorted(list(train_ids_set.intersection(test_ids_set)))}")
        raise ValueError("Overlapping station assignments detected.")
    if len(val_ids_set.intersection(test_ids_set)) > 0:
        logger.error(f"ERROR: Val and Test station sets overlap! Overlapping IDs: {sorted(list(val_ids_set.intersection(test_ids_set)))}")
        raise ValueError("Overlapping station assignments detected.")

    # Confirm total assigned stations matches total observed stations
    combined_ids_set = train_ids_set.union(val_ids_set).union(test_ids_set)
    if len(combined_ids_set) != len(all_observed_node_ids_set):
        missing_ids = all_observed_node_ids_set - combined_ids_set
        extra_ids_in_splits = combined_ids_set - all_observed_node_ids_set

        if missing_ids:
            logger.error(f"ERROR: Not all observed stations ({len(all_observed_node_ids_set)}) were assigned. Missing IDs: {sorted(list(missing_ids))}")
        if extra_ids_in_splits:
            logger.error(f"ERROR: Some assigned IDs are not in the observed station list. Extra IDs: {sorted(list(extra_ids_in_splits))}")

        raise ValueError("Total assigned stations does not match total observed stations!")

def _random_fill_remaining_reqs(num_target, station_ids, unassigned_ids_pool, split_name):
    """
    Randomly fill remaining station count targets, reproducible due to random seed initialised in
    execution function.
    """
    num_to_add = num_target - len(station_ids)
    added_count = 0
    
    while added_count < num_to_add and unassigned_ids_pool:
        node_id_to_add = unassigned_ids_pool.pop(0) # Take from the shuffled pool
        station_ids.add(node_id_to_add)
        added_count += 1
        
    if added_count > 0:
        logger.info(f"Randomly added {added_count} stations to '{split_name}' set to meet target.")
        
    return station_ids

def _save_ID_count_to_config(train_station_ids, val_station_ids, test_station_ids, catchment):
    """
    Save number of stations assigned to each id list by type to config for future reference.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    config_path = "config/project_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f)

        # Defensive: Ensure expected structure in catchment config
        if 'model' not in config[catchment]:
            config[catchment]['model'] = {}
        if 'data_partioning' not in config[catchment]['model']:
            config[catchment]['model']['data_partioning'] = {}

        # Save list lengths to yaml
        config[catchment]['model']['data_partioning']["len_train"] = int(len(train_station_ids))
        config[catchment]['model']['data_partioning']["len_val"] = int(len(val_station_ids))
        config[catchment]['model']['data_partioning']["len_test"] = int(len(test_station_ids))

        with open(config_path, 'W-MON') as f:
            yaml.dump(config, f)
        
        logger.info(f"Saved len_train={len(train_station_ids)}, len_val={len(val_station_ids)},"
                    f" len_test={len(test_station_ids)} to {config_path}\n")

    except Exception as e:
        logger.error(f"Failed to save node ID list lengths to config.yaml: {e}")
         
def define_station_id_splits(main_df_full: pd.DataFrame, catchment: str, test_station_shortlist: list, val_station_shortlist: list,
                             random_seed: int, perc_train: int = 70, perc_val: int = 15, perc_test: int = 15):
    """
    Split station nodes into training, validation and testing sets by node ID. Prioritises stations from ordered
    shortlists (see catchment model config), then fills remaining slots randomly.
    """
    np.random.seed(random_seed)
    logger.info(f"Defining station data split for {catchment} catchment...\n")
        
    # Verify main_df_full structure and dtype
    main_df_full['timestep'] = pd.to_datetime(main_df_full['timestep'])
    main_df_full = main_df_full.sort_values(by=['timestep', 'node_id']).reset_index(drop=True)

    # --- Get data partioning count by type and station id list ---
    
    (num_train_target, num_val_target, num_test_target, station_name_to_id,
     all_observed_node_ids_set, all_observed_station_names_set) = _get_split_count(main_df_full, perc_train,
                                                                                   perc_val, perc_test)
    
    logger.info(f"Target split counts: Train={num_train_target}, Val={num_val_target}, Test={num_test_target} "
                f"(Total Observed: {len(all_observed_node_ids_set)}).")
        
    # --- Get training station name list ---
    
    # Apply slicing before passing to helper
    val_shortlist_sliced = val_station_shortlist[0:num_val_target]
    test_shortlist_sliced = test_station_shortlist[0:num_test_target]
    
    # Get unique IDs from the sliced shortlists
    val_station_ids = set(_get_unique_node_ids_from_names(
        val_shortlist_sliced, station_name_to_id, all_observed_station_names_set, "validation (shortlist)"))
    test_station_ids = set(_get_unique_node_ids_from_names(
        test_shortlist_sliced, station_name_to_id, all_observed_station_names_set, "test (shortlist)"))
    
    # Check for overlaps between initial val and test sets
    overlapping_initial_ids = val_station_ids.intersection(test_station_ids)
    if len(overlapping_initial_ids) > 0:
        overlapping_names = [name for name, node_id in station_name_to_id.items() if node_id in overlapping_initial_ids]
        logger.error(f"ERROR: Overlapping station IDs detected in validation and test shortlists: "
                     f"{sorted(list(overlapping_names))}. Stations must be unique across shortlists.")
        raise ValueError("Overlapping station assignments detected in shortlists.")
    
    # Log initial assignment
    logger.info(f"Initial assignment from sliced shortlists: Val={len(val_station_ids)} (Target={num_val_target}), "
                f"Test={len(test_station_ids)} (Target={num_test_target}).\n")
    
    # --- Randomly fill any unfulfilled requirements from remaining stations (if required) ---
    
    if len(val_station_ids) != num_val_target or len(test_station_ids) != num_test_target:
       
        # Prepare Pool of Unassigned Stations for Random Filling
        assigned_from_shortlists = val_station_ids.union(test_station_ids)
        unassigned_ids_pool = list(all_observed_node_ids_set - assigned_from_shortlists)
        np.random.shuffle(unassigned_ids_pool)
        
        # Check validation set
        if len(val_station_ids) < num_val_target:
            val_station_ids = _random_fill_remaining_reqs(num_val_target, val_station_ids, unassigned_ids_pool, "validation")
        elif len(val_station_ids) > num_val_target:
            logger.warning(f"Warning: Validation set from shortlist ({len(val_station_ids)}) exceeds target "
                           f"({num_val_target}). All short-listed stations kept -> may affect final split distribution.")

        # Check test set
        if len(test_station_ids) < num_test_target:
            test_station_ids = _random_fill_remaining_reqs(num_test_target, test_station_ids, unassigned_ids_pool, "testing")
        elif len(test_station_ids) > num_test_target:
            logger.warning(f"Warning: Test set from shortlist ({len(test_station_ids)}) exceeds target "
                           f"({num_test_target}). All short-listed stations kept -> may affect final split distribution.")
    
    # --- Assign Remaining Unassigned Stations to Training Set ---
    
    train_station_ids = all_observed_node_ids_set - val_station_ids - test_station_ids
    
    # Convert sets back to sorted lists for consistent output
    train_station_ids_output = sorted(list(train_station_ids))
    val_station_ids_output = sorted(list(val_station_ids))
    test_station_ids_output = sorted(list(test_station_ids))

    logger.info(f"Training Nodes:  NODE COUNT = {len(train_station_ids_output)}, NODE_IDs: {train_station_ids_output}\n")
    logger.info(f"Validation Nodes:  NODE COUNT = {len(val_station_ids_output)}, NODE_IDs: {val_station_ids_output}\n")
    logger.info(f"Testing Nodes:  NODE COUNT = {len(test_station_ids_output)}, NODE_IDs: {test_station_ids_output}\n")
    
    # --- Confirm all stations are assigned and validate assignments ---
    
    _check_split_success(train_station_ids, val_station_ids, test_station_ids, all_observed_node_ids_set)
    
    # --- Save station ID count by type to config ---
    
    _save_ID_count_to_config(train_station_ids, val_station_ids, test_station_ids, catchment)

    return train_station_ids_output, val_station_ids_output, test_station_ids_output

# Divide preprocessing datafame into PyG data objects using train/val/test station id subsets
def _build_x_tensor(df_snapshot, gwl_x_features, sentinel_value, val_station_ids, test_station_ids):
    """
    For validation and test stations, set their GWL-specific features in x to sentinels/zeros.
    All non-station nodes are already 'masked' to sentinel value (num) / 0.0 (cat)
    """
    # Identify nodes whose gwl feaetures should be masked in training obj
    training_masks_nodes = set(val_station_ids + test_station_ids)
    
    # Identify all relevant x features (no indexes or y feat)
    all_x_features = [col for col in df_snapshot.columns
                    if col not in ['node_id', 'timestep', 'gwl_value']]

    # prepare the 'x' feat matrix
    temp_df = df_snapshot.copy()
    
    # Add a single 'gwl_data_is_observed' indicator column, initialised to observed for all nodes
    temp_df['gwl_data_is_observed'] = 1.0
    
    # Apply GWL feature masking for identified nodes within the `x` features
    for id_to_mask in training_masks_nodes:
        node_mask = (temp_df["node_id"] == id_to_mask)
        
        # Flag to determine if this node's GWL data is considered 'observed' at this ts for input feats
        is_gwl_observed_at_this_node_timestep = False
        
        # Ensure the node exists in the snapshot
        if node_mask.any():
            gwl_val_at_node = df_snapshot.loc[node_mask, 'gwl_value'].iloc[0]
            # gwl_val_at_node is NaN if it's a non-station node or a test station
            if not pd.isna(gwl_val_at_node):
                is_gwl_observed_at_this_node_timestep = True
                
        # If node id mask exists (defensive, should exist), then mask x feats
        if not is_gwl_observed_at_this_node_timestep:
            for feat in gwl_x_features:
                if feat in temp_df.columns:
                    
                    # Set numerical features to sentinel filler value
                    if temp_df[feat].dtype in ['float64', 'float32', 'int64', 'int32']:
                        temp_df.loc[node_mask, feat] = sentinel_value
                    
                    # Set OHE categorical features to 0.0 (Non-Occurrence)
                    elif temp_df[feat].dtype == 'uint8':
                        temp_df.loc[node_mask, feat] = 0.0
    
    # Ensure the indicator feature is included in the final x_df
    all_x_features.append('gwl_data_is_observed')
    
    # prepare the 'x' feat matrix
    x_df = temp_df[all_x_features].copy()
                        
    # Build x tensor (dtype requiring numerical input)
    x = torch.tensor(x_df.values, dtype=torch.float)
    
    return x

def _build_y_tensor(df_snapshot):
    """
    All non-station nodes will currently be nan. This is as expected, as will be masked out of
    the loss calculation, howerver, the model requires a numerical input so 0.0 will be used to fill.
    -> These 0.0 value nodes represent ungauged stations, including val and test stations here.
    """
    
    # Replace nan's with 0.0 and build y tensor, .view() reshapes to infer dim '-1' (size = all other feats)
    y_values = df_snapshot["gwl_value"].values
    y = torch.tensor(np.nan_to_num(y_values, nan=0.0), dtype=torch.float).view(-1, 1)
    
    return y

def _build_object_masks(grouped_by_timestep, train_station_ids: list, val_station_ids: list,
                        test_station_ids: list, all_timesteps_list: list, gwl_x_features: list,
                        sentinel_value: float, mesh_node_count: int, all_node_ids: list,
                        edge_index_tensor: torch.Tensor, edge_attr_tensor: torch.Tensor):
    """
    tbd
    """
    # Initialise loop counter
    count = 1
    total_timesteps = len(grouped_by_timestep)
    
    # Loop through timesteps
    for timestep_t, df_t in grouped_by_timestep:
        logging.info(f"Processing timestep {count} of {total_timesteps}...")
        
        # Create copy to avoid modifying main df in place affecting future loops
        df_snapshot = df_t.copy()
        
        # Apply Feature Masking within `x` for val and test stations
        x = _build_x_tensor(df_snapshot, gwl_x_features, sentinel_value, val_station_ids, test_station_ids)
        
        # Prepare the target `y` and its masks
        y = _build_y_tensor(df_snapshot)
        
        # --- Build masks by type ---
        
        # Initialise all masks to node_id count length and to False for the entire mesh
        train_mask = torch.zeros(mesh_node_count, dtype=torch.bool)
        val_mask = torch.zeros(mesh_node_count, dtype=torch.bool)
        test_mask = torch.zeros(mesh_node_count, dtype=torch.bool)
        
        # Map current snapshot node_ids to their global index for mask creation
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
        
        # Initialise trackers for logging
        train_count = 0
        val_count = 0
        test_count = 0
        
        # Loop thorugh each node ID in the temporal data snapshot
        for idx, node_id_in_snapshot in enumerate(df_snapshot["node_id"]):
            global_idx = node_id_to_idx[node_id_in_snapshot]
            
            # Check if there's observed GWL data for this node, timestep (crucial to calc loss)
            gwl_val = df_snapshot.loc[df_snapshot['node_id'] == node_id_in_snapshot, 'gwl_value'].iloc[0]
            is_observed_gwl = not pd.isna(gwl_val)
            
            # --- Adjust all initialised False to True where meeting conditions for each mask ---
            
            # Defined training mask
            if node_id_in_snapshot in train_station_ids and is_observed_gwl:
                train_mask[global_idx] = True
                train_count += 1
            
            # Define validation mask
            elif node_id_in_snapshot in val_station_ids and is_observed_gwl:
                val_mask[global_idx] = True
                val_count += 1
            
            # Define testing mask
            elif node_id_in_snapshot in test_station_ids:  # Test cannot be observed
                test_mask[global_idx] = True
                test_count += 1
        
        # Final single log statement per timestep
        logger.info(f"   - Masks applied for timestep {timestep_t.date()}: "
            f"Train: {train_count} nodes, Val: {val_count} nodes, Test: {test_count} nodes.\n")
        
        # Map per-row node_id for this timestep
        snapshot_node_ids = df_snapshot["node_id"].tolist()
        node_id_tensor = torch.tensor([node_id_to_idx[n] for n in snapshot_node_ids], dtype=torch.long)

        # Create the PyG Data object for this timestep
        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            timestep=timestep_t # Store directly in data obj for easier temporal handling
        )
        
        # Store global node indices in the Data object
        data.node_id = node_id_tensor

        # Append data to full list and iterate loop counter
        all_timesteps_list.append(data)
        count += 1
        
    logger.info(f"PyG Data objects created for all timesteps: {len(all_timesteps_list)} total.")
    
    return all_timesteps_list

def build_pyg_object(processed_df: pd.DataFrame, sentinel_value: float, train_station_ids: list,
                     val_station_ids: list, test_station_ids: list, gwl_feats: list,
                     edge_index_tensor: torch.Tensor, edge_attr_tensor: torch.Tensor, catchment: str):
    """
    tbd
    """
    logger.info(f"Starting PyG Data object creation for {catchment} catchment.\n")

    # Ensure 'timestep' is a datetime object for comparison
    if processed_df['timestep'].dtype != 'datetime64[ns]':
        logger.info(f"Comverting 'timestep' dtype to datetime64[ns]")
        processed_df['timestep'] = pd.to_datetime(processed_df['timestep'])

    # Define single list containing Data objects for all timesteps (each Data object has the spatial train/val/test masks embedded)
    all_timesteps_list = []

    # Get a list and length of unique node_ids
    all_node_ids = sorted(processed_df["node_id"].unique().tolist())
    mesh_node_count = len(all_node_ids)
    
    # Identify groundwater features that are not target variable (for masking)
    gwl_x_features_initial = [feat for feat in gwl_feats if feat != 'gwl_value']
    static_gwl_feats = ['gwl_mean', 'gwl_dip']
    
    # Filter out static_gwl_feats from gwl_x_features_initial
    gwl_x_features_initial = [feat for feat in gwl_x_features_initial if feat not in static_gwl_feats]

    
    # Get all gwl x (not target) features using pre OHE col list
    gwl_x_features = []
    
    for col in processed_df.columns:
        for gwl_suff in gwl_x_features_initial:
            if col.startswith(gwl_suff):
                gwl_x_features.append(col)
    
    logger.info(f"Gwl x features list:\n{gwl_x_features}")
    
    # Group by timestep to create data group for each snapshot in model
    grouped_by_timestep = processed_df.groupby('timestep')
    
    # --- Create Object Masks ---
    
    all_timesteps_list = _build_object_masks(grouped_by_timestep, train_station_ids,
                        val_station_ids, test_station_ids, all_timesteps_list, gwl_x_features,
                        sentinel_value, mesh_node_count, all_node_ids, edge_index_tensor,
                        edge_attr_tensor)
    
    return all_timesteps_list
