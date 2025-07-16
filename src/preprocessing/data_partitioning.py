# Import Libraries
import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
    
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

def _check_split_success(train_station_ids, val_station_ids, test_station_ids, observed_nodes_list):
    """
    Verify split results add up to total station ID list length and asset that no station
    IDs are overlapping (causing leakage and bloating model performance).
    """
    # Confirm all stations are assigned
    if len(observed_nodes_list) != (len(train_station_ids) + len(val_station_ids) + len(test_station_ids)):
        logging.warning(f"Warning: Total number of assigned IDs does not match total catchment IDs.")
        raise ValueError("Overlapping station assignments detected.")
    
    # Final validation of disjoint sets
    assert len(set(train_station_ids).intersection(val_station_ids)) == 0, \
        "ERROR after assignment: Train and Val station sets overlap!"
    assert len(set(train_station_ids).intersection(test_station_ids)) == 0, \
        "ERROR after assignment: Train and Test station sets overlap!"
    assert len(set(val_station_ids).intersection(test_station_ids)) == 0, \
        "ERROR after assignment: Val and Test station sets overlap!"
    assert len(train_station_ids) + len(val_station_ids) + len(test_station_ids) == len(observed_nodes_list), \
        "Total assigned stations does not match total observed stations!"

def _random_fill_remaining_reqs(num_target, station_ids, unassigned_ids_pool):
    """
    Randomly fill remaining station count targets, reproducible due to random seed initialised in
    execution function.
    """
    num_to_add = num_target - len(station_ids)
    added_val_count = 0
    
    while added_val_count < num_to_add and unassigned_ids_pool:
        node_id_to_add = unassigned_ids_pool.pop(0) # Take from the shuffled pool
        station_ids.add(node_id_to_add)
        added_val_count += 1
        
    if added_val_count > 0:
        logger.info(f"Randomly added {added_val_count} stations to validation set to meet target.")
        
    return station_ids
            
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
                f"(Total Observed: {len(all_observed_node_ids_set)}).\n")
        
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
    logger.info(f"Initial assignment from sliced shortlists: "
                f"Val={len(val_station_ids)} (Target={num_val_target}), "
                f"Test={len(test_station_ids)} (Target={num_test_target}).")
    
    # --- Randomly fill any unfulfilled requirements from remaining stations (if required) ---
    
    if len(val_station_ids) != num_val_target or len(test_station_ids) != num_test_target:
       
        # Prepare Pool of Unassigned Stations for Random Filling
        assigned_from_shortlists = val_station_ids.union(test_station_ids)
        unassigned_ids_pool = list(all_observed_node_ids_set - assigned_from_shortlists)
        np.random.shuffle(unassigned_ids_pool)
        
        # Check validation set
        if len(val_station_ids) < num_val_target:
            val_station_ids = _random_fill_remaining_reqs(num_val_target, val_station_ids, unassigned_ids_pool)
        elif len(val_station_ids) > num_val_target:
            logger.warning(f"Warning: Validation set from shortlist ({len(val_station_ids)}) exceeds target "
                           f"({num_val_target}). All short-listed stations kept -> may affect final split distribution.")

        # Check test set
        if len(test_station_ids) < num_test_target:
            test_station_ids = _random_fill_remaining_reqs(num_test_target, test_station_ids, unassigned_ids_pool)
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
    _check_split_success(train_station_ids_output, val_station_ids_output, test_station_ids_output,
                         all_observed_node_ids_set)

    return train_station_ids_output, val_station_ids_output, test_station_ids_output

# def apply_sentinel_mask(df: pd.DataFrame, cols: list, sentinel: float = -999.0):