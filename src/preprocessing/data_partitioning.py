# Import Libraries
import os
import sys
import torch
import logging
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
def _get_node_ids_from_names(names_list, name_to_id_map, num, split_name):
    """
    Get corresponding node ID's from station list.
    """
    ids = []
    count = 0
    while count < num:
        for name in names_list:
            if name in name_to_id_map:
                ids.append(name_to_id_map[name])
                count +=1
            else:
                logging.warning(f"Warning: Station '{name}' specified for {split_name}"
                                f" split not found in observed stations. Ignoring station...")
    return ids

def _get_training_stations(station_name_to_id, test_station_shortlist, val_station_shortlist,
                           num_test, num_val):
    """
    Get list of stations to be associated with training model from remaining stations after
    validation and testing stations have been assigned.
    """
    train_station_names = []
    for station_name, _ in station_name_to_id.items():
        if station_name not in test_station_shortlist[0:num_test] and \
            station_name not in val_station_shortlist[0:num_val]:
                train_station_names.append(station_name)
                
    logger.info(f"Final list of station names for training model:\n    {train_station_names}\n")
    return train_station_names

def _get_split_count(main_df_full, perc_train, perc_val, perc_test):
    """
    Calculate total station count by type and define station and nodes lists from count.
    """
    # Verify main_df_full structure and dtype
    main_df_full['timestep'] = pd.to_datetime(main_df_full['timestep'])
    main_df_full = main_df_full.sort_values(by=['timestep', 'node_id']).reset_index(drop=True)

    # Get list of observed nodes (mesh nodes containing a gwl monitoring station)
    observed_nodes_df = main_df_full[main_df_full['station_name'].notna()][['station_name', 'node_id']].drop_duplicates()
    station_name_to_id = dict(zip(observed_nodes_df['station_name'], observed_nodes_df['node_id']))
    observed_nodes_list = sorted(observed_nodes_df['node_id'].tolist())

    # Check station split is as expected based on function call
    percentage_total = perc_train + perc_val + perc_test
    if percentage_total != 100:
        logger.warning(f"Warning: Requested Train/Test/Val station split does not add up to 100% of data.")

    # Calculate station number split
    num_train = round(len(observed_nodes_list) * (perc_train / 100))
    num_val = round(len(observed_nodes_list) * (perc_val / 100))
    num_test = round(len(observed_nodes_list) * (perc_test / 100))

    # Assert that calculated station split matches station list length
    assert (num_train + num_val + num_test) == len(observed_nodes_list), \
        "Total stations in station split does not match station count."
    
    return num_train, num_val, num_test, station_name_to_id, observed_nodes_list

def _check_split_success(train_station_ids, val_station_ids, test_station_ids, assigned_ids,
                         observed_nodes_list):
    """
    Verify split results add up to total station ID list length and asset that no station
    IDs are overlapping (causing leakage and bloating model performance).
    """
    # Confirm all stations are assigned
    if len(assigned_ids) != (len(train_station_ids) + len(val_station_ids) + len(test_station_ids)):
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

def define_station_id_splits(main_df_full: pd.DataFrame, catchment: str, test_station_shortlist: list, val_station_shortlist: list,
                          perc_train: int = 70, perc_val: int = 15, perc_test: int = 15):
    """
    Split station nodes into training, validation and testing sets by node ID.
    """
    logger.info(f"Defining station data split for {catchment} catchment...\n")
        
    # --- Get data partioning count by type and station id list ---
    
    (num_train, num_val, num_test, station_name_to_id,
     observed_nodes_list) = _get_split_count(main_df_full, perc_train, perc_val, perc_test)
        
    # --- Get training station name list ---
    
    train_station_names = _get_training_stations(station_name_to_id, test_station_shortlist,
                                                 val_station_shortlist, num_test, num_val)

    # --- Initialise lists of node_ids by type ---
    
    train_station_ids = _get_node_ids_from_names(train_station_names, station_name_to_id, num_train, "train")
    logger.info(f"Training Nodes:  NODE COUNT = {len(train_station_ids)}, NODE_IDs: {train_station_ids}\n")
    
    val_station_ids = _get_node_ids_from_names(val_station_shortlist[0:num_val], station_name_to_id, num_val, "validation")
    logger.info(f"Validation Nodes:  NODE COUNT = {len(val_station_ids)}, NODE_IDs: {val_station_ids}\n")
    
    test_station_ids = _get_node_ids_from_names(test_station_shortlist[0:num_test], station_name_to_id, num_test, "test")
    logger.info(f"Testing Nodes:  NODE COUNT = {len(test_station_ids)}, NODE_IDs: {test_station_ids}\n")
    
    assigned_ids = set(train_station_ids + val_station_ids + test_station_ids)
    
    # --- Confirm all stations are assigned and validate assignments ---
    
    _check_split_success(train_station_ids, val_station_ids, test_station_ids, assigned_ids, observed_nodes_list)

    return train_station_ids, val_station_ids, test_station_ids

# def apply_sentinel_mask(df, cols, sentinel=-999):