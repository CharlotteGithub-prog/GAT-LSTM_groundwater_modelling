# Import Libraries
import os
import sys
import torch
import logging
import datetime
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

# Create data loaders for each timestep (7a)
def _check_data_loader(full_dataset_loader, all_timesteps_list):

    # Assert the number of batches matches the number of timesteps
    assert len(full_dataset_loader) == len(all_timesteps_list), "DataLoader batch count mismatch!"

    # Iterate through the first three batches and check shape and consistency
    num_batches_to_check = min(3, len(full_dataset_loader))

    for i, data_batch in enumerate(full_dataset_loader):
        if i >= num_batches_to_check:
            break
        
        logger.info(f"--- Batch {i+1} from DataLoader: ---")
        logger.info(f"    x shape: {data_batch.x.shape}")
        logger.info(f"    y shape: {data_batch.y.shape}")
        logger.info(f"    edge_index shape: {data_batch.edge_index.shape}")
        logger.info(f"    edge_attr shape: {data_batch.edge_attr.shape}\n")
        
        # Verify edge_index and edge_attr are consistent across batches (static graph as expected)
        if i == 0:
            first_edge_index = data_batch.edge_index
            first_edge_attr = data_batch.edge_attr
            
        else:
            assert torch.equal(data_batch.edge_index, first_edge_index), "edge_index changed across batches!"
            assert torch.equal(data_batch.edge_attr, first_edge_attr), "edge_attr changed across batches!"

    logger.info("DataLoader functionality check complete. All assertions passed.")

def build_data_loader(all_timesteps_list, batch_size, shuffle, catchment):
    
    logger.info(f"Building DataLoader for {catchment} catchment...\n")
    # Build dataset loader using global config vals
    full_dataset_loader = DataLoader(all_timesteps_list, batch_size=batch_size, shuffle=shuffle)
    logger.info(f"Created DataLoader with {len(full_dataset_loader)} batches (timesteps).")
    
    logger.info(f"Checking DataLoader has been built as expected...\n")
    _check_data_loader(full_dataset_loader, all_timesteps_list)
    
    return full_dataset_loader
