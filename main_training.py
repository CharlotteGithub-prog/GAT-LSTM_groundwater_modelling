# This file is part of the Dissertation project and is licensed under the MIT License.

### FULL PIPELINE ###
# Expected Processing Time for Eden Catchment (with API calls = False): ## hrs ## minutes
#        - 00 hrs 05 minutes 58 seconds -> to end of section 6b

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================

# --- 1a. Library Imports ---
import os
import sys
import torch
import random
import joblib
import logging
import numpy as np
import pandas as pd

# --- 1b. Project Imports ---
from src.utils.config_loader import load_project_config, deep_format, expanduser_tree
from src.model import model_building
from src.training import model_training

# --- 1c. Logging Config ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # Ensure logging config is respected (override any module logs)
    
logging.basicConfig(
    level=logging.INFO,
    # format='%(levelname)s - %(message)s',  # Uncomment for short logging
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',  # Uncomment for full logging
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
config = load_project_config(config_path="config/project_config.yaml")
notebook = False

# --- 1d. Set up root directory paths in config ---

raw_data_root = config["global"]["paths"]["raw_data_root"]
results_root = config["global"]["paths"]["results_root"]

# Reformat config roots
config = deep_format(
    config,
    raw_data_root=raw_data_root,
    results_root=results_root
)
config = expanduser_tree(config)
            
# --- 1e. Set up seeding to define global states ---
random_seed = config["global"]["pipeline_settings"]["random_seed"]
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 1f. Define catchment(s) and API calls to Process --
catchments_to_process = config["global"]["pipeline_settings"]["catchments_to_process"]
run_defra_API_calls = config["global"]["pipeline_settings"]["run_defra_api"]
run_camels_API_calls = config["global"]["pipeline_settings"]["run_camels_api"]
run_outlier_detection = config["global"]["pipeline_settings"]["run_outlier_detection"]

# Run full pipeline by catchment
try:
    for catchment in catchments_to_process:

        # # --- Load training requirements ---
        
        # parquet_path = os.path.join(config[catchment]["paths"]["final_df_path"], 'processed_df.parquet')
        # processed_df = pd.read_parquet(parquet_path, engine='pyarrow')
        
        # aux_dir = config[catchment]["paths"]["aux_dir"]
        # train_station_ids = joblib.load(os.path.join(aux_dir, "train_station_ids.joblib"))
        # val_station_ids = joblib.load(os.path.join(aux_dir, "val_station_ids.joblib"))
        # test_station_ids = joblib.load(os.path.join(aux_dir, "test_station_ids.joblib"))
        # gwl_feats = joblib.load(os.path.join(aux_dir, "gwl_feats.joblib"))
        
        # graph_output_dir = config[catchment]["paths"]["graph_data_output_dir"]
        # edge_index_tensor = torch.load(os.path.join(graph_output_dir, "edge_index_tensor.pt"))
        # edge_attr_tensor = torch.load(os.path.join(graph_output_dir, "edge_attr_tensor.pt"))
        
        # # --- 6d. Creat PyG data object using partioned station IDs (from 6a) ---
        
        # # For feature ablation
        # processed_df_final = processed_df.drop(columns=['streamflow_total_m3']).copy()
        
        # # Run time approx. 12.5 mins to build 4018 timesteps of objects (0.19s per Object)
        # gwl_ohe_cols = joblib.load(os.path.join(config[catchment]["paths"]["scalers_dir"], "gwl_ohe_cols.pkl"))
        # all_timesteps_list = data_partitioning.build_pyg_object(
        #     processed_df=processed_df_final,
        #     sentinel_value=config["global"]["graph"]["sentinel_value"],
        #     train_station_ids=train_station_ids,
        #     val_station_ids=val_station_ids,
        #     test_station_ids=test_station_ids,
        #     gwl_feats=gwl_feats,
        #     gwl_ohe_cols=gwl_ohe_cols,
        #     edge_index_tensor=edge_index_tensor,
        #     edge_attr_tensor=edge_attr_tensor,
        #     scalers_dir=config[catchment]["paths"]["scalers_dir"],
        #     catchment=catchment
        # )

        # torch.save(all_timesteps_list, config[catchment]["paths"]["pyg_object_path"])
        # logger.info(f"Pipeline Step 'Build PyG Data Objects' complete for {catchment} catchment.\n")

        # --- 6e. Define Graph Adjacency Matrix (edge_index -> 8 nearest neighbours) ---
        # Already generated in Step 5e and incorporated into PyG objects in step 6d.
        # BUT: Build helper to add self loops (duplicate edges in both directions)

        # ====================================================================================================
        # SECTION 7: MODEL
        # ----------------------------------------------------------------------------------------------------
        # Instantiate GAT-LSTM Model using PyTorch Geometric:
        #   - Construct PyTorch Geometric Data objects (one per timestep), passing edge_index and edge_attr as
        #     separate arguments to the Data constructor, alongside x (node features) and y (targets).
        # ====================================================================================================

        # --- 7a. Build Data Loaders by Timestep ---
        
        all_timesteps_list = torch.load(config[catchment]["paths"]["pyg_object_path"])  # UPDATE TO INCLUDE FILENAME AND METADATA
        # full_dataset_loader = model_building.build_data_loader(
        #     all_timesteps_list=all_timesteps_list,
        #     batch_size = config["global"]["model"]["data_loader_batch_size"],
        #     shuffle = config["global"]["model"]["data_loader_shuffle"],
        #     catchment=catchment,
        #     seed=config["global"]["pipeline_settings"]["random_seed"]
        # )

        # logger.info(f"Pipeline Step 'Create PyG DataLoaders' complete for {catchment} catchment.\n")
        
        # --- 7b. Define Graph Neural Network Architecture including loss and optimiser definition ---

        # Adjust model architecture and params in catchment-specific config. TODO: Further optimise hyperparams.
        model, device, optimizer, criterion = model_building.instantiate_model_and_associated(
            all_timesteps_list=all_timesteps_list,
            config=config,
            catchment=catchment
        )

        logger.info(f"Pipeline Step 'Instantiate GAT-LSTM Model' complete for {catchment} catchment.\n")

        # ==============================================================================
        # SECTION 8: TRAINING
        # ==============================================================================
        
        # --- 8a. Implement Training Loop ---

        train_losses, val_losses = model_training.run_training_and_validation(
            num_epochs=config[catchment]["training"]["num_epochs"],
            early_stopping_patience=config[catchment]["training"]["early_stopping_patience"],
            lr_scheduler_factor=config[catchment]["training"]["lr_scheduler_factor"],
            lr_scheduler_patience=config[catchment]["training"]["lr_scheduler_patience"],
            min_lr=config[catchment]["training"]["min_lr"],
            gradient_clip_max_norm=config[catchment]["training"]["gradient_clip_max_norm"],
            model_save_dir=config[catchment]["paths"]["model_dir"],
            loss_delta=config[catchment]["training"]["loss_delta"],
            verbose=config[catchment]["training"]["verbose"],
            catchment=catchment,
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            all_timesteps_list=all_timesteps_list,
            scalers_dir=config[catchment]["paths"]["scalers_dir"],
            config=config
        )

        logger.info(f"Pipeline Step 'Train and Validate Model' complete for {catchment} catchment.")

        relative_train_path, relative_val_path = model_training.save_train_val_losses(
            output_analysis_dir=config[catchment]["paths"]["model_dir"],
            train_losses=train_losses,
            val_losses=val_losses,
            config=config,
            catchment=catchment
        )

        logger.info(f"Pipeline Step 'Save Training and Validation Losses' complete for {catchment} catchment.")

        # --- 8b. Model Checkpointing and Logging ---

# If critical pipeline error, exit with an error code
except Exception as e:
    logger.critical(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)
    sys.exit(1)