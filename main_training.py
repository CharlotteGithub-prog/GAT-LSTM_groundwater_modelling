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
import argparse
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

# --- 1g. Override config for specific test run ---

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, default=os.getenv("TEST",""))
parser.add_argument("--vals", type=str, default=os.getenv("VALS",""))
args = parser.parse_args()

test_override = [args.test] if args.test else None
vals_override = args.vals.split(":") if args.vals else None

for catchment in config["global"]["pipeline_settings"]["catchments_to_process"]:
    if test_override:
        config[catchment]["model"]["data_partioning"]["test_station_shortlist"] = test_override
    if vals_override:
        config[catchment]["model"]["data_partioning"]["val_station_shortlist"] = vals_override
    
    logger.info(f"[Overrides] test={test_override} vals={vals_override} for {catchment} catchment")

# Run full pipeline by catchment
try:
    for catchment in catchments_to_process:

        # --- 7a. Get Data Loaders by Timestep ---
        
        # get current_station name and normalise to a single station name string
        current_station = config[catchment]["model"]["data_partioning"]["test_station_shortlist"]
        
        if isinstance(current_station, (list, tuple)):
            if len(current_station) == 0:
                raise ValueError("test_station_shortlist is empty.")
            current_station = current_station[0]
        
        elif not isinstance(current_station, str):
            raise TypeError(f"Unexpected type for test_station_shortlist: {type(current_station)}")
        
        current_station = current_station.strip().lower()
        
        timesteps_path = config[catchment]["paths"]["pyg_object_path"]
        timesteps_dir, filename = os.path.split(timesteps_path)
        base, ext = os.path.splitext(filename)
        
        suffix_map = {
            "ainstable": "ainstable_20250818_215159",  # Older: 20250814_110329
            "baronwood": "baronwood_20250818_205012",
            "bgs_ev2": "bgs_ev2_20250818_212241",
            "castle_carrock": "castle_carrock_20250818_213009",
            "cliburn_town_bridge_2": "cliburn_town_bridge_2_20250818_214321",
            "coupland": "coupland_20250818_173942",
            "croglin": "croglin_20250820_010926",  # with no lags; other: 20250818_180550
            "east_brownrigg": "east_brownrigg_20250818_184745",
            "great_musgrave": "great_musgrave_20250818_190654",
            "hilton": "hilton_20250818_200206",
            "longtown": "longtown_20250818_201159",
            "renwick": "renwick_20250818_202143",
            "scaleby": "scaleby_20250818_202740",
            "skirwith": "skirwith_20250818_203906"
        }
        
        # get full all_timesteps_list filepath
        new_filename = f"{base}_{suffix_map[current_station]}{ext}" \
                       if current_station in suffix_map else filename
        full_filepath = os.path.join(timesteps_dir, new_filename)
        logger.info(f"Using all_timesteps_list path: {full_filepath}")

        # CAtch errors immediately
        if not os.path.isfile(full_filepath):
            raise FileNotFoundError(f"Missing PyG file: {full_filepath}")
        
        all_timesteps_list = torch.load(full_filepath)
        
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
        
        base_scalers_dir = config[catchment]["paths"]["scalers_dir"]
        scalers_dir = (os.path.join(base_scalers_dir, suffix_map[current_station]) 
            if current_station in suffix_map else base_scalers_dir)
        if not os.path.isdir(scalers_dir):
            raise FileNotFoundError(f"scalers_dir not found: {scalers_dir}")
        logger.info(f"Using scalers_dir: {scalers_dir}")

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
            scalers_dir=scalers_dir,
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