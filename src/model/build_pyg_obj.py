# Load library imports
import os
import sys
import torch
import joblib
import random
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter

# Load project Imports
from src.utils.config_loader import load_project_config, deep_format, expanduser_tree
from src.preprocessing.data_partitioning import build_pyg_object

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)
config = load_project_config(config_path="config/project_config.yaml")
notebook = True

# Set up root directory paths in config
raw_data_root = config["global"]["paths"]["raw_data_root"]
results_root = config["global"]["paths"]["results_root"]

# Reformat config roots
config = deep_format(
    config,
    raw_data_root=raw_data_root,
    results_root=results_root
)
config = expanduser_tree(config)

# Set up seeding to define global states
random_seed = config["global"]["pipeline_settings"]["random_seed"]
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define notebook demo catchment
catchments_to_process = config["global"]["pipeline_settings"]["catchments_to_process"]
catchment = catchments_to_process[0]
run_defra_API_calls = config["global"]["pipeline_settings"]["run_defra_api"]

logger.info(f"Show Notebook Outputs: {notebook}")
logger.info(f"Notebook Demo Catchment: {catchment.capitalize()}")

# --- Load training requirements ---

parquet_path = os.path.join(config[catchment]["paths"]["final_df_path"], 'processed_df.parquet')
processed_df = pd.read_parquet(parquet_path, engine='pyarrow')

aux_dir = config[catchment]["paths"]["aux_dir"]
train_station_ids = joblib.load(os.path.join(aux_dir, "train_station_ids.joblib"))
val_station_ids = joblib.load(os.path.join(aux_dir, "val_station_ids.joblib"))
test_station_ids = joblib.load(os.path.join(aux_dir, "test_station_ids.joblib"))
gwl_feats = joblib.load(os.path.join(aux_dir, "gwl_feats.joblib"))

graph_output_dir = config[catchment]["paths"]["graph_data_output_dir"]
edge_index_tensor = torch.load(os.path.join(graph_output_dir, "edge_index_tensor.pt"))
edge_attr_tensor = torch.load(os.path.join(graph_output_dir, "edge_attr_tensor.pt"))

# --- 6d. Creat PyG data object using partioned station IDs (from 6a) ---

# For feature ablation
# processed_df_final = processed_df.drop(columns=['streamflow_total_m3']).copy()
# processed_df_final = processed_df.drop(columns=['streamflow_total_m3', 'HOST_soil_class_freely_draining_soils', 'HOST_soil_class_high_runoff_(impermeable)', 
#                                                'HOST_soil_class_impeded_saturated_subsurface_flow', 'HOST_soil_class_peat_soils', 'aquifer_productivity_High',
#                                                'aquifer_productivity_Low', 'aquifer_productivity_Mixed', 'aquifer_productivity_Moderate',
#                                                'aquifer_productivity_nan', 'gwl_mean', 'gwl_dip', 'distance_to_river']).copy()

processed_df_final = processed_df.copy()

# Run time approx. 12.5 mins to build 4018 timesteps of objects (0.19s per Object)
gwl_ohe_cols = joblib.load(os.path.join(config[catchment]["paths"]["scalers_dir"], "gwl_ohe_cols.pkl"))
all_timesteps_list = build_pyg_object(
    processed_df=processed_df_final,
    sentinel_value=config["global"]["graph"]["sentinel_value"],
    train_station_ids=train_station_ids,
    val_station_ids=val_station_ids,
    test_station_ids=test_station_ids,
    gwl_feats=gwl_feats,
    gwl_ohe_cols=gwl_ohe_cols,
    edge_index_tensor=edge_index_tensor,
    edge_attr_tensor=edge_attr_tensor,
    scalers_dir=config[catchment]["paths"]["scalers_dir"],
    catchment=catchment
)

torch.save(all_timesteps_list, config[catchment]["paths"]["pyg_object_path"])
logger.info(f"Pipeline Step 'Build PyG Data Objects' complete for {catchment} catchment.\n")