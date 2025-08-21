import os
import sys
import torch
import random
import joblib
import logging
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from collections import Counter
from permetrics.regression import RegressionMetric
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load project Imports
from src.utils.config_loader import load_project_config, deep_format, expanduser_tree
from src.model.model_building import build_data_loader, instantiate_model_and_associated
from src.utils.config_loader import load_project_config

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def run_testing_and_plotting(model, path, scaler_path, test_station, device, all_timesteps_list, config,
                             catchment, iteration: int = 1):
    """
    Testing and Plotting Process: 
       - Transfer .pt model across from NCC using rsync
       - Transfer timestep PyG object from hard drive and update filepath in section above
       - Update path below
       - Update test_station below
       - Update scalers below
       - Update test and val station lists in config
    """

    path = "data/04_model/eden/model/pt_model/model_20250818-193354_GATTrue_LSTMTrue_GATH12_GATD0-4_GATHC64_GATOC64_GATNL2_LSTHC32_LSTNL1_OUTD1_LR0-001_WD0-001_SM0-1_E250_ESP35_LRSF0-5_LRSP8_MINLR1e-06_LD0-0001_GCMN1-0.pt"
    scaler_path = "data/03_graph/eden/scalers/great_musgrave_20250818_190654/target_scaler.pkl"
    iteration = 1  # Complete
    test_station = "great_musgrave"

    mean_gwl_map = {
        "ainstable": 84.6333698214874,
        "baronwood": 85.8373720963633,
        "bgs_ev2": 87.2166125260539,
        "castle_carrock": 133.19521880854,
        "cliburn_town_bridge_2": 110.805906037388,
        "coupland": 135.670365012452,
        "croglin": 167.758299820582,
        "east_brownrigg": 106.74319765862,
        "great_musgrave": 152.209015790055,
        "hilton": 214.739017912584,
        "longtown": 18.1315500711501,
        "renwick": 177.683627274689,
        "scaleby": 41.1093269995661,
        "skirwith": 130.796279748829
    }

    mean_gwl = mean_gwl_map[test_station]

    best_model = model  # Assume model object already defined and moved to correct device
    best_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    best_model.eval()
    logger.info(f"Loaded best model from {path}")

    # Load target scaler
    target_scaler = joblib.load(scaler_path)
    logger.info(f"Loaded target scaler from: {scaler_path}")

    scale = torch.tensor(target_scaler.scale_, device=device)
    mean = torch.tensor(target_scaler.mean_, device=device)

    # Initialise global LSTM state
    if best_model.run_LSTM:
        lstm_state_store = {
            'h': torch.zeros(best_model.num_layers_lstm, best_model.num_nodes, best_model.hidden_channels_lstm).to(device),
            'c': torch.zeros(best_model.num_layers_lstm, best_model.num_nodes, best_model.hidden_channels_lstm).to(device)
        }
    else:
        lstm_state_store = None

    # Prepare lists for evaluation
    test_predictions_unscaled = []
    test_actuals_unscaled = []
    fusion_alphas = [] 

    logger.info("--- Starting Model Evaluation on Test Set ---")
    test_loop = tqdm(all_timesteps_list, desc="Evaluating on Test Set", leave=False)

    with torch.no_grad():
        for i, data in enumerate(test_loop):
            data = data.to(device)
            test_mask = data.test_mask
            known_data_mask = (data.train_mask | data.val_mask | data.test_mask)

            # Skip timesteps with no nodes with known ground truth
            if known_data_mask.sum() == 0:
                continue

            # Model forward pass on full node set
            predictions_all, (h_new, c_new), returned_node_ids = best_model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                current_timestep_node_ids=data.node_id,
                lstm_state_store=lstm_state_store
            )
            
            # Update LSTM memory for current nodes
            if best_model.run_LSTM:
                lstm_state_store['h'][:, returned_node_ids, :] = h_new.detach()
                lstm_state_store['c'][:, returned_node_ids, :] = c_new.detach()

            # Filter predictions/targets to test nodes
            preds_std = predictions_all[test_mask]
            targets_std = data.y[test_mask]

            # Inverse transform to original scale
            preds_np = preds_std.cpu().numpy()
            targets_np = targets_std.cpu().numpy()

            preds_unscaled = target_scaler.inverse_transform(preds_np)
            targets_unscaled = target_scaler.inverse_transform(targets_np)

            test_predictions_unscaled.extend(preds_unscaled.flatten())
            test_actuals_unscaled.extend(targets_unscaled.flatten())
            
            # Capture residual contribution relative to baseline (for interpretability)
            if best_model.run_GAT and best_model.run_LSTM:
                dbg = getattr(best_model, "last_debug", None)
                if dbg is not None:
                    residual = dbg.get("residual", None)
                    baseline = dbg.get("baseline", None)
                    if isinstance(residual, torch.Tensor) and isinstance(baseline, torch.Tensor):
                        res_abs = torch.abs(residual[test_mask]).sum().item()
                        base_abs = torch.abs(baseline[test_mask]).sum().item()
                        if base_abs > 0:
                            fusion_alphas.append(res_abs / base_abs)  # store rel contribution ratio

            if i < 5:  # Show first few predictions
                print("Sample predictions (m AOD):", preds_unscaled[:5].flatten())
                print("Sample actuals     (m AOD):", targets_unscaled[:5].flatten())

    # --- Final model prediction evaluation ---

    if len(test_actuals_unscaled) > 0:
        loss_type = config[catchment]["training"]["loss"]

        if loss_type == "MAE":
            final_test_metric = mean_absolute_error(test_actuals_unscaled, test_predictions_unscaled)
            logger.info(f"--- Final Test Set MAE (m AOD): {final_test_metric:.4f} ---\n")

        elif loss_type == "MSE":
            final_test_metric = mean_squared_error(test_actuals_unscaled, test_predictions_unscaled)
            logger.info(f"--- Final Test Set MSE (m AOD²): {final_test_metric:.4f} ---\n")

        else:
            logger.warning(f"Unrecognized loss type '{loss_type}' in config — skipping final metric calculation.\n")
    else:
        logger.warning("No test data found — check 'data.test_mask'.\n")

    logger.info("--- Model Evaluation on Test Set Complete ---\n")

    # Calculate and display the global average residual contribution
    if fusion_alphas:
        avg_rel_contrib = np.mean(fusion_alphas) * 100
        logger.info("--- Residual Contribution (on test node) ---")
        logger.info(f"Average GAT Residual Contribution: {avg_rel_contrib:.2f}%")
        logger.info(f"Average LSTM Contribution: {100 - avg_rel_contrib:.2f}%")
        logger.info("-------------------------------------------\n")