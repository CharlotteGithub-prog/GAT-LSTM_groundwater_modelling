import os
import sys
import logging
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import skew, boxcox
from feature_engine.timeseries.forecasting import LagFeatures

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def _save_lambda_to_config(lambda_val, config_path, catchment, feature):
    """
    Save calculated lambda value to config for later inversion of transformations for
    interpretability (by feature).
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f)

        # Ensure the 'preprocessing' key exists under the specific catchment
        if 'preprocessing' not in config[catchment]:
            config[catchment]['preprocessing'] = {}

        # Save lambda to yaml
        config[catchment]["preprocessing"][f"{feature}_boxcox_lambda"] = float(lambda_val)

        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        logging.info(f"Saved {feature}_boxcox_lambda to {config_path}")

    except Exception as e:
        logging.error(f"Failed to save {feature} lambda to config.yaml: {e}")

def derive_rainfall_features(csv_dir: str, processed_output_dir: str, start_date: str,
                             end_date: str, config_path: str, pred_frequency: str, catchment: str):
    """
    Derive lags and rolling averages (right aligned) from rainfall data and clip
    data to final model bounds
    """
    # Read in aggregated rainfall data (still at a daily level here)
    input_csv_path = f"{csv_dir}rainfall_daily_catchment_sum.csv"
    rainfall_df = pd.read_csv(input_csv_path)
    
    # Ensure datetime index first (particularly for rolling av's)
    rainfall_df['time'] = pd.to_datetime(rainfall_df['time'])
    rainfall_df = rainfall_df.set_index('time').sort_index()
    
    # Build 30 rolling averages
    logging.info(f'Building 30 day rolling rainfall averages for {catchment} catchment...')
    rainfall_df['rolling_30'] = rainfall_df['rainfall_volume_m3'].rolling(window='30D', min_periods=1).mean()
    
    # Build 60 rolling averages
    logging.info(f'Building 60 day rolling rainfall averages for {catchment} catchment...\n')
    rainfall_df['rolling_60'] = rainfall_df['rainfall_volume_m3'].rolling(window='60D', min_periods=1).mean()
    
    # Resample to pred_frequency timestep
    frequency_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}
    clean_pred_frequency = pred_frequency.lower().strip()
    frequency = frequency_map.get(clean_pred_frequency)
    if frequency is None:
        raise ValueError(f"Invalid prediction frequency: {pred_frequency}. Must be 'daily', "
                         f"'weekly', or 'monthly'.")
    
    # Apply resampling
    if clean_pred_frequency != 'daily':
        agg_map = {
            'rainfall_volume_m3': 'sum',
            'rolling_30': 'first',
            'rolling_60': 'first'
        }
        rainfall_df = rainfall_df.resample(frequency).agg(agg_map).copy()
        logger.info(f"Total rainfall aggregated to {pred_frequency} timestep.")
        
    # Apply box cox transform to raw data column
    logging.info(f"Tranforming rainfall data for {catchment} catchment...\n")
    logging.info(f"    Initial Rainfall Skewness: {skew(rainfall_df['rainfall_volume_m3']):.4f}")
    transformed_rainfall_volume, lambda_val = boxcox(rainfall_df['rainfall_volume_m3'] + 1)
    rainfall_df['rainfall_volume_m3'] = transformed_rainfall_volume
    logging.info(f"    Transformed Rainfall Skewness (Box-Cox, lambda={lambda_val:.4f}):"
                 f" {skew(rainfall_df['rainfall_volume_m3']):.4f}\n")

    # Save lambda to config for mathematical inversion after modelling
    _save_lambda_to_config(lambda_val, config_path, catchment, 'rainfall')

    # Build lags (of transformed data)
    logging.info(f'Building 7 {pred_frequency} lags for rainfall...')
    for i in range(1, 8):
        rainfall_df[f'rainfall_lag_{i}'] = rainfall_df['rainfall_volume_m3'].shift(periods=i)
        logging.info(f"    Lag {i} built ({len(rainfall_df[f'rainfall_lag_{i}'])} values shifted)")
    
    # Ensure datetime type and set as index
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Clip data to model bounds
    logging.info(f'Clipping rainfall data to model bounds of {start} to {end}')
    rainfall_trimmed = rainfall_df.loc[start:end].copy()
    logging.info(f'{len(rainfall_df)-len(rainfall_trimmed)} dates trimmed')
    
    # Save as csv
    final_csv_path = f"{processed_output_dir}rainfall_{pred_frequency}_catchment_sum_log_transform.csv"
    logging.info(f"Saving catchment-summed log-transformed {pred_frequency} rainfall to CSV: {final_csv_path}")
    rainfall_trimmed.to_csv(final_csv_path, index=True, index_label='time')
    
    return rainfall_trimmed

def transform_aet_data(merged_ts_aet: pd.DataFrame, catchment: str):
    """
    Transform skew of AET data.
    """
    # Define config path
    config_path = "config/project_config.yaml"
    
    # Log transform the raw rainfall and lagged data to reduce skew
    logging.info(f"Tranforming rainfall data for {catchment} catchment...\n")
    logging.info(f"    Initial Rainfall Skewness: {skew(merged_ts_aet['aet_total_volume_m3']):.4f}")
    
    # Apply box cox transform to raw data column
    transformed_rainfall_volume, lambda_val = boxcox(merged_ts_aet['aet_total_volume_m3'] + 1)
    merged_ts_aet['aet_total_volume_m3'] = transformed_rainfall_volume
    logging.info(f"    Transformed Rainfall Skewness (Box-Cox, lambda={lambda_val:.4f}):"
                 f" {skew(merged_ts_aet['aet_total_volume_m3']):.4f}\n")

    # Save lambda for mathematical inversion after modelling
    _save_lambda_to_config(lambda_val, config_path, catchment, 'aet')
    
    return merged_ts_aet
