import os
import sys
import logging
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from scipy.stats import skew
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

        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        logging.info(f"Saved {feature}_boxcox_lambda to {config_path}")

    except Exception as e:
        logging.error(f"Failed to save {feature} lambda to config.yaml: {e}")

def derive_rainfall_features(csv_dir: str, processed_output_dir: str, start_date: str,
                             end_date: str, config_path: str, catchment: str):
    """
    Derive lags and rolling averages (right aligned) from rainfall data and clip
    data to final model bounds
    """
    # Read in aggregated rainfall data
    input_csv_path = f"{csv_dir}rainfall_daily_catchment_sum.csv"
    rainfall_df = pd.read_csv(input_csv_path)
    
    # Log transform the raw rainfall and lagged data to reduce skew
    logging.info(f"Tranforming rainfall data for {catchment} catchment...\n")
    logging.info(f"    Initial Rainfall Skewness: {skew(rainfall_df['rainfall_volume_m3']):.4f}")
    
    # Apply box cox transform to raw data column
    transformed_rainfall_volume, lambda_val = boxcox(rainfall_df['rainfall_volume_m3'] + 1)
    rainfall_df['rainfall_volume_m3'] = transformed_rainfall_volume
    logging.info(f"    Transformed Rainfall Skewness (Box-Cox, lambda={lambda_val:.4f}):"
                 f" {skew(rainfall_df['rainfall_volume_m3']):.4f}\n")

    # Save lambda for mathematical inversion after modelling
    _save_lambda_to_config(lambda_val, config_path, catchment, 'rainfall')

    # Build lags (of transformed data)
    logging.info(f'Building 7 days of rainfall data lags for {catchment} catchment...')
    for i in range(1, 8):
        column_name = 'rainfall_lag_' + str(i)
        rainfall_df[column_name] = rainfall_df['rainfall_volume_m3'].shift(periods=i)
        logging.info(f"    Lag {i} built ({len(rainfall_df[column_name])} values shifted)")
    
    # Build 30 rolling averages
    logging.info(f'Building 30 day rolling rainfall averages for {catchment} catchment...')
    rainfall_df['rolling_30'] = rainfall_df['rainfall_volume_m3'].rolling(window=30).mean()
    
    # Build 60 rolling averages
    logging.info(f'Building 60 day rolling rainfall averages for {catchment} catchment...\n')
    rainfall_df['rolling_60'] = rainfall_df['rainfall_volume_m3'].rolling(window=60).mean()
    
    # Ensure datetime type and set as index
    rainfall_df['time'] = pd.to_datetime(rainfall_df['time'])
    rainfall_df = rainfall_df.set_index('time')
    
    # Clip data to model bounds
    logging.info(f'Clipping rainfall data to model bounds of {start_date} to {end_date}')
    rainfall_trimmed = rainfall_df.loc[start_date:end_date].copy()
    logging.info(f'{len(rainfall_df)-len(rainfall_trimmed)} dates trimmed')
    
    # Save as csv
    final_csv_path = f"{processed_output_dir}rainfall_daily_catchment_sum_log_transform.csv"
    logging.info(f"Saving catchment-summed log-transformed daily rainfall to CSV: {final_csv_path}")

    # Save log transformed df as csv to merge into main model df
    rainfall_trimmed.to_csv(final_csv_path)
    
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
