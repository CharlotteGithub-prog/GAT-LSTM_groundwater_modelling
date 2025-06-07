import os
import ast
import logging
import requests
import pandas as pd
from pyproj import Transformer

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

def load_timeseries_to_dict(stations_df: pd.DataFrame, col_order: list,
                            data_dir: str, inclusion_threshold: int):
    """
    Loads and cleans groundwater level timeseries data from CSV files.
    
    - Removes 'qcode' column if present.
    - Ensures all columns in `col_order` are present (filling missing with NA).
    - Reorders columns to match `col_order`.
    - Returns a dictionary of cleaned DataFrames keyed by station name.
    """ 
    logging.info(f'Converting API csv data to reference dict...\n')
    
    # Save pandas dataframes to a dictionary by station name
    time_series_data = {}

    for index, row in stations_df.iterrows():
        uri = row['measure_uri']
        measure_id = uri.split("/")[-1]
        name = row['station_name'].title().strip().replace(" ", "_")
        
        # Read CSV into placeholder df to manipulate
        temp_df = pd.read_csv(f"{data_dir}{measure_id}_readings.csv", index_col=0, low_memory=False)
        
        # Drop 'qcode' column if present
        if 'qcode' in temp_df.columns:
            temp_df = temp_df.drop(columns=['qcode'])
        
        # Reorder columns (fill missing with NA)
        for col in col_order:
            if col not in temp_df.columns:
                print(f'Warning: {name} did not contain {col}')
                temp_df[col] = pd.NA
        temp_df = temp_df[col_order]
        
        # Save to dictionary if data over threshold
        if len(temp_df) > inclusion_threshold:
            time_series_data[name] = temp_df
        else:
            logging.info(f"Station {name} contained insufficient data -> dropping dataframe."
                         f"({len(temp_df)} < {inclusion_threshold})")
        
        logging.info(f"{name} successfully saved to dict.")
    
    logging.info(f"{len(time_series_data)} stations saved to dict.\n")    
    return time_series_data