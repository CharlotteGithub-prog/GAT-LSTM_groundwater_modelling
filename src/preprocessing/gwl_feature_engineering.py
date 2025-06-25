import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from feature_engine.timeseries.forecasting import LagFeatures

def build_lags(df_dict: dict, catchment: str):
    
    # Loop through stations in dictionary
    for station, df in df_dict.items():
        logging.info(f"Building lagged gwl features for {station}...")
        
        # Build 7 days of lagged groundwater data
        for i in range(1, 8):
            column_name = 'gwl_lag' + str(i)
            df[column_name] = df['value'].shift(periods=i)
            
        logging.info(f"{station}: Data Lagged by {i} day(s)\n")
    
    logging.info(f"Lagged data built for all stations in {catchment} catchment\n.")
    return df_dict

def build_seasonality_features(df_dict: dict, catchment: str):
    
    # Build sin and cos seasonality to always have a non-zero seasonality
    for station, df in df_dict.items():
        logging.info(f"Building seasonality features for {station}...")
        
        # Calculate doy using DateTime index
        day_of_year = df.index.dayofyear
        
        # Calculate cosine and sine feature values
        df['season_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['season_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    logging.info(f"Seasonality data built for all stations in {catchment} catchment\n.")
    return df_dict

def trim_to_model_bounds(df_dict: dict, start_date: str, end_date: str):
    """
    tbd
    """
    # Initialise new dataframe to store imputed, cleaned and trimmed data
    trimmed_df = {}
    
    logging.info(f"Trimming data frames to final model range...")

    # Trim data frame by station to bounds of model
    for station, df in df_dict.items():
        trimmed_df[station] = df.loc[start_date:end_date].copy()
    
    logging.info(f"All stations trimmed from {start_date} to {end_date}\n")
                
    return trimmed_df

def plot_final_gwl_timeseries(df_dict: dict, output_path: str, highlight_column: str = 'data_type', dpi: int = 300):
    """
    Plot groundwater level time series with colour-coded imputation types using the 'data_type' column.

    """
    # Define fixed colors for each data type
    color_map = {
        'raw': '#1f77b4',               
        'interpolated_short': '#ff7f0e',
        'imputed_long': "#56cb54", 
        'masked': "#aeaeae"         
    }

    logging.info(f"Unique data types:")
    for station_name, df in df_dict.items():
        logging.info(f"    {station_name}: {df['data_type'].dropna().unique()}")
        fig, ax = plt.subplots(figsize=(15, 4))
        
        # Draw base line
        ax.plot(df.index, df['value'],
            color=color_map['raw'],
            linewidth=0.9,
            zorder=0,
            label='Raw')  

        # Loop through each unique value in the highlight_column
        for data_type, color in color_map.items():
            if data_type not in df[highlight_column].unique():
                continue

            # Boolean mask
            mask = df['data_type'] == data_type

            # Group by consecutive values of True using .diff() and cumsum
            segment_groups = (mask != mask.shift()).cumsum()

            for _, segment in df[mask].groupby(segment_groups):
                ax.plot(segment.index, segment['value'],
                        label=data_type.replace('_', ' ').title(),
                        color=color,
                        alpha=0.9,
                        linewidth=0.9)

        # Axis formatting
        ax.set_title(f'{station_name} Groundwater Level (Final)')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_xlim(pd.Timestamp('2013-06-30'), pd.Timestamp('2025-06-30'))
        ax.set_ylabel('Groundwater Level (mAOD)')
        ax.grid(True)
        
        # Deduplication and Plot Legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels_dict = dict(zip(labels, handles))
        sorted_handles = [unique_labels_dict[label] for label in unique_labels_dict]
        ax.legend(sorted_handles, unique_labels_dict, loc="center left", bbox_to_anchor=(1.01, 0.5), title='Data Type')

        # Tidy and save
        fig.tight_layout()
        plt.savefig(f"{output_path}/final_plots/{station_name}_final_gwl_plot.png", dpi=dpi)
        plt.close()

def save_trimmed_dict(trimmed_df_dict: dict, trimmed_output_dir: str, model_start_date: str, model_end_date: str):
    """
    Copy and save dictionary of trimmed data frames before engingeering final gwl features.
    """
    logging.info(f"Saving trimmed data from {model_start_date[:-9]} to {model_end_date[:-9]}...\n")
    
    # Confirm directory exists and make copy of dict to save without warning
    os.makedirs(trimmed_output_dir, exist_ok=True)
    trimmed_df_output = trimmed_df_dict.copy()
    
    # Save dict to catchment data
    if trimmed_df_dict:
        for station_name, df in trimmed_df_dict.items():
            output_path = os.path.join(trimmed_output_dir, f"{station_name}_trimmed.csv")
            df.to_csv(output_path)
    else:
        logging.warning(f"    No trimmed data available. Skipping save.\n")

def trim_and_save(df_dict: dict, model_start_date: str, model_end_date: str, trimmed_output_dir: str, ts_path: str,
                  notebook: bool, highlight_column: str = 'data_type', dpi: int = 300):
    """
    Trims dataframes to model bounds (2014–2024 inclusive), plots final ts data,
    fills missing quality markers, drops redundant columns, and saves CSVs per station.
    """
    # --- Trim df's to model data range ---
    trimmed_df_dict = trim_to_model_bounds(
        df_dict=df_dict,
        start_date=model_start_date,
        end_date=model_end_date  
    )
    
    # --- Plot final df with raw and imputed data, marked using data_type column ---
    plot_final_gwl_timeseries(
        df_dict=trimmed_df_dict,
        output_path=ts_path
    )
    
    # Drop dateTime col as already index and fill any missing quality markers
    for station, df in trimmed_df_dict.items():
        
        # Drop unneeded columns
        df.drop(columns='date', inplace=True)
        df.drop(columns='measure', inplace=True)
        logging.info(f"Columns 'date' and 'measure' dropped for {station}.")
        
        # Fill NaNs
        df.loc[(df['data_type'] == 'raw') & (df['quality'].isna()), 'quality'] = 'Unchecked'
        df.loc[df['quality'].isna(), 'quality'] = 'Missing'
        logging.info(f"All missing quality markers filled for {station}.")

    # Save trimmed dataframe dict
    save_trimmed_dict(
        trimmed_df_dict=trimmed_df_dict,
        trimmed_output_dir=trimmed_output_dir,
        model_start_date=model_start_date,
        model_end_date=model_end_date
    )
    
    return trimmed_df_dict