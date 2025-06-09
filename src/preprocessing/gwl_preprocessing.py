import os
import ast
import joblib
import logging
import numpy as np
import pandas as pd
import multiprocessing
import logging.handlers
from hampel import hampel
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from joblib import Parallel, delayed

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

def load_timeseries_to_dict(stations_df: pd.DataFrame, col_order: list,
                            data_dir: str, inclusion_threshold: int):
    """
    Loads and cleans groundwater level timeseries data from CSV files.
    
    - Removes 'qcode' column if present.
    - Ensures all columns in `col_order` are present (filling missing with NA).
    - Reorders columns to match `col_order`.
    - Removes stations with number of points below threshold.
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
            logging.info(f"{name} successfully saved to dict.")
        else:
            logging.info(f"Station {name} contained insufficient data -> dropping dataframe."
                         f"({len(temp_df)} < {inclusion_threshold})")
    
    logging.info(f"{len(time_series_data)} stations saved to dict.\n")    
    return time_series_data

def plot_timeseries(time_series_df: pd.DataFrame, station_name: str, output_path: str,
                    outlier_mask: pd.Series = None, title_suffix: str = "", save_suffix: str = "",
                    notebook: bool = False, plot_outliers: bool = True, dpi: int = 300):
    """
    Plot timeseries data colour coded by quality mark.
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Define fixed colours for each quality level
    quality_colors = {
        'Good': '#70955F',
        'Estimated': '#549EB1',
        'Suspect': '#DF6607',
        'Unchecked': '#e89c1d',
        'Missing': '#9c9acd'
    }

    for quality, color in quality_colors.items():
        temp = time_series_df.copy()
        temp['value'] = temp['value'].where(temp['quality'] == quality, np.nan)
        ax.plot(
            temp['dateTime'],
            temp['value'],
            label=quality,
            color=color,
            alpha=0.8,
            linewidth=1.5
        )

    # If an outlier mask is identifed, plot the outliers
    if plot_outliers:
        if outlier_mask is not None and outlier_mask.sum() > 0:
            
            # Apply outlier mask to identify only points to mark with X
            corrected_values = time_series_df['value'][outlier_mask]
            corrected_datetimes = time_series_df['dateTime'][outlier_mask]

            # Plot markers for the detected outliers
            ax.scatter(
                corrected_datetimes,
                corrected_values,
                color='red',
                marker='x',
                s=50, # marker size
                label='Outlier',
                zorder=5 # Ensures markers are on top
            )
        
        else:
            logger.info(f"No outliers to plot for station {station_name}")

    # Apply auto locators and formatters to clean up ticks
    locator = mdates.AutoDateLocator(minticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    ax.set_title(f'{station_name} Groundwater Level 2014-2024{title_suffix}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Groundwater Level (mAOD)')
    ax.grid(True)
    ax.legend(title="Quality", loc="center left", bbox_to_anchor=(1.01, 0.5))

    plt.tight_layout()
    plt.savefig(f"{output_path}{station_name}{save_suffix}.png", dpi=dpi)
    
    if not notebook:
        plt.close()

    # return plt

def initial_threshold_cleaning(df: pd.DataFrame, station_name: str, iqr_multiplier: float = 5.0):
    """
    Performs initial data type conversion, drops unparseable rows,
    and applies a hard realistic range check based on IQR calcs.
    """
    df_cleaned = df.copy()
    
    # --- Dynamic Hard Range Check using IQR ---
    clean_values = df_cleaned['value'].dropna()
    
    if not clean_values.empty:
        Q1 = clean_values.quantile(0.25)
        Q3 = clean_values.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define limits using 3.0 for iqr_multiplier (as common for extreme outliers)
        lower_bound_iqr = Q1 - (iqr_multiplier * IQR)
        upper_bound_iqr = Q3 + (iqr_multiplier * IQR)
        logger.info(f"Station {station_name}: Hard limits {lower_bound_iqr:.2f} - {upper_bound_iqr:.2f}")

        # Identify values outside this dynamic range
        out_of_range_mask = (df_cleaned['value'] < lower_bound_iqr) | \
                            (df_cleaned['value'] > upper_bound_iqr)
        
        num_out_of_range = out_of_range_mask.sum()
        if num_out_of_range > 0:
            logger.info(f"Station {station_name}: Identified {num_out_of_range} points outside"
                        f"IQR-based range and set to NaN.")
            df_cleaned.loc[out_of_range_mask, 'value'] = np.nan
        else:
            logger.info(f"Station {station_name}: No points out of range.")
        
    return df_cleaned

def identify_residual_outliers(original_values: pd.Series):
    """
    Identifies potential outliers based on residuals (deviation from rolling median)
    that occur near data gaps (NaNs ahead). Returns a boolean mask.
    """
    # Rolling median and residuals
    rolling_median = original_values.rolling(window=30, min_periods=10, center=False).median()
    residual = (original_values - rolling_median).abs()
    residual_threshold = 0.3  # Threshold for significant deviation (in meters)

    # Identify values near a gap (NaN coming next)
    gap_ahead_mask = original_values.shift(-1).isna() | original_values.shift(-2).isna()

    # Combine into a new mask iddentifying points at end of a segment
    residual_outlier_mask = residual > residual_threshold
    end_of_segment_mask = residual_outlier_mask & gap_ahead_mask & original_values.notna()
    
    return end_of_segment_mask

def outlier_detection(gwl_time_series_dict: dict, output_path: str, dpi: int, dict_output: str,
                      notebook: bool = False):
    """
    Detects and corrects outliers in groundwater level time series data using 
    Hampel filtering and residual-based heuristics. Also generates diagnostic plots.

    Args:
        gwl_time_series_dict (dict): Dictionary mapping station names to raw time series DataFrames.
        output_path (str): Directory path to save output plots for each station.
        notebook (bool, optional): If True, displays plots inline (for use in Jupyter notebooks). 
                                   Defaults to False.

    Returns:
        dict: Dictionary of cleaned and filtered DataFrames keyed by station name.
    """
    total_stations = len(gwl_time_series_dict)
    loop_count = 1
    
    # Initialise dict to store the processed DataFrames
    processed_gwl_time_series_dict = {}
    
    for station_name, raw_csv in gwl_time_series_dict.items():
        
        # To keep track of how far through the processing is
        logging.info(f'Processing {loop_count} / {total_stations}: {station_name}...\n')
        loop_count += 1

        # Ensure 'value' column is numeric and plot 'before' plot of raw ts data
        raw_csv['value'] = pd.to_numeric(raw_csv['value'], errors='coerce')
        raw_csv['dateTime'] = pd.to_datetime(raw_csv['dateTime'], errors='coerce')
        
        plot_timeseries(raw_csv, station_name, output_path, title_suffix=" - Raw Data",
                                      save_suffix='_raw', notebook=notebook, dpi=dpi)

        # Apply initial threshold cleaning
        csv_cleaned = initial_threshold_cleaning(raw_csv, station_name, iqr_multiplier=5.0)

        # Store original values to later detect changes and apply Hampel filter
        original_values = csv_cleaned['value'].copy()
        hampel_result = hampel(original_values, window_size=250, n_sigma=10.0)
        hampel_filtered_values = hampel_result.filtered_data  # filtered data Series (outliers replaced by medians)

        # Create outlier mask using original numeric (not NaN) values
        hampel_outlier_mask = ~np.isclose(original_values, hampel_filtered_values, equal_nan=True)
        hampel_outlier_mask = hampel_outlier_mask & original_values.notna()

        # Identify residual outliers near data gaps in lower-quality time series values.
        end_of_segment_mask = identify_residual_outliers(original_values)
        
        # Create outlier mask
        non_good_mask = ~raw_csv['quality'].isin(['Good'])
        final_outlier_mask_for_plotting = (hampel_outlier_mask | end_of_segment_mask) & non_good_mask
        csv_filtered = csv_cleaned.copy()

        # Points Hampel handled AND are not 'Good' quality AND are not flagged by the residual check: replace with filtered values
        station_mask = hampel_outlier_mask & (~end_of_segment_mask) & non_good_mask
        csv_filtered.loc[station_mask, 'value'] = hampel_filtered_values.loc[station_mask]

        # Points ONLY detected by residual check AND are not 'Good' quality AND not already handled by Hampel: set to NaN
        residual_only_mask = end_of_segment_mask & (~hampel_outlier_mask) & non_good_mask
        csv_filtered.loc[residual_only_mask, 'value'] = np.nan

        # Log total number of replaced outliers
        total_replaced_outliers = (station_mask | residual_only_mask).sum()
        logger.info(f"Total {total_replaced_outliers} outliers detected and replaced by Hampel filter in '{station_name}'.")

        # Pass the filtered DataFrame and the generated outlier_mask, to highlight replaced points in plot
        plot_timeseries(csv_filtered, station_name, output_path, outlier_mask=final_outlier_mask_for_plotting,
                                    title_suffix=" - Hampel Filtered", save_suffix='_filtered', notebook=notebook,
                                    dpi=dpi)
        
        logging.info(f"Processing {station_name} complete.\n")
        
        if notebook:
            plt.show()
            
        # Store the processed DataFrame in the new dictionary
        processed_gwl_time_series_dict[station_name] = csv_filtered  
        
    # Save to avoid reprocessing every time and return modified dict
    joblib.dump(processed_gwl_time_series_dict, dict_output)
    return processed_gwl_time_series_dict

def resample_daily_average(dict: dict, start_date: str, end_date: str, path: str,
                           notebook: bool = False):
    """
    Resample the gwl data to a daily timestep for model.
    """
    logger.info(f"Initalising resampling of gwl data to daily timestep.\n")
    
    # Initialise new dict to store resampled data
    daily_dict = {}
    
    # Define global date range for model
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    full_date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Loop through remaining gwl monitoring stations
    for station_name, df in dict.items():
        logger.info(f"Resampling {station_name} to daily timestep...")
        
        # Check sorted by date and set date as index
        df = df.dropna(subset=['dateTime'])
        df = df.sort_values('dateTime')
        df = df.set_index('dateTime')
        
        # Define aggregation functions
        agg_funcs = {
            'station_name': 'first',
            'date': 'first',
            'value': 'mean',
            'quality': lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA,
            'measure': 'first'
        }
        
        # Resample and aggregate
        daily_df = df.resample('1D').agg(agg_funcs)
        daily_df = daily_df.reindex(full_date_range)  # All stations cover full time range
        daily_df = daily_df.reset_index().rename(columns={'index': 'dateTime'})
        daily_dict[station_name] = daily_df
        
        # For logging
        expected_days = (end_date - start_date).days + 1  # +1 to be inclusive
        valid_days = daily_df['value'].notna().sum()
        percent_complete = (valid_days / expected_days) * 100

        logger.info(f"    {station_name} resampled -> now contains {valid_days} non-zero data points.")
        logger.info(f"    Data covers {percent_complete:.1f}% of time period.\n")

    # Save time series aggragated plots
    for station_name, df in daily_dict.items():
        plot_timeseries(
            time_series_df=df,
            station_name=station_name,
            output_path=path,
            outlier_mask=None,
            title_suffix=" - daily timestep",
            save_suffix="_aggregated_daily",
            notebook=notebook,
            plot_outliers=False
        )
        
        # For logging
        save_path = f"{path}{station_name}_aggregated_daily.png"
        logger.info(f"{station_name} time series data in daily timestep saved to {save_path}.\n")
        
    return daily_dict
