import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
from hampel import hampel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import PchipInterpolator

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def load_timeseries_to_dict(stations_df: pd.DataFrame, col_order: list, data_dir: str,
                            inclusion_threshold: int, station_list_output: str, catchment: str):
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
    
    # Initialise list to store rows above threshold
    included_station_metadata_rows = []

    for index, row in stations_df.iterrows():
        uri = row['measure_uri']
        measure_id = uri.split("/")[-1]
        name = row['station_name'].lower().strip().replace(" ", "_")
        
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
            included_station_metadata_rows.append(row)
            logging.info(f"{name} successfully saved to dict.")
        else:
            logging.info(f"Station {name} contained insufficient data -> dropping dataframe."
                         f"({len(temp_df)} < {inclusion_threshold})")

    logging.info(f"{len(time_series_data)} stations saved to dict.\n")    
    
    # Create df from the collected rows of included stations (above threshold)
    filtered_stations_for_output = pd.DataFrame(included_station_metadata_rows)

    # Save this updated filtered df to csv
    filtered_stations_for_output.to_csv(station_list_output, index=False)
    logger.info(f"[{catchment}] Saved processed station list to: {station_list_output}")
    logger.info(f"Station location reference table head:\n\n{stations_df.head()}\n")
    logger.info(f"Total Stations: {len(stations_df)}")
      
    return time_series_data

def plot_timeseries(time_series_df: pd.DataFrame, station_name: str, output_path: str,
                    outlier_mask: pd.Series = None, title_suffix: str = "", save_suffix: str = "",
                    notebook: bool = False, plot_outliers: bool = True, dpi: int = 300,
                    legend_type: str = None):
    """
    Plot timeseries data colour coded by quality mark.
    """
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Determine the source for the x-axis data (dateTime column or index)
    use_datetime_column = 'dateTime' in time_series_df.columns
    
    if legend_type == 'quality':
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
            x_data = temp['dateTime'] if use_datetime_column else temp.index
            
            ax.plot(
                x_data,
                temp['value'],
                label=quality,
                color=color,
                alpha=0.8,
                linewidth=1.5
            )
    
    if legend_type == 'interpolation':
        # Define fixed colours for interpolation flag
        interpolation_colors = {
            'raw': '#70955F',
            'interpolated_short': '#DF6607'
        }
        
        for interp_flag, color in interpolation_colors.items():
            temp = time_series_df.copy()
            temp['value'] = temp['value'].where(temp['data_type'] == interp_flag, np.nan)
            label = "Interpolated" if interp_flag else "Original"
            x_data = temp['dateTime'] if use_datetime_column else temp.index

            ax.plot(
                x_data,
                temp['value'],
                label=label,
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
    ax.legend(title=legend_type.title(), loc="center left", bbox_to_anchor=(1.01, 0.5))

    plt.tight_layout()
    plt.savefig(f"{output_path}{station_name}{save_suffix}.png", dpi=dpi)
    
    if not notebook:
        plt.close()

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
                        f" IQR-based range and set to NaN.")
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
                                      save_suffix='_raw', notebook=notebook, dpi=dpi,
                                      legend_type='quality')

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
                                    dpi=dpi, legend_type='quality')
        
        logging.info(f"Processing {station_name} complete.\n")
        
        if notebook:
            plt.show()
            
        # Store the processed DataFrame in the new dictionary
        processed_gwl_time_series_dict[station_name] = csv_filtered  
        
    # Save to avoid reprocessing every time and return modified dict
    joblib.dump(processed_gwl_time_series_dict, dict_output)
    return processed_gwl_time_series_dict

def resample_timestep_average(gwl_data_dict: dict, start_date: str, end_date: str, path: str,
                              pred_frequency: str = 'daily', notebook: bool = False):
    """
    Resample the gwl data to the user specified timestep for model.
    """
    logger.info(f"Initalising resampling of gwl data to {pred_frequency} timestep.\n")
    
    # Initialise new dict to store resampled data
    timestep_dict = {}
    
    # Get model frequency
    frequency_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}
    clean_pred_frequency = pred_frequency.lower().strip()
    
    frequency = frequency_map.get(clean_pred_frequency)
    if frequency is None:
        raise ValueError(f"Invalid prediction frequency: {pred_frequency}. Must be 'daily', "
                         f"'weekly', or 'monthly'.")
    
    # Define global date range for model
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    full_date_range = pd.date_range(start_date, end_date, freq=frequency)
    
    # Loop through remaining gwl monitoring stations
    for original_station_name, df in gwl_data_dict.items():
        standardised_station_name = original_station_name.lower().replace(' ', '_')
        logger.info(f"Resampling {standardised_station_name} to {pred_frequency} timestep...")
        
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
        timestep_df = df.resample(frequency).agg(agg_funcs)
        
        # Ensure all stations cover full time range
        timestep_df = timestep_df.reindex(full_date_range)
        timestep_df = timestep_df.reset_index().rename(columns={'index': 'dateTime'})
        
        # Ensure station names are lowercase to avoid issues later in the pipeline
        if 'station_name' in timestep_df.columns:
            timestep_df['station_name'] = standardised_station_name
            
        # Add processed DataFrame to timestep_dict using standardised name (standardisation is key!)
        timestep_dict[standardised_station_name] = timestep_df
        
        # For logging
        expected_days = len(full_date_range)
        valid_days = timestep_df['value'].notna().sum()
        percent_complete = (valid_days / expected_days) * 100

        logger.info(f"    {standardised_station_name} resampled -> now contains "
                    f"{valid_days} non-zero data points.")
        logger.info(f"    Data covers {percent_complete:.1f}% of time period.\n")

    # Save time series aggragated plots
    for plot_station_name, plot_df in timestep_dict.items():
        plot_timeseries(
            time_series_df=plot_df,
            station_name=plot_station_name,
            output_path=path,
            outlier_mask=None,
            title_suffix=f" - {pred_frequency} timestep",
            save_suffix=f"_aggregated_{pred_frequency}",
            notebook=notebook,
            plot_outliers=False,
            legend_type='quality'
        )
        
        # For logging
        save_path = os.path.join(path, f"{plot_station_name}_aggregated_{pred_frequency}.png")
        logger.info(f"{plot_station_name} time series data in {pred_frequency} timestep saved to {save_path}.\n")
    
    return timestep_dict

def remove_spurious_data(target_df: pd.DataFrame, station_name: str, path: str, pred_frequency: str, notebook: bool = False):
    """
    Remove predefined data points identified as spurious from domain-based
    analysis.
    """
    # Initialise counter of spurious points removed
    changes_made = False
    
    # Check dateTime data type
    target_df = target_df.copy()
    target_df['dateTime'] = pd.to_datetime(target_df['dateTime'], errors='coerce')

    
    # Manual adjustment for renwick
    if station_name == 'renwick':
        start_date = '2023-01-06'
        end_date = '2023-10-23'

        # Create a boolean mask for the period to be removed
        removal_mask = (target_df['dateTime'] >= start_date) & \
                       (target_df['dateTime'] <= end_date)

        num_removed = removal_mask.sum()
        if num_removed > 0:
            target_df.loc[removal_mask, 'value'] = np.nan
            target_df.loc[removal_mask, 'quality'] = 'Missing'
            logger.info(f"Station {station_name}: Removed {num_removed} data points "
                        f"between {start_date} and {end_date}.")
            changes_made = True
    
    # Manual adjustment for ainstable
    elif station_name == 'ainstable':
        # Phase shift section
        start_date = '2021-06-16'
        end_date = '2021-11-17'
    
        # Create a boolean mask for the period to be shifted
        shift_mask = (target_df['dateTime'] >= start_date) & \
                    (target_df['dateTime'] <= end_date)

        # Apply the shift
        shift_value = 0.4654
        target_df.loc[shift_mask, 'value'] += shift_value
        logger.info(f"Station {station_name}: Shifted values by +{shift_value} "
                    f"between {start_date} and {end_date}.")
        changes_made = True
        
        # Inversion secrtion
        start_date = '2022-08-01'
        end_date = '2023-04-18'

        invert_mask = (target_df['dateTime'] >= start_date) & \
                    (target_df['dateTime'] <= end_date)
        
        # Reflect around the segment mean
        segment = target_df.loc[invert_mask, 'value']
        segment_mean = segment.mean()
        target_df.loc[invert_mask, 'value'] = segment_mean - (segment - segment_mean)
        logger.info(f"Station {station_name}: Inverted segment between "
                    f"{start_date} and {end_date} around mean = {segment_mean:.4f}")
        
        # Shift by phase
        shift_value = -0.5928
        target_df.loc[invert_mask, 'value'] += shift_value
        logger.info(f"Station {station_name}: Shifted values by +{shift_value} "
                    f"between {start_date} and {end_date}.")
        
        # Manually remove boundary points that cause issues
        boundary_dates_to_remove = [
            (pd.Timestamp('2021-06-13'), pd.Timestamp('2021-06-17')),
            (pd.Timestamp('2021-11-15'), pd.Timestamp('2021-11-19')),
            (pd.Timestamp('2023-04-16'), pd.Timestamp('2023-04-20'))
        ]
        
        for start_date, end_date in boundary_dates_to_remove:
            mask = (target_df['dateTime'] >= start_date) & (target_df['dateTime'] <= end_date)
            if mask.any(): # Check if any dates within the range exist in the DataFrame
                target_df.loc[mask, 'value'] = np.nan
                target_df.loc[mask, 'quality'] = 'Missing'
                changes_made = True
                print(f"Removed data for range: {start_date.date()} to {end_date.date()}")

    # Manual adjustment for east_brownrigg
    elif station_name == 'east_brownrigg':
        start_date = '2014-11-02'
        end_date = '2015-03-14'

        # Create a boolean mask for the period to be removed
        removal_mask = (target_df['dateTime'] >= start_date) & \
                       (target_df['dateTime'] <= end_date)

        num_removed = removal_mask.sum()
        if num_removed > 0:
            target_df.loc[removal_mask, 'value'] = np.nan
            target_df.loc[removal_mask, 'quality'] = 'Missing'
            logger.info(f"Station {station_name}: Removed {num_removed} data points "
                        f"between {start_date} and {end_date}.")
            changes_made = True

    # resave plot if values replaced  
    if changes_made:      
        plot_timeseries(
            time_series_df=target_df,
            station_name=station_name,
            output_path=path,
            outlier_mask=None,
            title_suffix=f" - {pred_frequency} timestep",
            save_suffix=f"_aggregated_{pred_frequency}",
            notebook=notebook,
            plot_outliers=False,
            legend_type='quality'
        )
        
        # For logging
        save_path = f"{path}{station_name}_aggregated_{pred_frequency}.png"
        logger.info(f"{station_name} time series data in {pred_frequency} timestep saved to {save_path}.\n")
    
    return target_df

def print_missing_gaps(df: pd.DataFrame, station_name: str, max_steps: int):
    """
    Purely for debugging and EDA.
    """
    is_nan = df['value'].isna()
    gap_ids = (is_nan != is_nan.shift()).cumsum()
    gaps = df[is_nan].groupby(gap_ids)

    # Find the total data points missing and the number of gaps
    gap_lengths = [len(group) for id, group in gaps]
    total_missing = sum(gap_lengths)
    total_gaps = len(gap_lengths)

    print(f"{station_name} contains {total_missing} missing data points across {total_gaps} gaps.\n")
    
    for i, length in enumerate(gap_lengths, start=1):
        action = "interpolate" if length <= max_steps else "do not interpolate"
        print(f"    Gap {i}: {length} data points ({action})")
        
    return total_missing

def interpolate_short_gaps(df: pd.DataFrame, station_name: str, path: str, pred_frequency: str,
                           max_steps: int = 30, notebook: bool = False):
    """
    Interpolate missing points up to threshold defined in config using polynomial
    spline interpolation (specifically PCHIP) and flag all interpolated data for model
    to identify lower reliability.
    """
    # Create duplicate to modify and initialise data status
    interpolated_df = df.copy()
    interpolated_df['data_type']='raw'
    total_interpolated = 0
    
    # Initialise variable to track max large gap (for large imputations)
    max_uninterpolated_gap_length = 0
    
    is_nan = interpolated_df['value'].isna()
    gap_ids = (is_nan != is_nan.shift()).cumsum()
    gaps = interpolated_df[is_nan].groupby(gap_ids)

    # For more verbose logging and debugging
    total_missing = print_missing_gaps(interpolated_df, station_name, max_steps)
    
    # Interpolate using PCHIP
    valid = interpolated_df['value'].notna()
    
    # Check if there are enough valid points for PCHIP interpolation
    if len(valid[valid]) < 2: # PchipInterpolator needs at least 2 non-NaN points
        logging.warning(f"Station {station_name}: Not enough valid data points for PCHIP interpolation. Skipping short gap interpolation.")
        for id, group in gaps:
            if group.iloc[0]['value'] is None: # Only process actual NaN groups
                max_uninterpolated_gap_length = max(max_uninterpolated_gap_length, len(group))
        return station_name if total_missing > 0 else None, interpolated_df, max_uninterpolated_gap_length
    
    # Call PchhipInterpolator (using the df's index directly)
    interp = PchipInterpolator(interpolated_df.loc[valid].index.astype('int64'),
                               interpolated_df.loc[valid, 'value'])

    for id, group in gaps:
        idx = group.index
        gap_start_date = group.index[0]
        gap_end_date = group.index[-1]
        
        print(f'TEST: STATION {station_name}, START: {gap_start_date} END {gap_end_date}.')

        # Skip if at start or end of full time range
        if (gap_start_date == interpolated_df.index.min()) or (gap_end_date == interpolated_df.index.max()):
            max_uninterpolated_gap_length = max(max_uninterpolated_gap_length, len(group))
            continue
        
        # Otherwise apply as expected
        if len(group) <= max_steps:
            interpolated_df.loc[idx, 'value'] = interp(interpolated_df.loc[idx].index.astype('int64'))
            interpolated_df.loc[idx, 'data_type'] = 'interpolated_short'
            total_interpolated += len(group)
        
        else:
            max_uninterpolated_gap_length = max(max_uninterpolated_gap_length, len(group))

            
    logging.info(f"{station_name}: Total interpolated points = {total_interpolated+1}\n{'-'*60}\n")
    
    # If some not interpolated add key to list for large interp function
    if total_interpolated + 1 < total_missing:
        gap = station_name
        logging.info(f"{station_name} added to list for future interpolation.")
    else:
        gap = None
        logging.info(f"{station_name}: All interpolation complete.")
    
    # resave plot if values replaced
    if total_interpolated > 0: 
        plot_timeseries(
            time_series_df=interpolated_df,
            station_name=station_name,
            output_path=path,
            outlier_mask=None,
            title_suffix=f" - {pred_frequency} timestep",
            save_suffix=f"_aggregated_{pred_frequency}",
            notebook=notebook,
            plot_outliers=False,
            legend_type='interpolation'
        )

    logger.info(f"{station_name} updated plot saved to {path}{station_name}_aggregated_{pred_frequency}.png\n")

    return gap, interpolated_df, max_uninterpolated_gap_length

def handle_short_gaps(timestep_data: dict, path: str, max_steps: int, start_date: str,
                      end_date: str, pred_frequency: str = 'daily', notebook: bool=False):
    """
    Handles short-gap imputation for groundwater level time series.

    - Interpolates small gaps (e.g., < max_steps days) using `interpolate_short_gaps`.
    - Logs and returns stations still requiring long-gap imputation.
    - Reindexes all time series to a complete timestep range for modelling.

    Returns:
        timestep_data (dict): Updated time series per station.
        gaps_list (list): Stations still needing long-gap interpolation.
        station_max_gap_lengths_calculated (dict): Max gap lengths per station.
    """
    for station_name, df_data in timestep_data.items():
        if 'dateTime' in df_data.columns:
            df_data['dateTime'] = pd.to_datetime(df_data['dateTime'], errors='coerce')
            df_data = df_data.set_index('dateTime').sort_index()
            timestep_data[station_name] = df_data # Update the dict with the indexed DataFrame

    gaps_list = []
    station_max_gap_lengths_calculated = {}

    for station_name, df in timestep_data.items():
        gap_status_for_large_interp, updated_df, max_gap_len_for_this_station = interpolate_short_gaps(
            df=df,
            station_name=station_name,
            path=path,
            max_steps=max_steps,
            pred_frequency=pred_frequency,
            notebook=notebook
        )
        
        # Update timestep_data with the processed (interpolated) DataFrame
        timestep_data[station_name] = updated_df

        if gap_status_for_large_interp: # If the station still needs large gap interp
            gaps_list.append(station_name)
            if max_gap_len_for_this_station > 0: # Only store if there was an actual large gap
                station_max_gap_lengths_calculated[station_name] = max_gap_len_for_this_station
            
    logging.info(f"Stations still needing interpolation: {gaps_list}\n")
    logging.info(f"Max uninterpolated gap lengths per station:\n{station_max_gap_lengths_calculated}\n")
    
    # Get model frequency
    frequency_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}
    clean_pred_frequency = pred_frequency.lower().strip()
    
    frequency = frequency_map.get(clean_pred_frequency)
    if frequency is None:
        raise ValueError(f"Invalid prediction frequency: {pred_frequency}. Must be 'daily', "
                         f"'weekly', or 'monthly'.")

    # Define full date range based on config
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

    for station_name, df_data in timestep_data.items():
        df_data = df_data.reindex(full_date_range)
        timestep_data[station_name] = df_data  # Update the dict with the reindexed DataFrame
            
    return timestep_data, gaps_list, station_max_gap_lengths_calculated
