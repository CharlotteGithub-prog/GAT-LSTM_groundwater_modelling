# Library Imports
import os
import sys
import copy
import joblib  # If adding parallel processing in final pipeline
import hashlib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as s

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def stable_hash(station):
    """
    Generate a consistent integer hash for a station name using MD5.
    
    Hashlib is used rather than standard hash() to be consistent across invocations. Using hash
    in the seeding over simlpe numeric seeding ensures spatial diversity with independent synthetic
    gaps for each station - mitigating the need for hardcoded gap indices when validating
    
    Return:
        int: Stable integer hash in the range [0, 9999]. Range ensured by % 10000.
    """
    station_hash = int(hashlib.md5(station.encode()).hexdigest(), 16) % 10000
    logging.info(f"Stable station hash for {station}: {station_hash}")
    
    return station_hash

def define_catchment_size(spatial_df: pd.DataFrame, name: str, threshold_m: int):
    """
    Return True is catchment width of height exceeds threshold. Catchment size will dictate distance
    calculation formulas in subsequent interpolation calculations.
    """
    logging.info(f"Checking if {name} is a large catchment...\n")
    
    min_easting = spatial_df['easting'].min()
    max_easting = spatial_df['easting'].max()
    min_northing = spatial_df['northing'].min()
    max_northing = spatial_df['northing'].max()

    easting_range_m = max_easting - min_easting
    northing_range_m = max_northing - min_northing
    logging.info(f"{name} easting_range_m: {easting_range_m}")
    logging.info(f"{name} northing_range_m: {northing_range_m}\n")
    
    logging.info(f"Large Catchment?: {easting_range_m > threshold_m or northing_range_m > threshold_m}"
                 f" (threshold: {threshold_m}m)\n")

    # Return true 
    return easting_range_m > threshold_m or northing_range_m > threshold_m

def calculate_station_distances(spatial_df: pd.DataFrame, use_haversine: bool = False,
                                radius: float = 6371000.0):
    """
    Calculate pairwise distances between stations using either:
    - Euclidean distance (for small catchments, using easting/northing), or
    - Haversine distance (for large catchments, using latitude/longitude in rads).

    Returns:
        pd.DataFrame: Square symmetric distance matrix (in m)
    """
    # Initialise distance matrix using station list
    stations = spatial_df['station_name'].values
    n = len(stations)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        row_i = spatial_df.iloc[i]  # cache for speed
        for j in range(i, n):  # only compute half as symmetric
            row_j = spatial_df.iloc[j]  # cache for speed
    
            # Caculating using greater-circle (haversine)
            if use_haversine:
                
                # Calculate latitude vals in radians
                phi_1 = np.radians(row_i['lat'])
                phi_2 = np.radians(row_j['lat'])
                delta_phi = phi_2 - phi_1
                
                # Calculate longitude vals in radians
                lambda_1 = np.radians(row_i['lon'])
                lambda_2 = np.radians(row_j['lon'])
                delta_lambda = lambda_2 - lambda_1
                
                # Calculate haversine formula
                a = np.sin(delta_phi / 2)**2 + \
                    np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                dist = radius * c

            else:
                dx = row_i['easting'] - row_j['easting']
                dy = row_i['northing'] - row_j['northing']
                
                # Calculate Euclidean formula
                dist = np.sqrt(dx**2 + dy**2)
                
            # Populate dist matrix, mirroring the values
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return pd.DataFrame(distance_matrix, index=stations, columns=stations)
                
def calculate_station_correlations(df: pd.DataFrame, catchment: str):
    # Extract just the 'value' series from each station's DataFrame and store in a list
    series_list = []
    for station_name, station_df in df.items():
        # Rename the series to the station_name for proper column naming after concat
        if 'value' in station_df.columns:
            series_list.append(station_df['value'].rename(station_name))
            
    gwl_data_for_correlation = pd.concat(series_list, axis=1)
    gwl_data_for_correlation = gwl_data_for_correlation.sort_index()

    correlation_matrix = gwl_data_for_correlation.corr(method='pearson')
    min_common_observations = 365 # Example: Require at least 1 year of overlapping data
    
    def count_non_nan_overlap(series1, series2):
        return pd.concat([series1, series2], axis=1).dropna().shape[0]

    filtered_correlation_matrix = correlation_matrix.copy()

    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 == col2:
                continue
            
            # Retrieve the full series from gwl_data_for_correlation for overlap count
            overlap_count = count_non_nan_overlap(gwl_data_for_correlation[col1], gwl_data_for_correlation[col2])
            
            if overlap_count < min_common_observations:
                filtered_correlation_matrix.loc[col1, col2] = np.nan
                filtered_correlation_matrix.loc[col2, col1] = np.nan
    
    # Handle negative correlations (not useful here) -> Set all values < 0 to 0
    processed_correlation_matrix = filtered_correlation_matrix.mask(filtered_correlation_matrix < 0, 0)

    logging.info(f"{catchment}: Correlation matrix calculated with a minimum of {min_common_observations} overlapping observations.\n")    
    return processed_correlation_matrix

def plot_score_scatters(scores_for_gappy_station: pd.Series, distance_matrix: pd.DataFrame,
                        gappy_station: str, output_path: str, k_decay: float):
    # Get data points for plotting (filter out NaN scores to avoid plotting issues)
    plot_data = pd.Series(scores_for_gappy_station).dropna() 
    
    # Plot scatters when data is available
    if not plot_data.empty:
        plot_distances_km = [distance_matrix.loc[gappy_station, donor_station] / 1000 for
                             donor_station in plot_data.index]
        plot_scores = plot_data.values

        # Set up plt figure
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_distances_km, plot_scores, c='blue', alpha=0.7)
        plt.title(f'Influence Score vs. Distance for {gappy_station} (k={k_decay})')
        plt.xlabel('Distance (km)')
        plt.ylabel('Influence Score')
        plt.grid(True)
        
        # Save the plot to results and close plt before continuing score loop
        plot_filename = f"{output_path}{gappy_station}_influence_score_plot.png"
        plt.savefig(plot_filename, dpi=300)
        logging.info(f"Influence score plot saved to: {plot_filename}\n")
        plt.close()

    else:
        logging.info(f"No valid scores to plot for {gappy_station}.\n")

def filter_stations_to_threshold(station_scores: dict, threshold: float = 0.3):
    """
    Filter station influence score dictionaries to only include donor stations that have a score
    above the predefined threshold.
    """
    # Iitialise filtered dict
    filtered_station_scores = {}
    
    # Filter to only include scores above threshold
    for gappy_station, donor_scores in station_scores.items():
        filtered_donors = {donor: score for donor, score in donor_scores.items() if score >= threshold}
        
        if filtered_donors: # Only add donors to filtered_station_scores if there is a donor above threshold
            filtered_station_scores[gappy_station] = filtered_donors
        else:
            logging.info(f"No donor stations found above threshold {threshold} for {gappy_station}.")
    
    return filtered_station_scores

def score_station_proximity(df_dist: pd.DataFrame, gaps_list: list, correlation_matrix: pd.DataFrame,
                            distance_matrix: pd.DataFrame, k_decay: float, output_path: str,
                            threshold: float):
    """
    Identified options for formula: s=ce^{-kd} or s=\frac{c}{d+1}. Trying exp decay first.
    """
    station_scores = {}
    
    # Check stations requiring imputation are indexes in scoring reference df's
    for gappy_station in gaps_list:
        if gappy_station not in correlation_matrix.index or gappy_station not in distance_matrix.index:
            logging.info(f"Warning: {gappy_station} station not in scoring matrices.")
            continue
        
        logging.info(f"Calculating scores for gappy station: {gappy_station}")
   
        # Get correlation and distance data for imputation station
        corr_series = correlation_matrix.loc[gappy_station]
        dist_series = distance_matrix.loc[gappy_station]  # note that dist here is in m, formula needs km
        
        # Initialise station specific score dict
        scores_for_gappy_station = {}
        
        # Loop through all other stations to calculate scores
        for other_station in correlation_matrix.columns:
            if other_station == gappy_station:
                continue  # Skip scoring the station against itself
        
            if other_station in corr_series.index and other_station in dist_series.index:
                c = corr_series.loc[other_station]
                d = dist_series.loc[other_station] / 1000  # Convert to km
                k = k_decay
                
                # Handle NaN values - If corr or dist is NaN, the score should be 0
                if pd.isna(c) or pd.isna(d) or c == 0 or d == 0:
                    score = 0
                else:
                    score = c * np.exp(-k * d)
                    
                scores_for_gappy_station[other_station] = score
            
            else:
                logging.warning(f"Station {other_station} not found in series for {gappy_station}. Skipping score calculation.")
                
        # Store the calculated scores for the current gappy station
        station_scores[gappy_station] = scores_for_gappy_station
        
        # Produce station-specfic scatter plots to help define inclusion threshold for imputation influence
        plot_score_scatters(scores_for_gappy_station, distance_matrix, gappy_station, output_path, k_decay)
    
    # Adjust station_scores to remove stations below threshold
    filtered_station_scores = filter_stations_to_threshold(station_scores, threshold)
    
    # Log filtered scores for debugging
    for gappy_station_with_donors, donors_above_threshold in filtered_station_scores.items():
        logging.info(f"Filtered scores for {gappy_station_with_donors} (above threshold {threshold}):\n"
                     f"{pd.Series(donors_above_threshold).sort_values(ascending=False).round(4)}")
        logging.info("-" * 25)
    
    return filtered_station_scores

def calculate_donor_offsets(gappy_station_name: str, gappy_station_df: pd.DataFrame,
                            donor_names: list, df_dict_original: dict):
    """
    Calculates the average difference (offset) between the gappy station's values
    and each donor station's values during their common (non-NaN) observation periods.
    """
    offsets = {}
    gappy_series = gappy_station_df['value'].dropna() # Only use observed values not NaN's

    # Skip if donor data not available
    for donor_name in donor_names:
        if donor_name not in df_dict_original:
            continue

        donor_series = df_dict_original[donor_name]['value'].dropna()

        # Find common observation periods for both series
        common_index = gappy_series.index.intersection(donor_series.index)
       
        # Calculate the difference for common observed periods to define y-axis shift bias
        if len(common_index) > 0:
            diff_series = gappy_series.loc[common_index] - donor_series.loc[common_index]
            offsets[donor_name] = diff_series.mean()
        else:
            logging.warning(f"No common observation periods between {gappy_station_name} "
                            f"and {donor_name} to calculate offset.")
            offsets[donor_name] = 0.0 # Default to no offset (although this may not be rigorous)
            
    return offsets

def weighted_imputation(nan_timestamps: pd.DatetimeIndex, imputation_method: str, sorted_donors: pd.Series,
                        gappy_station_df: pd.DataFrame, donor_data_for_gaps: pd.DataFrame,
                        gappy_station_name: str, donor_offsets: dict = None,
                        imputed_value_flag: str = 'Imputed_LargeGap'):
    """
    Performs imputation for a single gappy station across its NaN timestamps
    using either single best or weighted average (donor score) method.
    Includes boundary shift and a new smoother endpoint adjustment,
    with specific handling for gaps at the start or end of the data.
    Incorporates a fallback to seasonal mean if donor-based imputation fails.
    """
    # Initialise donor offsets as dict even if None (to expected form)
    if donor_offsets is None:
        donor_offsets = {}
    
    # Ensure 'data_type' column exists for imputation flagging
    if 'data_type' not in gappy_station_df.columns:
        gappy_station_df['data_type'] = 'raw'

    # Use all filtered donors (current approach - could limit to top X in future)
    donors_to_use_series = sorted_donors 
    relevant_donor_values = donor_data_for_gaps[donors_to_use_series.index]

    # Pre-calculate seasonal means for fallback if needed
    # Use only non-missing data from the gappy station for this calculation to avoid bias from existing NaNs
    gappy_station_non_na_values = gappy_station_df['value'].dropna()
    seasonal_means = pd.Series(dtype=float) # Initialise as empty Series to ensure it's always defined
    if not gappy_station_non_na_values.empty:
        # Calculate mean for each day of the year (1-366)
        seasonal_means = gappy_station_non_na_values.groupby(gappy_station_non_na_values.index.dayofyear).mean()
        logging.info(f"Calculated seasonal means for {gappy_station_name} from {len(gappy_station_non_na_values)} non-NaN historical points.")
    else:
        logging.warning(f"No non-NaN data found for {gappy_station_name} to calculate seasonal means. Seasonal mean fallback will not be effective.")


    # Go timestep by timestep within the gaps and perform weighted imputation
    total_imputed_values = 0
    
    # --- 1. Initial imputation with donor data weighting and offset (with seasonal mean fallback) ---
    
    for timestamp in nan_timestamps:
        imputed_value = np.nan # Set to avoid error if issue with imputation
        if imputation_method == 'weighted_average':  # Test other options (e.g. highest score only)
            # Initialise weighting trackers
            weighted_sum = 0.0
            total_weight = 0.0
            
            # Loop through donors and calculate total weighting
            for donor_name, score in donors_to_use_series.items():
                donor_value = relevant_donor_values.loc[timestamp, donor_name]
                offset = donor_offsets.get(donor_name, 0.0) # Get offset for donor, default to 0 if none

                # Skip NaN values as will not support imputation
                if not pd.isna(donor_value):
                    adjusted_donor_value = donor_value + offset
                    weighted_sum += adjusted_donor_value * score
                    total_weight += score

            # Calculate weighted imputed value from donors
            if total_weight > 0:
                imputed_value = weighted_sum / total_weight
                # gappy_station_df['data_type'] = 'imputed_long'
        
        # Fallback to seasonal mean if donor-based imputation resulted in NaN
        if pd.isna(imputed_value):
            day_of_year = timestamp.dayofyear
            # Ensure seasonal_means is not empty and contains the specific day_of_year
            if not seasonal_means.empty and day_of_year in seasonal_means: 
                imputed_value = seasonal_means.loc[day_of_year]
                gappy_station_df.loc[timestamp, 'data_type'] = 'imputed_long'
            #     logging.info(f"Could not impute {gappy_station_name} at {timestamp} using donors. Falling back to seasonal mean ({imputed_value:.2f}).")
            # else:
            #     logging.info(f"Could not impute {gappy_station_name} at {timestamp} using donors or seasonal mean (not available/empty). Gap remains.")

        # Apply the imputed value and flag it (critical for model performance) in the current station's df
        if not pd.isna(imputed_value):
            gappy_station_df.loc[timestamp, 'value'] = imputed_value
            gappy_station_df.loc[timestamp, 'data_type'] = 'imputed_long'  # Flag data as imputed
            total_imputed_values += 1
        
        # Make it clear if imputation was not successful for data points  
        else:
            logging.info(f"Could not impute {gappy_station_name} at {timestamp}. Gap remains after all imputation attempts.\n")

    # Reevaluate using boundary shift and smooth endpoint adjustment
    gap_segments_to_shift = get_consecutive_gap_segments(nan_timestamps)
    logging.info(f"Identified {len(gap_segments_to_shift)} segments for boundary handling and adjustment for {gappy_station_name}.")

    for segment_timestamps in gap_segments_to_shift:
        if segment_timestamps.empty:
            logging.info(f"Skipping empty segment for {gappy_station_name}.")
            continue

        gap_start_date = segment_timestamps.min()
        gap_end_date = segment_timestamps.max()
        logging.info(f"Processing gap segment from {gap_start_date.date()} to {gap_end_date.date()} ({len(segment_timestamps)} days) for {gappy_station_name}.")

        # Find the last observed value before this gap
        pre_gap_index = gappy_station_df.index < gap_start_date
        value_before_gap = np.nan
        if pre_gap_index.any():
            last_observed_series = gappy_station_df.loc[pre_gap_index, 'value'].dropna()
            value_before_gap = last_observed_series.iloc[-1] if not last_observed_series.empty else np.nan
        else:
            value_before_gap = np.nan

        # Find the first observed value after this gap
        post_gap_index = gappy_station_df.index > gap_end_date
        value_after_gap = np.nan
        if post_gap_index.any():
            first_observed_series = gappy_station_df.loc[post_gap_index, 'value'].dropna()
            value_after_gap = first_observed_series.iloc[0] if not first_observed_series.empty else np.nan
        else:
            value_after_gap = np.nan

        # Get the currently imputed segment values (from initial imputation pass)
        imputed_segment_values_current = gappy_station_df.loc[segment_timestamps, 'value']

        # Ensure there are valid imputed values in the segment to work with
        if imputed_segment_values_current.empty or imputed_segment_values_current.isnull().all():
            continue
            
        # Handle different types of gaps based on boundary availability (3 potential cases)

        # Case 1 => Gap at the very start of the data (left-sided open gap)
        if (gap_start_date == gappy_station_df.index.min()) and not pd.isna(value_after_gap):
            
            # Check if seasonal means are available and not empty
            if not seasonal_means.empty:
                # Get the day of year for the end of the gap
                day_of_year_at_end = gap_end_date.dayofyear

                if day_of_year_at_end in seasonal_means:
                    seasonal_value_at_end = seasonal_means.loc[day_of_year_at_end]
                    
                    # Calculate the offset needed to match value_after_gap to its seasonal expectation
                    offset_from_seasonal = value_after_gap - seasonal_value_at_end

                    adjusted_values = []
                    for ts in segment_timestamps:
                        day_of_year_ts = ts.dayofyear
                        if day_of_year_ts in seasonal_means:
                            adjusted_values.append(seasonal_means.loc[day_of_year_ts] + offset_from_seasonal)
                        else:
                            # If a seasonal mean for a specific day_of_year in the gap is missing, fall back to initially imputed value
                            logging.warning(f"    Seasonal mean not available for day {day_of_year_ts} in gap {ts.date()} for {gappy_station_name}. Retaining initial imputed value for this point.")
                            adjusted_values.append(gappy_station_df.loc[ts, 'value']) # Revert to prior imputation
                    
                    if adjusted_values:
                        gappy_station_df.loc[segment_timestamps, 'value'] = adjusted_values
                        gappy_station_df.loc[segment_timestamps, 'data_type'] = 'imputed_long'
            #         else:
            #             logging.warning(f"  No adjusted values generated for seasonal extrapolation for {gappy_station_name}. Segment retains initial imputed values.")

            #     else:
            #         # The segment will retain the values from the initial imputation (weighted average/seasonal mean fallback).
            #         logging.warning(f"  Seasonal mean for day {day_of_year_at_end} not available to anchor seasonal extrapolation for {gappy_station_name}. Segment retains initial imputed values.")

            # else:
            #     # The segment will retain the values from the initial imputation (weighted average/seasonal mean fallback).
            #     logging.warning(f"  Seasonal means are empty for {gappy_station_name}. Cannot perform seasonal extrapolation for start-of-data gap. Segment retains initial imputed values.")


        # Case 2 => Gap at the very end of the data (right-sided open gap)
        elif not pd.isna(value_before_gap) and (gap_end_date == gappy_station_df.index.max()):
            logging.info(f"  Handling end-of-data gap (last segment in DataFrame) for {gappy_station_name} (from {gap_start_date.date()} to {gap_end_date.date()}).")
            
            # Apply initial boundary shift to match value_before_gap.
            shift_amount = value_before_gap - imputed_segment_values_current.iloc[0]
            gappy_station_df.loc[segment_timestamps, 'value'] += shift_amount
            imputed_segment_values_current = gappy_station_df.loc[segment_timestamps, 'value'] # Update after shift

        # Case 3 => Gap in the middle of data (two-sided closed gap)
        else: 
                        
            # Apply Boundary Shift
            shift_amount = 0.0
            initial_imputed_value = imputed_segment_values_current.iloc[0]
            
            # Only shift if both boundary and initial imputed value are valid
            if not pd.isna(value_before_gap) and not pd.isna(initial_imputed_value):
                shift_amount = value_before_gap - initial_imputed_value
                gappy_station_df.loc[segment_timestamps, 'value'] += shift_amount
                
                # Update imputed_segment_values_current after the shift for scaling calculation
                imputed_segment_values_current = gappy_station_df.loc[segment_timestamps, 'value']

            # Smooth Endpoint Adjustment to ensure the end of the imputed segment smoothly transitions to value_after_gap.
            if not pd.isna(value_after_gap) and len(imputed_segment_values_current) > 0: 
                
                # Calculate the discrepancy at the end of the gap
                discrepancy_at_end = value_after_gap - imputed_segment_values_current.iloc[-1]

                # Apply a linear ramp to distribute this discrepancy across the segment
                num_points = len(imputed_segment_values_current)
                if num_points > 1:
                    adjustment_factors = np.linspace(0, 1, num_points)  # Create a linear ramp from 0 to 1 over the segment length
                    adjusted_values = imputed_segment_values_current.values + (adjustment_factors * discrepancy_at_end)
                else: # For a single point segment, just apply the full discrepancy
                    adjusted_values = imputed_segment_values_current.values + discrepancy_at_end

                gappy_station_df.loc[segment_timestamps, 'value'] = adjusted_values
    
    # For Logging
    total_gaps = len(nan_timestamps)
    successfully_imputed_count = gappy_station_df.loc[nan_timestamps, 'data_type'].eq('imputed_long').sum() # Count non-NaN (i.e. flagged) imputation methods
    percentage_imputed = (successfully_imputed_count / total_gaps) * 100
    
    logging.info(f"    -> Imputation Complete for {gappy_station_name}.")
    logging.info(f"    -> {total_imputed_values} / {total_gaps} successfully imputed ({percentage_imputed:.2f}%).\n")

def get_consecutive_gap_segments(nan_timestamps: pd.DatetimeIndex, max_imputation_length_threshold: int = None):
    """
    Groups consecutive NaN timestamps into blocks and returns a list of DatetimeIndex segments.
    Optionally filters segments by max_imputation_length_threshold.
    """
    if nan_timestamps.empty:
        logging.info("Input nan_timestamps is empty. Returning empty list.")
        return []

    # Sort and calculate differences between consecutive timestamps (in days)
    nan_timestamps = nan_timestamps.sort_values()
    time_diffs = pd.Series(nan_timestamps).diff().dt.days
    
    # Identify the start of each new gap block - True if start of sequence or first element
    gap_start_flags = (time_diffs != 1) | (pd.isna(time_diffs))
    gap_start_integer_indices = np.where(gap_start_flags)[0]
    
    logging.info(f"gap_start_flags generated. Number of gap starts identified: {len(gap_start_integer_indices)}")

    segments = []
    # current_segment_start_idx = 0

    for i in range(len(gap_start_integer_indices)):
        start_idx_of_segment = gap_start_integer_indices[i]
        
        # Determine end index for current segment
        if i < len(gap_start_integer_indices) - 1:
            end_idx_of_segment = gap_start_integer_indices[i+1]
        
        # If it's the last start index, this segment goes to the end of nan_timestamps.
        else: 
            end_idx_of_segment = len(nan_timestamps)
            
        # Slice the original nan_timestamps using the integer indices to get the DatetimeIndex segment
        segment = nan_timestamps[start_idx_of_segment:end_idx_of_segment]
        logging.info(f"  Segment {i+1}: Start index {start_idx_of_segment}, End index {end_idx_of_segment}, Length {len(segment)} days.")

        # Apply (somewhat optional) length threshold filter
        if max_imputation_length_threshold is None or len(segment) <= max_imputation_length_threshold:
            segments.append(segment)
        else:
            # Note: This will only show if logging level is DEBUG or lower.
            logging.debug(f"  Skipping segment from {segment.min().date()} to {segment.max().date()} "
                          f"({len(segment)} days) due to exceeding max_imputation_length_threshold ({max_imputation_length_threshold}).")
    
    return segments

def group_consecutive_gaps(nan_timestamps: pd.DatetimeIndex, max_imputation_length_threshold: int):
    """
    Groups consecutive NaN timestamps into blocks and filters them by max_imputation_length_threshold.
    Returns a DatetimeIndex of timestamps that should be imputed.
    """
    # Ensure timestamps are sorted
    nan_timestamps = nan_timestamps.sort_values()

    # Find where the consecutive sequence have a difference greater that 1 day (non-consecutive)
    time_diffs = pd.Series(nan_timestamps).diff().dt.days
    gap_starts_indices = time_diffs[time_diffs != 1].index.tolist()  # Identify gap starts

    # Ensure very first NaN timestamp is included as a gap start if it's the first in the series.
    if not nan_timestamps.empty and (not gap_starts_indices or nan_timestamps[0] < nan_timestamps[gap_starts_indices[0]]):
        gap_starts_indices.insert(0, nan_timestamps.index[0]) # nan_timestamps.index gets positional index
    
    # Convert positional indices to actual timestamps and filter nan_timestamps by these start times
    gap_starts = nan_timestamps[nan_timestamps.isin(nan_timestamps[gap_starts_indices])]
    timestamps_to_impute = []

    # Loop through identified gap start times
    for i in range(len(gap_starts)):
        start_of_current_gap = gap_starts[i]
        
        # Calculate the end of the current gap
        if i + 1 < len(gap_starts):
            next_gap_start = gap_starts[i+1]
            current_gap_segment = nan_timestamps[(nan_timestamps >= start_of_current_gap) & (nan_timestamps < next_gap_start)]
        else:
            current_gap_segment = nan_timestamps[nan_timestamps >= start_of_current_gap]

        # Calculate the length of the current gap segment (total number of NaN days in gap)
        gap_length = len(current_gap_segment) 

        # If a valid (not empty) gap segment then compare gap length to threshold length
        if gap_length > 0:
            # If below threshold add gap timestamps to imputable list, else skip it
            if gap_length <= max_imputation_length_threshold:
                timestamps_to_impute.extend(current_gap_segment.tolist())
                logging.info(f"  Gap from {current_gap_segment.min().date()} to {current_gap_segment.max().date()} "
                              f"({gap_length} days) will be imputed.")
            else:
                logging.info(f"  Skipping imputation for gap from {current_gap_segment.min().date()} to "
                             f"{current_gap_segment.max().date()} ({gap_length} days) as it exceeds "
                             f"threshold of {max_imputation_length_threshold} days.")
    
    return pd.DatetimeIndex(timestamps_to_impute).sort_values()  # Return the sorted imputable timestamps

def synthetic_gap_allocation(gap_lengths_to_test: list, min_around: int, valid_synthetic_indices: pd.Series,
                             gappy_station: str, current_gappy_station_df: pd.DataFrame, df_dict_original: dict,
                             df_for_validation: dict, synthetic_gap_flags: dict, seed: int=None):
    """
    Create synthetic gaps based on dynamic gap requirements by station. Keep a tracker of these
    synthetic timestamps to easily revert later.
    """
    # Using hash to get independent synthetic gaps for each station (for spatial diversity)
    if seed is not None:
        np.random.seed(seed + stable_hash(gappy_station))
    
    # Initialise original_values_masked here (as accumulated across all synthetic gaps) and gap counter
    original_values_masked = pd.Series(dtype=float)
    gaps_created_count = 0
    
    # Loop through different synthetic gap lengths needed
    for gap_length_days in gap_lengths_to_test:
        
        # Track gap creation attempts to avoid infinite loop if impossible
        attempt_count = 0
        max_attempts = 100
        
        # Attempt up to max_attempts iterations
        while attempt_count < max_attempts:
            
            # Calculate required gap size
            required_points = gap_length_days + 2 * min_around
            
            # Immediately break is synthetic gap size exceeds valid (non NaN) points available
            if len(valid_synthetic_indices) <= required_points:
                logging.debug(f"Not enough valid indices for {gappy_station} to create a gap of {gap_length_days} days"
                                f" respecting min_data_points_around_gap. Breaking attempts for this gap length.")
                break
            
            # Otherwise, randomly initialise start position and calc synthetic gap start and end date
            start_index_pos = np.random.randint(min_around, len(valid_synthetic_indices) - required_points + min_around)
            gap_start_date = valid_synthetic_indices[start_index_pos]
            gap_end_date = gap_start_date + pd.Timedelta(days=gap_length_days - 1)

            # Ensure the proposed gap is entirely within the existing data index, otherwise start next loop
            if gap_start_date not in current_gappy_station_df.index or gap_end_date not in current_gappy_station_df.index:
                attempt_count += 1
                continue

            # Check if the proposed gap overlaps with any existing NaNs in the *original* data, otherwise start next loop
            original_data_in_gap_check = df_dict_original[gappy_station]['value'].loc[gap_start_date:gap_end_date]
            if original_data_in_gap_check.isna().any():
                attempt_count += 1
                continue
            
            # Store original values from df_dict before masking
            temp_original_values = df_dict_original[gappy_station]['value'].loc[gap_start_date:gap_end_date].copy()
            
            # Filter and concatenate original_values_masked (filters are necessary to avoid FutureWarnings)
            series_to_concat = []
            if not original_values_masked.empty:
                series_to_concat.append(original_values_masked)
                
            # Check if the extracted Series for the current gap is empty before appending it to avpoid warnings
            if not temp_original_values.empty: 
                series_to_concat.append(temp_original_values)

            # Only concatenate if there's something to concatenate
            if series_to_concat:
                original_values_masked = pd.concat(series_to_concat)
            else:
                original_values_masked = pd.Series(dtype=float)
            
            # Mask the data in the copy to perform validation
            df_for_validation[gappy_station].loc[gap_start_date:gap_end_date, 'value'] = np.nan
            gaps_created_count += 1
            
            # Track location of synthetic gaps
            gap_range = pd.date_range(start=gap_start_date, end=gap_end_date)
            synthetic_gap_flags[gappy_station].update(gap_range)
            
            logging.info(f"Created synthetic gap ({gaps_created_count}/{len(gap_lengths_to_test)}) of {gap_length_days} "
                         f"days for {gappy_station} from {gap_start_date.date()} to {gap_end_date.date()}.")
            break  # successfully made gap, move to next gap length

        if attempt_count == max_attempts:
            logging.warning(f"Failed to create synthetic gap for {gappy_station} of length {gap_length_days}.")
    
    if gaps_created_count == 0:
        logging.info(f"No synthetic gaps were successfully created for {gappy_station}.")

    logging.info('-' * 25)
    return original_values_masked

def find_validation_gap_lengths(max_len_station_actual: int, max_imputation_length_threshold: int,
                                gappy_station: str, predefined_large_gap_lengths: list,
                                valid_synthetic_indices: pd.Series, min_around: int):
    # Work out gap lengths to test (validation) for each station based on remaining gaps
    gap_lengths_to_test = []
    
    # If the max actual gap is too long for this function (e.g. BGS EV2), test up to the max_imputation_length_threshold
    if max_len_station_actual > max_imputation_length_threshold:
        logging.info(f"Station {gappy_station} has a max actual gap ({max_len_station_actual} days) > max imputation threshold"
                        f" ({max_imputation_length_threshold} days). Testing up to {max_imputation_length_threshold} days.")
        gap_lengths_to_test = [gap for gap in predefined_large_gap_lengths if gap <= max_imputation_length_threshold]
    
    # Otherwise, for stations with max actual gaps within the imputation range [30-180], test up to its max actual gap
    else:
        logging.info(f"Station {gappy_station} has a max actual gap of {max_len_station_actual} days. Testing relevant"
                     f" predefined large gaps.")
        
        # Append one time step past max and stop looping once exceeded
        for gap_len in predefined_large_gap_lengths:
            if gap_len <= max_len_station_actual + 30:
                gap_lengths_to_test.append(gap_len)
            elif gap_len > max_len_station_actual + 30:
                break

    # Final check for sufficient valid data points after determining gap_lengths_to_test, skip station if insufficient for testing
    if not gap_lengths_to_test or len(valid_synthetic_indices) < max(gap_lengths_to_test) + 2 * min_around:
        logging.warning(f"Not enough valid data points for {gappy_station} to create synthetic gaps of required lengths"
                        f" ({gap_lengths_to_test}). Skipping synthetic validation for this station.")
        return []
    
    return gap_lengths_to_test

def plot_imputation_validation_results(df_dict_original: dict, gappy_station: str, rmse: float, mae: float,
                                       imputed_df_dict_synthetic_run: dict, original_values_masked: pd.Series,
                                       imputed_values_at_synthetic_gaps: pd.Series, output_path:str):
    # Setup up plotting figure
    fig, ax = plt.subplots(figsize=(15, 4))
    
    ax.plot(imputed_df_dict_synthetic_run[gappy_station].index, imputed_df_dict_synthetic_run[gappy_station]['value'],
                label='Imputed Data (Validation Run)', color='#ff7f0e', alpha=1)
    
    ax.plot(df_dict_original[gappy_station].index, df_dict_original[gappy_station]['value'],
                label='Original Data (incl. real NaNs)', color='#1f77b4', alpha=1)

    # Plot the actual original points that were masked for synthetic gaps
    ax.plot(original_values_masked.index, original_values_masked.values,
                'o', label='Original (Synthetic Gap)', color='#d62728', alpha=0.5, markersize=2)
    
    # Plot the imputed points at the synthetic gap locations
    ax.plot(imputed_values_at_synthetic_gaps.index, imputed_values_at_synthetic_gaps.values,
                'x', label='Imputed (Synthetic Gap)', color='#2ca02c', alpha=0.5, markersize=2)
    
    ax.set_title(f'Synthetic Gap Imputation for {gappy_station}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Groundwater Level')
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True)
    
    # Save plot
    fig.tight_layout()
    plot_filename = f"{output_path}{gappy_station}_synthetic_imputation_plot.png"
    plt.savefig(plot_filename, dpi=300)
    logging.info(f"Synthetic imputation plot saved to: {plot_filename}\n")
    
    plt.close()

def _kge(y_true, y_pred):
    """
    Kling-Gupta Efficiency (Gupta et al., 2009).
    Returns NaN if correlation or variance terms are degenerate.
    """
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or y_pred.size < 2:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        return np.nan

    mu_o, mu_s = np.mean(y_true), np.mean(y_pred)
    sd_o, sd_s = np.std(y_true, ddof=1), np.std(y_pred, ddof=1)
    if sd_o == 0 or mu_o == 0:
        return np.nan

    alpha = sd_s / sd_o
    beta  = mu_s / mu_o
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

def calc_validation_performance_metrics(
    original_values_masked: pd.Series,
    imputed_df_dict_synthetic_run: dict,
    gappy_station: str,
    validation_results: dict
):
    """
    Calculates RMSE, MAE, KGE for imputed synthetic gaps and updates validation_results.
    Returns (rmse, mae, kge, imputed_values_at_synthetic_gaps).

    This version:
      - aligns by timestamp once,
      - computes RMSE and MAE independently,
      - uses a NaN-safe KGE,
      - avoids any accidental reuse of intermediate variables.
    """
    import numpy as np
    import pandas as pd

    # Ensure unique, sorted index for the truth series
    truth = original_values_masked.copy()
    truth = truth[~truth.index.duplicated(keep="first")].sort_index()

    # Pull imputed series for the station and align to truth timestamps
    imputed_series = imputed_df_dict_synthetic_run[gappy_station]['value']
    imputed_at_gaps = imputed_series.reindex(truth.index)

    # Build aligned frame and drop rows with any NaNs
    aligned = pd.DataFrame(
        {"original": truth, "imputed": imputed_at_gaps}
    ).dropna()

    if aligned.empty:
        # Nothing to score
        return np.nan, np.nan, np.nan, imputed_at_gaps

    # Convert to numpy arrays to avoid any pandas alignment surprises
    y_true = aligned["original"].to_numpy(dtype=float)
    y_pred = aligned["imputed"].to_numpy(dtype=float)

    # --- Metrics (computed independently) ---
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    kge  = float(_kge(y_true, y_pred))

    # Record
    validation_results[gappy_station] = {"rmse": rmse, "mae": mae, "kge": kge}

    # Return metrics and the imputed values at the synthetic gap timestamps (for plotting)
    return rmse, mae, kge, imputed_at_gaps

def correct_suspect_variance_scaleby(df_dict: dict, pred_frequency: str):
    """
    Use seasonal imputation for end of year portion of scaleby dataset to reduce excess
    donor variance.
    """
    df = df_dict['scaleby']
    gap_start = pd.to_datetime('2024-11-07')
    gap_end = pd.to_datetime('2025-01-21')

    # Build seasonal climatology from previous years
    ref_start = gap_start - pd.DateOffset(years=6)
    ref_end = gap_start - pd.DateOffset(days=1)
    ref_data = df.loc[ref_start:ref_end].copy()
    ref_data['doy'] = ref_data.index.dayofyear
    
    climate_data = ref_data.groupby('doy')['value'].mean()

    # Impute the gap using matching day-of-year values
    gap_mask = (df.index >= gap_start) & (df.index <= gap_end)
    gap_dates = df.loc[gap_mask].copy()
    gap_dates['doy'] = gap_dates.index.dayofyear

    # Replace in original dataframe
    df.loc[gap_mask, 'value'] = gap_dates['doy'].map(climate_data)
    
    # Apply correction only if matched resolution
    if pred_frequency.lower().strip() == 'daily':
        df.loc['2024-12-31', 'value'] = min(df['value'].mean(), 41.0)
    
    # gap_mask = df['masked'] & (df['data_type'] == 'gap_long')
    df.loc[gap_mask, 'data_type'] = 'imputed_long'
    # df.loc[gap_mask, 'masked'] = False

    df_dict['scaleby'] = df
    print(f"Applied seasonal imputation for scaleby from {gap_start.date()} to {gap_end.date()}")
    
    return df_dict

def correct_suspect_variance_renwick(df_dict: dict):
    """
    Use seasonal imputation for masked portion of renwick dataset to increase masking
    width to reduce unreliable range.
    """
    df = df_dict['renwick']
    gap_start = pd.to_datetime('2023-10-23')
    gap_end = pd.to_datetime('2023-11-08')
    gap_mask = (df.index >= gap_start) & (df.index <= gap_end)

    df.loc[gap_mask, 'value'] = np.nan
    df.loc[gap_mask, 'data_type'] = 'masked'  # Later logic should also handle this
    df.loc[gap_mask, 'masked'] = True
    
    return df_dict

def clean_df_dict(df_dict, df_dict_original, synthetic_gap_flags):
    """
    tbd
    """
    # Clean dataframe of synthetic and validation data
    for station, synthetic_dates in synthetic_gap_flags.items():
        if station not in df_dict:
            logging.warning(f"Warning: Station {station} not in df_dict.")
            continue
        revert_count = 0
        for dt in synthetic_dates:
            if (
                dt in df_dict[station].index
                and not pd.isna(df_dict_original[station].loc[dt, 'value'])  # Only revert if original was real data
                and df_dict_original[station].loc[dt, 'value'] != 0
            ):
                df_dict[station].loc[dt, 'value'] = df_dict_original[station].loc[dt, 'value']
                df_dict[station].loc[dt, 'data_type'] = 'raw'
                revert_count += 1

    # # Trim data frame by station to bounds of model
    # for station, df in df_dict.items():
    #     trimmed_df[station] = df.loc[start_date:end_date].copy()
        
    # Set NaN to mean
    for station, df in df_dict.items():
        df['masked'] = df['value'].isna()  # Create boolean masked col
        df.loc[df['value'].isna(), 'data_type'] = 'masked'  # Set data_type to masked
        
        # Set NaN vlaues to mean for input into GAT model
        station_mean = df['value'].mean()
        # df['value'].fillna(station_mean, inplace=True)  # TODO: SOLVE
        df['value'] = df['value'].fillna(df['value'].mean())
    
    logging.info(f"All stations cleaned and complete with NaN filled to mean.\n")
                
    return df_dict

def mask_data_gaps(df_dict: dict):
    """
    Set points equal to ts mean and make binary 'masked' column as well as setting data_type to 'masked'.
    """
    for station, station_df in df_dict.items():
        salary_nan_count = station_df['value'].isna().sum()
        print(f"{station}: {salary_nan_count} NaNs")

def handle_large_gaps(df_dict: pd.DataFrame, gaps_list: list, catchment: str, spatial_path: str, path: str,
                      threshold_m: int, radius: int, output_path: str, threshold: float, predefined_large_gap_lengths: list,
                      max_imputation_length_threshold: int, min_around: int, station_max_gap_lengths: dict,
                      pred_frequency: str, k_decay: float = 0.1, random_seed: int = None):
    """
    Handle large gap prcoessing pipeline.
    """
    # Load catchment spatial data and determine catchment size for interpolation type
    spatial_df = pd.read_csv(spatial_path)
    spatial_df['station_name'] = spatial_df['station_name'].str.lower().str.replace(' ', '_')
    large_catchment = define_catchment_size(spatial_df, catchment, threshold_m)
    
    # Calculate pairwise distances using size-specific method
    distance_matrix = calculate_station_distances(spatial_df, large_catchment, radius)
    distance_matrix = np.round(distance_matrix, decimals=2)
    logging.info(f"{catchment}: Distance matrix calculated using "
                 f"{'Haversine' if large_catchment else 'Euclidean'} method.\n")
    
    # Calculate correlation matrix in range [0, 1]
    correlation_matrix = calculate_station_correlations(df_dict, catchment)

    # Score nearby stations for stations in gaps_list
    filtered_scores = score_station_proximity(df_dict, gaps_list, correlation_matrix, distance_matrix,
                                      k_decay, output_path, threshold)
    
    # --- Mask and imputate synthetic (for validation) and NaN data gaps --- 
    
    # Deep copy dataframe for future reference, only validate on copy not modifying original
    df_dict_original = copy.deepcopy(df_dict)
    
    # Validate synthetic imputation performace + plot imputed vs actual ts plots (overlaid)
    (synthetic_imputation_performace, synthetic_gap_flags, imputed_df_dict_synthetic_run,
     summary_df) = synthetic_gap_imputation_validation(
        df_dict_original=df_dict_original,
        gaps_list=gaps_list,
        min_around=min_around,
        predefined_large_gap_lengths=predefined_large_gap_lengths,
        max_imputation_length_threshold=max_imputation_length_threshold,
        filtered_scores=filtered_scores,
        validation_plot_path=output_path,
        imputation_plot_path=path,
        pred_frequency=pred_frequency,
        station_max_gap_lengths=station_max_gap_lengths,
        random_seed=random_seed
    )
    
    # --- Revert df back to original with only real gaps imputed and mask---
    cleaned_df_dict = clean_df_dict(
        df_dict=imputed_df_dict_synthetic_run,
        df_dict_original=df_dict_original,
        synthetic_gap_flags=synthetic_gap_flags,
    )
    
    # --- Return final df with all gaps imputed or masked -> all with 100% coverage ---
    return synthetic_imputation_performace, cleaned_df_dict

def impute_across_large_gaps(
    df_dict_to_impute: dict,
    filtered_scores: dict,
    max_imputation_length_threshold: int,
    df_dict_original: dict,
    imputation_method: str = 'weighted_average',
    imputed_value_flag: str = 'Imputed_LargeGap',
    distance_matrix: pd.DataFrame = None,
    k_decay: float = 0.1
):
    """
    Impute across larger gaps using one of:
      - 'weighted_average'  : donor-station (your existing method)
      - 'pchip'             : per-station PCHIP interpolation
      - 'spline'            : per-station cubic spline interpolation
      - 'linear'            : per-station linear interpolation
    Notes:
      • Kriging removed as a faulty implementation.
      • Linear interpolation added as a common baseline.
    """
    import numpy as np
    import pandas as pd
    import logging

    logging.info(f"Imputing large gaps in the data with method='{imputation_method}'...")

    # Deep copy so we don't mutate caller's dict
    imputed_df_dict = {station: df.copy(deep=True) for station, df in df_dict_to_impute.items()}

    # ---- helpers ----
    def _interp_fill(station_name: str, method: str, imputable_nan_timestamps: pd.DatetimeIndex):
        s = imputed_df_dict[station_name]['value'].copy()
        try:
            # Added a specific case for linear interpolation for clarity
            if method == 'linear':
                s_interp = s.interpolate(method='linear', limit_direction='both')
            elif method == 'pchip':
                s_interp = s.interpolate(method='pchip', limit_direction='both')
            elif method == 'spline':
                try:
                    s_interp = s.interpolate(method='spline', order=3, limit_direction='both')
                except Exception:
                    s_interp = s.interpolate(method='time', limit_direction='both')
            else:
                # Default fallback for any other method not explicitly listed
                s_interp = s.interpolate(method='time', limit_direction='both')
        except Exception as e:
            logging.warning(f"Interpolation failed for {station_name} with method '{method}': {e}. Falling back to linear.")
            s_interp = s.interpolate(method='linear', limit_direction='both')

        imputed_vals = s_interp.reindex(imputable_nan_timestamps)
        mask = imputed_vals.notna()
        if mask.any():
            idx_to_write = imputable_nan_timestamps[mask]
            imputed_df_dict[station_name].loc[idx_to_write, 'value'] = imputed_vals[mask]
            imputed_df_dict[station_name].loc[idx_to_write, 'data_type'] = imputed_value_flag

    # -----------------

    # Decide which stations to process:
    if imputation_method in ('weighted_average'):
        stations_to_process = list(filtered_scores.keys())
    else:
        # PCHIP, Spline, Linear, etc. will all process the same set of stations
        stations_to_process = list(filtered_scores.keys())

    for gappy_station in stations_to_process:
        if gappy_station not in imputed_df_dict:
            logging.info(f"Skipping {gappy_station}: not present in df_dict_to_impute.")
            continue

        gappy_station_df = imputed_df_dict[gappy_station]

        # Ensure DateTime index
        if not isinstance(gappy_station_df.index, pd.DatetimeIndex):
            gappy_station_df['dateTime'] = pd.to_datetime(gappy_station_df['dateTime'], errors='coerce')
            gappy_station_df = gappy_station_df.set_index('dateTime').sort_index()
            imputed_df_dict[gappy_station] = gappy_station_df

        # Identify NaNs to consider and reduce to gaps within threshold
        nan_timestamps = gappy_station_df['value'][gappy_station_df['value'].isna()].index
        if nan_timestamps.empty:
            logging.info(f"No NaNs found for {gappy_station}, skipping.")
            continue

        if len(nan_timestamps) <= max_imputation_length_threshold:
            imputable_nan_timestamps = nan_timestamps
        else:
            imputable_nan_timestamps = group_consecutive_gaps(nan_timestamps, max_imputation_length_threshold)

        if imputable_nan_timestamps.empty:
            logging.info(f"No NaNs within threshold for {gappy_station}, skipping.")
            continue

        # -------- donor-based path --------
        if imputation_method == 'weighted_average':
            donor_scores = filtered_scores.get(gappy_station, {})
            sorted_donors_series = pd.Series(donor_scores).sort_values(ascending=False)
            donor_names = [d for d in sorted_donors_series.index if d in imputed_df_dict]
            if not donor_names:
                logging.info(f"{gappy_station}: no valid donors available; skipping donor imputation.")
                continue

            # Build donor value frame aligned to imputable timestamps
            donor_data_for_gaps = pd.DataFrame(index=imputable_nan_timestamps)
            for donor in donor_names:
                donor_df = imputed_df_dict[donor]
                if not isinstance(donor_df.index, pd.DatetimeIndex):
                    donor_df['dateTime'] = pd.to_datetime(donor_df['dateTime'], errors='coerce')
                    donor_df = donor_df.set_index('dateTime').sort_index()
                    imputed_df_dict[donor] = donor_df
                donor_data_for_gaps[donor] = donor_df['value'].reindex(imputable_nan_timestamps)

            donor_data_for_gaps.dropna(axis=1, how='all', inplace=True)
            if donor_data_for_gaps.empty:
                logging.info(f"{gappy_station}: donors have no data at gap timestamps; skipping.")
                continue

            filtered_sorted = sorted_donors_series[sorted_donors_series.index.intersection(donor_data_for_gaps.columns)]
            if filtered_sorted.empty:
                logging.info(f"{gappy_station}: no overlapping donors after filtering; skipping.")
                continue

            donor_offsets = calculate_donor_offsets(
                gappy_station_name=gappy_station,
                gappy_station_df=imputed_df_dict[gappy_station],
                donor_names=donor_data_for_gaps.columns.tolist(),
                df_dict_original=df_dict_original
            )

            weighted_imputation(
                nan_timestamps=imputable_nan_timestamps,
                imputation_method='weighted_average',
                sorted_donors=filtered_sorted,
                gappy_station_df=imputed_df_dict[gappy_station],
                donor_data_for_gaps=donor_data_for_gaps,
                gappy_station_name=gappy_station,
                donor_offsets=donor_offsets,
                imputed_value_flag=imputed_value_flag
            )

        # -------- temporal interpolation paths --------
        elif imputation_method in ('pchip', 'spline'):
            _interp_fill(gappy_station, imputation_method, imputable_nan_timestamps)

        # -------- NEW LINEAR INTERPOLATION BASELINE --------
        elif imputation_method == 'linear':
            logging.info(f"Using linear interpolation for {gappy_station}.")
            _interp_fill(gappy_station, 'linear', imputable_nan_timestamps)

        else:
            logging.warning(f"Unknown imputation_method='{imputation_method}'. Falling back to 'pchip'.")
            _interp_fill(gappy_station, 'pchip', imputable_nan_timestamps)

    logging.info("Large gap imputation process complete for all stations.")
    return imputed_df_dict

def synthetic_gap_imputation_validation(
    df_dict_original: dict,
    gaps_list: list,
    min_around: int,
    predefined_large_gap_lengths: list,
    max_imputation_length_threshold: int,
    filtered_scores: dict,
    validation_plot_path: str,
    imputation_plot_path: str,
    pred_frequency: str,
    station_max_gap_lengths: dict = None,
    random_seed: int = None,
    imputation_method: str = "weighted_average",
    distance_matrix: pd.DataFrame = None,   # <— NEW
    k_decay: float = 0.1                     # <— NEW
):
    """
    Synthetic validation:
      1) Create synthetic gaps per station (lengths adapted to each station).
      2) Impute across *all* synthetic gaps using the chosen `imputation_method`.
      3) Compute RMSE/MAE/KGE vs. the true (masked) values.
      4) Plot per-station validation overlays.
    """
    import numpy as np
    import pandas as pd
    import logging

    logging.info(f"Starting synthetic gap imputation validation with method='{imputation_method}'...")

    validation_results = {}

    # --- 1) Working copy for inserting synthetic gaps (original kept intact) ---
    working_df_for_synthetic_gaps = {st: df.copy(deep=True) for st, df in df_dict_original.items()}
    all_original_masked_values = {}
    synthetic_gap_flags = {station: set() for station in gaps_list}

    # --- 2) Create station-specific synthetic gaps ---
    for gappy_station in gaps_list:
        logging.info(f"  Creating synthetic gaps for {gappy_station}...")
        st_df = working_df_for_synthetic_gaps[gappy_station]
        if 'dateTime' in st_df.columns:
            st_df['dateTime'] = pd.to_datetime(st_df['dateTime'], errors='coerce')
            st_df = st_df.set_index('dateTime').sort_index()
            working_df_for_synthetic_gaps[gappy_station] = st_df

        valid_idx = df_dict_original[gappy_station]['value'].dropna().index
        max_len_actual = station_max_gap_lengths.get(gappy_station, 0) if station_max_gap_lengths else 0

        gap_lengths_to_test = find_validation_gap_lengths(
            max_len_station_actual=max_len_actual,
            max_imputation_length_threshold=max_imputation_length_threshold,
            gappy_station=gappy_station,
            predefined_large_gap_lengths=predefined_large_gap_lengths,
            valid_synthetic_indices=valid_idx,
            min_around=min_around
        )
        if not gap_lengths_to_test:
            logging.warning(f"  No valid synthetic gap lengths for {gappy_station}. Skipping.")
            continue

        original_masked_series = synthetic_gap_allocation(
            gap_lengths_to_test=gap_lengths_to_test,
            min_around=min_around,
            valid_synthetic_indices=valid_idx,
            gappy_station=gappy_station,
            current_gappy_station_df=working_df_for_synthetic_gaps[gappy_station],
            df_dict_original=df_dict_original,
            df_for_validation=working_df_for_synthetic_gaps,
            synthetic_gap_flags=synthetic_gap_flags,
            seed=random_seed
        )
        if original_masked_series.empty:
            logging.info(f"  No synthetic gaps created for {gappy_station}.")
            continue

        all_original_masked_values[gappy_station] = original_masked_series

    if not any(len(v) for v in all_original_masked_values.values()):
        logging.warning("No synthetic gaps were created in any station. Aborting validation.")
        empty_summary = pd.DataFrame(
            {"RMSE_mean": [], "MAE_mean": [], "KGE_mean": [], "n": [],
             "RMSE_sd": [], "MAE_sd": [], "KGE_sd": [],
             "RMSE_ci": [], "MAE_ci": [], "KGE_ci": []}
        )
        return validation_results, synthetic_gap_flags, {}, df_dict_original, empty_summary

    # --- 3) Impute across *all* synthetic gaps using chosen method (now passes distance_matrix + k_decay) ---
    imputed_df_dict_synthetic_run = impute_across_large_gaps(
        df_dict_to_impute=working_df_for_synthetic_gaps,
        filtered_scores=filtered_scores,
        max_imputation_length_threshold=max_imputation_length_threshold,
        df_dict_original=df_dict_original,
        imputation_method=imputation_method,
        imputed_value_flag='Imputed_LargeGap',
        distance_matrix=distance_matrix,     # <— NEW
        k_decay=k_decay                      # <— NEW
    )

    # Optional fixes
    imputed_df_dict_synthetic_run = correct_suspect_variance_scaleby(imputed_df_dict_synthetic_run, pred_frequency)
    imputed_df_dict_synthetic_run = correct_suspect_variance_renwick(imputed_df_dict_synthetic_run)

    # --- 4) Metrics + plots ---
    rows = []
    for gappy_station in gaps_list:
        masked_truth = all_original_masked_values.get(gappy_station)
        if masked_truth is None or masked_truth.empty:
            continue

        rmse, mae, kge, imputed_vals = calc_validation_performance_metrics(
            original_values_masked=masked_truth,
            imputed_df_dict_synthetic_run=imputed_df_dict_synthetic_run,
            gappy_station=gappy_station,
            validation_results=validation_results
        )
        if not (isinstance(rmse, float) and isinstance(mae, float)) or (np.isnan(rmse) or np.isnan(mae)):
            continue

        plot_imputation_validation_results(
            df_dict_original=df_dict_original,
            gappy_station=gappy_station,
            rmse=rmse,
            mae=mae,
            imputed_df_dict_synthetic_run=imputed_df_dict_synthetic_run,
            original_values_masked=masked_truth,
            imputed_values_at_synthetic_gaps=imputed_vals,
            output_path=imputation_plot_path
        )

        rows.append({"station": gappy_station, "method": imputation_method,
                     "rmse": rmse, "mae": mae, "kge": kge})

    # --- 5) Summary across stations for this run ---
    if rows:
        station_df = pd.DataFrame(rows).set_index("station")
        n = len(station_df)
        rmse_mean = station_df["rmse"].mean()
        mae_mean  = station_df["mae"].mean()
        kge_mean  = station_df["kge"].mean()
        rmse_sd = station_df["rmse"].std(ddof=1); rmse_ci = 1.96 * (rmse_sd / np.sqrt(n)) if n > 1 else np.nan
        mae_sd  = station_df["mae"].std(ddof=1);  mae_ci  = 1.96 * (mae_sd  / np.sqrt(n)) if n > 1 else np.nan
        kge_sd  = station_df["kge"].std(ddof=1);  kge_ci  = 1.96 * (kge_sd  / np.sqrt(n)) if n > 1 else np.nan

        summary_df = pd.DataFrame({
            "RMSE_mean":[rmse_mean], "MAE_mean":[mae_mean], "KGE_mean":[kge_mean],
            "n":[n],
            "RMSE_sd":[rmse_sd], "MAE_sd":[mae_sd], "KGE_sd":[kge_sd],
            "RMSE_ci":[rmse_ci], "MAE_ci":[mae_ci], "KGE_ci":[kge_ci],
        }, index=[imputation_method])
    else:
        summary_df = pd.DataFrame(
            {"RMSE_mean":[], "MAE_mean":[], "KGE_mean":[], "n":[],
             "RMSE_sd":[], "MAE_sd":[], "KGE_sd":[],
             "RMSE_ci":[], "MAE_ci":[], "KGE_ci":[]}
        )

    logging.info(f"Validation complete for method='{imputation_method}'.")
    return validation_results, synthetic_gap_flags, imputed_df_dict_synthetic_run, summary_df
