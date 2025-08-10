# Import Libraries
import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.preprocessing.hydroclimatic_feature_engineering import _save_lambda_to_config
    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

# Preprocess shared features in final dataframe
def _group_features_by_type(processed_df):
    """
    Divide column name into data type lists for subsequent preprocessing subsetting.
    """
    # Define numerical feature list
    numerical_features = ['mean_elevation', 'mean_slope_deg', 'mean_aspect_sin', 'mean_aspect_cos',
                        'rainfall_volume_m3', 'rainfall_lag_1', 'rainfall_lag_2', 'rainfall_lag_3',
                        'rainfall_lag_4', 'rainfall_lag_5', 'rainfall_lag_6', 'rainfall_lag_7',
                        'rolling_30', 'rolling_60', '2m_temp', 'aet_volume', 'surface_pressure',
                        'season_sin', 'season_cos', 'bedrock_perm_avg', 'superficial_perm_avg',
                        'distance_to_river', 'streamflow_total_m3']
    # Check list defensively
    numerical_features = [feat for feat in numerical_features if feat in processed_df.columns]

    # Define categorical feature list
    categorical_features = ['land_cover_code', 'geo_bedrock_type', 'geo_superficial_type',
                            'bedrock_flow_type', 'superficial_flow_type', 'HOST_soil_class',
                            'aquifer_productivity']
    categorical_features = [feat for feat in categorical_features if feat in processed_df.columns]

    # Define groundwater data feature list
    gwl_features = ['gwl_value', 'gwl_data_quality', 'gwl_data_type', 'gwl_masked', 'gwl_lag1',
                    'gwl_lag2', 'gwl_lag3', 'gwl_lag4', 'gwl_lag5', 'gwl_lag6', 'gwl_lag7',
                    'gwl_mean', 'gwl_dip']
    gwl_features = [feat for feat in gwl_features if feat in processed_df.columns]
    
    # Check feature count as expected
    other = ['node_id', 'timestep', 'Unnamed: 0']
    other = [feat for feat in other if feat in processed_df.columns]

    feature_count = len(numerical_features) + len(categorical_features) + len(gwl_features) + len(other)
    assert feature_count == len(processed_df.columns), \
        f"{len(processed_df.columns) - feature_count} features missing from definition."
        
    # Final tighter checks (remove above once happy)
    defined = set(numerical_features) | set(categorical_features) | set(gwl_features) | set(other)
    missing = [c for c in processed_df.columns if c not in defined]
    if missing:
        logger.warning(f"{len(missing)} columns not assigned to a type. First few: {missing[:10]}")
    
    return numerical_features, categorical_features, gwl_features

def _transform_skewed_data(processed_df, catchment):
    """
    Transform skew of slope and elevation data.
    """
    # Define config path
    config_path = "config/project_config.yaml"
    cols = ['mean_elevation', 'mean_slope_deg']
    
    for col in cols:
        # Log transform the data to reduce skew
        logger.info(f"Tranforming {col} data for {catchment} catchment...\n")
        logger.info(f"    Initial {col} Skewness: {skew(processed_df[col]):.4f}")
        
        # Apply box cox transform to raw data column
        transformed_vals, lambda_val = boxcox(processed_df[col] + 1)
        processed_df[col] = transformed_vals
        logger.info(f"    Transformed {col} Skewness (Box-Cox, lambda={lambda_val:.4f}):"
                    f" {skew(processed_df[col]):.4f}\n")

        # Save lambda for mathematical inversion after modelling
        _save_lambda_to_config(lambda_val, config_path, catchment, col)
    
    return processed_df

def _plot_standardised_data_aligned(processed_df, random_seed, violin_plt_path):
    """
    Plot simple violin plot to visually confirm standardisation success.
    """
    plt.figure(figsize=(15, 8))

    # Create random sample to reduce computation of repeated passes over full df 
    sample_df = processed_df.sample(n=500000, random_state=random_seed)
    
    # All rainfall lags are essentially identical so only keep 1
    sample_df = sample_df.drop(columns=['rainfall_lag_2', 'rainfall_lag_3', 'rainfall_lag_4', 'rainfall_lag_5',
                                        'rainfall_lag_6', 'rainfall_lag_7'], errors='ignore')
    
    sample_df = sample_df.rename(columns={'rainfall_lag_1': 'rainfall_lags'})
    features_to_plot = [
        'mean_elevation', 'mean_slope_deg', 'mean_aspect_sin', 'mean_aspect_cos',
        'rainfall_volume_m3', 'rainfall_lags',  'rolling_30', 'rolling_60', '2m_temp',
        'aet_volume', 'surface_pressure','season_sin', 'season_cos', 'bedrock_perm_avg',
        'superficial_perm_avg', 'aquifer_productivity', 'streamflow_total_m3'
    ]

    # Filter plot_df to include only the columns that actually exist and are in the desired list
    final_plot_columns = [col for col in features_to_plot if col in sample_df.columns]

    # Create violin plot using the specific list of columns
    sns.violinplot(data=sample_df[final_plot_columns], orient="v", inner="quartile", color="skyblue")

    # Stylise plot
    plt.title('Distribution of Standardised Numerical Features (Sampled Data)', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Value (Standardised)', fontsize=12)
    plt.ylim(-8, 8)
    plt.xticks(rotation=45, ha='right') # Rotate labels
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid
    plt.tight_layout() # Adjust layout to prevent overlapping
    
    # Save plot to figure path
    plt.savefig(violin_plt_path, dpi=300)
    
    plt.show()
    plt.close()

def _save_shared_feat_encoders(shared_scaler, shared_encoder, catchment, scaler_dir):
    
    # Ensure the directory exists
    os.makedirs(scaler_dir, exist_ok=True)

    # Define full paths for saving
    shared_scaler_path = os.path.join(scaler_dir, "shared_scaler.pkl")
    shared_encoder_path = os.path.join(scaler_dir, "shared_encoder.pkl")

    # Save the objects
    joblib.dump(shared_scaler, shared_scaler_path)
    logger.info(f"Shared scaler saved to: {shared_scaler_path}")

    joblib.dump(shared_encoder, shared_encoder_path)
    logger.info(f"Shared encoder saved to: {shared_encoder_path}")

    logger.info(f"Pipeline Step 'Save Scalers and Encoders' complete for {catchment} catchment.")

def preprocess_shared_features(main_df_full, catchment, random_seed, violin_plt_path, scaler_dir):
    """
    Apply final preprocessing to shared features only to reduce operation repetition after splitting
    while avoiding data leakage. GWL preprocessingn is undertaken post data split.
    """
    logger.info(f"Beginning preprocessing of shared data for {catchment} catchment (avoiding leakage)...\n")

    # Copy main df, dropping unneeded columns and renaming columns for clarity
    processed_df = main_df_full.copy().drop(columns='station_name', errors='ignore')
    processed_df = processed_df.rename(columns={
        '2m_temp_area_weighted_mean': '2m_temp',
        'aet_total_volume_m3': 'aet_volume',
        'surface_pressure_area_weighted_mean': 'surface_pressure',
        'mean_slope_degrees': 'mean_slope_deg'
    })

    # --- Get feature lists split by type ---

    num_feats, cat_feats, gwl_feats = _group_features_by_type(processed_df)
    
    # --- Transform slope and elevation cols (previously kept raw for adge_attrs) ---
    
    _transform_skewed_data(processed_df, catchment)

    # --- Standardising Numerical ---

    if num_feats:
        logger.info(f"Beginning standardisation of {len(num_feats)} numerical features...")
        shared_scaler = StandardScaler()  # shared_scaler to differentiate from gwl_scaler for inversions
        shared_scaler.fit(processed_df[num_feats])
        processed_df[num_feats] = shared_scaler.transform(processed_df[num_feats])
        logger.info("Numerical features successfully standardised.\n")
    else:
        logger.info("No numerical features found in dataframe, skipping standardisation.\n")

    # --- One Hot Encoding Categorical ---

    # Initialise shared encoder for stability (if no cat feats)
    shared_encoder = None

    if cat_feats:
        # Fit on all available data for feature(s)
        logger.info(f"Beginning one hot encoding of {len(cat_feats)} categorical features...")
        shared_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        shared_encoder.fit(processed_df[cat_feats])
        encoded_features = shared_encoder.transform(processed_df[cat_feats])

        # Build encoded cols
        one_hot_df = pd.DataFrame(
            encoded_features,
            columns=shared_encoder.get_feature_names_out(cat_feats),
            index=processed_df.index
        )
        
        shared_ohe_cols = shared_encoder.get_feature_names_out(cat_feats).tolist() if cat_feats else []
        joblib.dump(shared_ohe_cols, os.path.join(scaler_dir, "shared_ohe_cols.pkl"))
        
        # concat orginal df except cateogrical features with now one hot encoded cols
        processed_df = pd.concat([processed_df.drop(columns=cat_feats, axis=1), one_hot_df], axis=1)
        
        # Update numerical and cateogrical features list for future use with encoded col names
        num_feats.extend(shared_encoder.get_feature_names_out(cat_feats).tolist())
        cat_feats = []
        logger.info("Categorical features successfully one hot encoded.\n")

    else:
        logger.info("No categorical features found in dataframe, skipping one hot encoding\n")
        
    # --- Clean up df and assert no final NaN values ---

    # processed_df[num_feats] = round(processed_df[num_feats], 4)
    # processed_df = processed_df.drop(columns='Unnamed: 0')
       
    processed_df = processed_df.drop(columns='Unnamed: 0', errors='ignore')

    # Enforce float32 on numerics (cast OHE already float64 too for memory)
    num_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[num_cols] = processed_df[num_cols].astype(np.float32)

    assert processed_df[num_feats].isna().sum().sum() == 0, \
        f"{processed_df[num_feats].isna().sum().sum()} NaN values found in numerical features after preprocessing.\n"
            
    # --- Plot standardised features as violin plot to verify success ---
    
    _plot_standardised_data_aligned(processed_df, random_seed, violin_plt_path)
    
    # Save Encoders
    _save_shared_feat_encoders(shared_scaler, shared_encoder, catchment, scaler_dir)
    
    return processed_df, shared_scaler, shared_encoder, gwl_feats

# Preprocess gwl featues in final dataframe
def _fill_missing_with_sentinel(df, cols, sentinel_value):
    """
    Fill all 'missing value' assignments in gwl data columns with sentinel value as required for input into GAT. df
    is modified in place so no return needed.
    """
    n_missing_before_sentinel_fill = df[cols].isnull().sum().sum()

    if n_missing_before_sentinel_fill > 0:
        logger.info(f"Filling {n_missing_before_sentinel_fill} missing values in specified columns with sentinel = {sentinel_value}.")
        df.loc[:, cols] = df.loc[:, cols].fillna(sentinel_value)
        logger.info(f"Sentinel fill complete for columns: {cols}.\n")
    else:
        logger.info(f"No missing values found in specified columns, skipping sentinel fill.\n")

def _standardise_gwl_station_data(df, num_cols, sentinel_value, gwl_scaler, all_gwl_station_ids,
                                  train_station_ids):
    """
    Standardise all OBSERVED GWL values using only training data to fit scaler to avoid leakage. df is
    modified in place so no return needed.
    """
    if not num_cols:
        logger.info("No numerical GWL features found, skipping standardisation.\n")
        return

    logger.info(f"Beginning standardisation of {len(num_cols)} numerical GWL features "
                f"(fitted on training data only)...")

    # Select only training station nodes and confirm none contain NaN or sentinel
    train_data_for_std = df[df['node_id'].isin(train_station_ids)][num_cols].copy()
    
    # Check no training data has been filled with sentinel vals, temporarily exclude if so
    n_sentinels_in_train_num = (train_data_for_std == sentinel_value).sum().sum()
    if n_sentinels_in_train_num > 0:
        train_data_for_std.replace(sentinel_value, np.nan, inplace=True)
        logger.warning(f"Found {n_sentinels_in_train_num} sentinel values in training numerical data for fitting.")
        
    # Check for persisting NaNs
    n_nan_in_train_num_before_fit = train_data_for_std.isna().sum().sum()
    assert n_nan_in_train_num_before_fit == 0, \
        f"Original training numerical data contains {n_nan_in_train_num_before_fit} genuine NaN values (not sentinels)."

    # Fit on training data
    gwl_scaler.fit(train_data_for_std)
    
    # Guard against zero var outcomes (just set to 0) - very unlikely but causes break when insufficient stations
    if np.any(np.isclose(gwl_scaler.scale_, 0)):
        zero_cols = [num_cols[i] for i,s in enumerate(gwl_scaler.scale_) if np.isclose(s,0)]
        logger.warning(f"Zero variance in training for: {zero_cols} — setting scaled values to 0 for those columns.")
    
    # --- Apply transformation to ALL rows (train, val, test) for these cols - NOT SENTINEL! ---
    
    # Create a boolean mask for rows to transform
    obs_mask = df['node_id'].isin(all_gwl_station_ids)
    
    # Make copy of the observed rows/cols and record sentinels
    X = df.loc[obs_mask, num_cols].copy()
    sentinel_mask = (X == sentinel_value)
    
    # Temporarily replace sentinels with the per-column training mean so transform is stable
    means = pd.Series(gwl_scaler.mean_, index=num_cols)
    for col in num_cols:
        X.loc[sentinel_mask[col], col] = means[col]
        
    # Transform all observed rows/cols (shape matches fit)
    X_scaled = gwl_scaler.transform(X.values)

    # Write back to df where observed (not sentinel originally)
    df.loc[obs_mask, num_cols] = X_scaled
    
    # Restore original sentinel values
    for col in num_cols:
        mask = obs_mask & sentinel_mask[col]
        df.loc[mask, col] = sentinel_value
    
    # Confirm expected sizes
    n_restored = sum((df.loc[obs_mask, num_cols] == sentinel_value).sum())
    logger.info(f"Standardised {len(num_cols)} GWL features; restored {n_restored} sentinel entries unchanged.")

def _standardise_target_feat(df, target_scaler, train_station_ids, target_col='gwl_value'):
    """
    Standardise the target GWL value ('gwl_value') using only training data to fit the scaler.
    df is modified in place.
    """
    logger.info(f"Beginning standardisation of target feature '{target_col}' (fitted on training data only)...")

    # Select only training station nodes for the target column
    train_target_data = df[df['node_id'].isin(train_station_ids)][[target_col]].copy()

    # Ensure no NaNs in training target data before fitting
    n_nan_in_train_target_before_fit = train_target_data.isna().sum().sum()
    if n_nan_in_train_target_before_fit > 0:
        logger.warning(f"Found {n_nan_in_train_target_before_fit} NaN values in training target data before"
                       f" scaler fit. These will be ignored by fit.")
        pass

    # Fit on training data
    target_scaler.fit(train_target_data)

    # Apply transformation to all rows for the target column (NaNs propagated)
    # df.loc[:, target_col] = target_scaler.transform(df.loc[:, [target_col]])

    # Only transform rows where target is not NaN
    target_mask = df[target_col].notna()
    df.loc[target_mask, target_col] = target_scaler.transform(df.loc[target_mask, [target_col]])
    
    logger.info(f"Successfully standardised target feature '{target_col}'.\n")

def _encode_gwl_station_data(df, cat_cols, gwl_encoder, all_gwl_station_ids, train_station_ids):
    """
    One hot encode all OBSERVED GWL categorical data using only training data to fit encoder to avoid
    leakage. df is modified in place so no return needed.
    """
    if cat_cols:
        logger.info(f"Beginning one-hot encoding of {len(cat_cols)} categorical GWL features "
                    f"(fitted on training data only, applied to observed GWL stations)...")
        
        # Select only training station nodes and confirm none contain NaN or seninel
        train_data_for_encoder = df[df['node_id'].isin(train_station_ids)][cat_cols].copy()

        # Fill any NaNs in categorical columns with placeholder before encoding for fitting
        for col in cat_cols:
            if col in train_data_for_encoder.columns:
                n_cat_nan_before_fill = train_data_for_encoder[col].isna().sum()
                if n_cat_nan_before_fill > 0:
                    logger.info(f"    {n_cat_nan_before_fill} NaN values in '{col}' (training data) filling with '__MISSING_CAT__' for encoder fit.")
                    train_data_for_encoder.loc[:, col] = train_data_for_encoder[col].fillna('__MISSING_CAT__')
        
        # Fit encoder using only training data
        gwl_encoder.fit(train_data_for_encoder)
        
        # --- Apply transformation to ALL rows (train, val, test) for these cols ---
        
        mask_for_cat_transform = df['node_id'].isin(all_gwl_station_ids)
        
        # Prepare missing data for transformation (fill NaNs for all rows for consistent transformation)
        data_to_encode_for_transform = df[cat_cols].copy()
        for col in cat_cols:
            data_to_encode_for_transform.loc[:, col] = data_to_encode_for_transform[col].fillna('__MISSING_CAT__')
            
        # Tranform prepared data
        encoded_features_arr = gwl_encoder.transform(data_to_encode_for_transform)
        
        # Get new one hot encoded colums and initalise as 0 before merge
        new_ohe_cols = gwl_encoder.get_feature_names_out(cat_cols).tolist()
        for new_col in new_ohe_cols:
            df[new_col] = 0.0
        
        # Assign the transformed values to ONLY the masked rows
        df.loc[mask_for_cat_transform, new_ohe_cols] = encoded_features_arr[mask_for_cat_transform]
    
        # Drop original categorical columns from df
        df.drop(columns=cat_cols, axis=1, inplace=True)
        
        logger.info(f"Successfully one-hot encoded {len(cat_cols)} categorical GWL features across "
                f"{mask_for_cat_transform.sum()} rows for observed GWL stations.\n")
    else:
        logger.info("No categorical GWL features found, skipping one hot encoding.\n")

def _modify_final_col_order(processed_df):
    """
    Force 'timestep' and 'node_id' to be the first two columns for clarity.
    """
    logger.info(f"Adjuat final column order of dataframe.")
    
    # Get all current columns and create desired order for first cols
    all_cols = processed_df.columns.tolist()
    desired_initial_cols = ['timestep', 'node_id']
    
    # Filter out 'timestep' and 'node_id' from the full df
    remaining_cols = [col for col in all_cols if col not in desired_initial_cols]
    
    # Combine to create the new column order and reindex the df to apply the new order
    new_column_order = desired_initial_cols + remaining_cols
    processed_df = processed_df[new_column_order]
    
    return processed_df

def _save_gwl_encoders(gwl_scaler, gwl_encoder, target_scaler, catchment, scaler_dir):
    
    # Ensure the directory exists
    os.makedirs(scaler_dir, exist_ok=True)

    # Define full paths for saving
    gwl_scaler_path = os.path.join(scaler_dir, "gwl_scaler.pkl")
    gwl_encoder_path = os.path.join(scaler_dir, "gwl_encoder.pkl")
    target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")

    # Save the objects
    joblib.dump(gwl_scaler, gwl_scaler_path)
    logger.info(f"GWL scaler saved to: {gwl_scaler_path}")

    joblib.dump(gwl_encoder, gwl_encoder_path)
    logger.info(f"GWL encoder saved to: {gwl_encoder_path}")
    
    joblib.dump(target_scaler, target_scaler_path)
    logger.info(f"Target scaler saved to: {target_scaler_path}")

    logger.info(f"Pipeline Step 'Save Scalers and Encoders' complete for {catchment} catchment.")

def preprocess_gwl_features(processed_df, catchment, train_station_ids, val_station_ids, test_station_ids,
                            sentinel_value, scaler_dir):
    """
    Preprocesses groundwater level (GWL) features by handling missing values with sentinels,
    standardising numerical features, and one-hot encoding categorical features using only training data
    to prevent leakage. Integrate these processed features back into the full df and order cols.
    """
    logger.info(f"Processing final groundwater data for {catchment} catchment...\n")
    
    # Drop unneeded cols - masked col is a duplicate of 'masked' category in data type col
    processed_df = processed_df.drop(columns=['gwl_masked'])

    # --- Group columns by type ---

    # Define gwl specific columns including node_id and timestep (for merge)
    gwl_cols = ['timestep', 'node_id', 'gwl_value', 'gwl_data_quality', 'gwl_data_type', 'gwl_lag1',
                'gwl_lag2', 'gwl_lag3', 'gwl_lag4', 'gwl_lag5', 'gwl_lag6', 'gwl_lag7', 'gwl_mean', 'gwl_dip']
                # Note that 'gwl_mean' and 'gwl_dip' are static not gwl masked features but need preprocessing by station
    
    # Filter to ensure only columns actually present in processed_df are used
    gwl_cols = [col for col in gwl_cols if col in processed_df.columns]

    # Define columns by type for various preprocessing requirements (defensively using processed_df checks)
    num_cols = [f'gwl_lag{i}' for i in range(1, 8) if f'gwl_lag{i}' in processed_df.columns]
    num_cols.extend(['gwl_mean', 'gwl_dip'])
    cat_cols = [col for col in ['gwl_data_quality', 'gwl_data_type', 'gwl_masked'] if col in processed_df.columns]
    
    idx_cols = ['timestep', 'node_id']
    target_col = 'gwl_value'
    
    # Define list of all (observed) gwl station IDs
    all_gwl_station_ids = list(set(train_station_ids + val_station_ids + test_station_ids))

    # Take copy of original df to process
    df = processed_df.copy()

    # --- Fill numerical NaNs with Sentinel Value ---

    _fill_missing_with_sentinel(df, num_cols, sentinel_value)

    # --- Standardise training data rows and propagate standardisation params to val and test rows ---

    # Initialise standard scaler and standardise numerical values using training data params
    gwl_scaler = StandardScaler()
    _standardise_gwl_station_data(df, num_cols, sentinel_value, gwl_scaler,
                                  all_gwl_station_ids, train_station_ids)
    
    # Standardise gwl_value (target variable, y)
    # target_scaler = None
    target_scaler = StandardScaler()
    _standardise_target_feat(df, target_scaler, train_station_ids)
        
    # --- One hot encode categorical gwl features ---

    # Initialise categorical encode and encode categorical columns using training data params
    gwl_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    _encode_gwl_station_data(df, cat_cols, gwl_encoder, all_gwl_station_ids, train_station_ids)
    
    # Save cols for later -999/0.0 fill type reference
    gwl_ohe_cols = gwl_encoder.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    joblib.dump(gwl_ohe_cols, os.path.join(scaler_dir, "gwl_ohe_cols.pkl"))
    
    # --- Final Sentinel Fill for any remaining NaNs in GWL X-Features ---
    
    final_gwl_x_features = num_cols + gwl_encoder.get_feature_names_out(cat_cols).tolist() if cat_cols else num_cols
    _fill_missing_with_sentinel(df, final_gwl_x_features, sentinel_value)
        
    logger.info(f"Final GWL data processing complete for {catchment} catchment.")

    # --- Update original processed_df with reprocessed GWL columns ---
    
    # Ensure all original GWL input columns are dropped (numerical and categorical)
    original_gwl_input_cols = num_cols + cat_cols
    processed_df = processed_df.drop(columns=original_gwl_input_cols, errors='ignore')

    # Merge the updated GWL features back into processed_df
    processed_df = pd.merge(
        processed_df,
        df[idx_cols + final_gwl_x_features + [target_col]],
        on=['timestep', 'node_id'],
        how='left'
    )
    
    # Clean up final df, col order and log completion of processing
    processed_df = processed_df.drop(columns='gwl_value_x').rename(columns={'gwl_value_y': 'gwl_value'})
    processed_df = _modify_final_col_order(processed_df)
    
    # Enforce float32 on numeric columns (includes scaled GWL features and target)
    numeric_like_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_like_cols] = processed_df[numeric_like_cols].astype(np.float32)

    logger.info(f"Updated processed_df with standardised and encoded GWL features.")
    
    # Save Encoders
    _save_gwl_encoders(gwl_scaler, gwl_encoder, target_scaler, catchment, scaler_dir)

    # Return the modified df and the fitted transformers for potential inverse transforms or inspection
    return processed_df, gwl_scaler, gwl_encoder
