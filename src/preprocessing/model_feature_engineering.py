# Import Libraries
import sys
import logging
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

def _group_features_by_type(processed_df):
    # Define numerical feature list
    numerical_features = ['mean_elevation', 'mean_slope_deg', 'mean_aspect_sin', 'mean_aspect_cos',
                        'rainfall_volume_m3', 'rainfall_lag_1', 'rainfall_lag_2', 'rainfall_lag_3',
                        'rainfall_lag_4', 'rainfall_lag_5', 'rainfall_lag_6', 'rainfall_lag_7',
                        'rolling_30', 'rolling_60', '2m_temp', 'aet_volume', 'surface_pressure',
                        'season_sin', 'season_cos']
    # Check list defensively
    numerical_features = [feat for feat in numerical_features if feat in processed_df.columns]

    # Define categorical feature list
    categorical_features = ['land_cover_code']
    categorical_features = [feat for feat in categorical_features if feat in processed_df.columns]

    # Define groundwater data feature list
    gwl_features = ['gwl_value', 'gwl_data_quality', 'gwl_data_type', 'gwl_masked', 'gwl_lag1',
                    'gwl_lag2', 'gwl_lag3', 'gwl_lag4', 'gwl_lag5', 'gwl_lag6', 'gwl_lag7']
    gwl_features = [feat for feat in gwl_features if feat in processed_df.columns]
    
    # Check feature count as expected
    other = ['node_id', 'timestep', 'Unnamed: 0']
    other = [feat for feat in other if feat in processed_df.columns]
    
    feature_count = len(numerical_features) + len(categorical_features) + len(gwl_features) + len(other)
    assert feature_count == len(processed_df.columns), \
        f"{len(processed_df.columns) - feature_count} features missing from definition."
    
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
        'aet_volume', 'surface_pressure','season_sin', 'season_cos'
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

def preprocess_shared_features(main_df_full, catchment, random_seed, violin_plt_path):
    """
    Apply final preprocessing to shared features only to reduce operation repetition after splitting
    while avoiding data leakage. GWL preprocessingn is undertaken post data split.
    """
    logger.info(f"Beginning preprocessing of shared data for {catchment} catchment (avoiding leakage)...\n")

    # Copy main df, dropping unneeded columns and renaming columns for clarity
    processed_df = main_df_full.copy().drop(columns='station_name')
    processed_df = processed_df.rename(columns={
        '2m_temp_area_weighted_mean': '2m_temp',
        'aet_total_volume_m3': 'aet_volume',
        'surface_pressure_area_weighted_mean': 'surface_pressure',
        'mean_slope_degrees': 'mean_slope_deg'
    })

    # --- Get feature lists split by type ---

    num_feats, cat_feats, _ = _group_features_by_type(processed_df)
    
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

    if cat_feats:
        # Fit on all available data for feature(s)
        logger.info(f"Beginning one hot encoding of {len(cat_feats)} categorical features...")
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(processed_df[cat_feats])
        encoded_features = encoder.transform(processed_df[cat_feats])

        # Build encoded cols
        one_hot_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(cat_feats),
            index=processed_df.index
        )
        
        # concat orginal df except cateogrical features with now one hot encoded cols
        processed_df = pd.concat([processed_df.drop(columns=cat_feats, axis=1), one_hot_df], axis=1)
        
        # Update numerical and cateogrical features list for future use with encoded col names
        num_feats.extend(encoder.get_feature_names_out(cat_feats).tolist())
        cat_feats = []
        logger.info("Categorical features successfully one hot encoded.\n")

    else:
        logger.info("No categorical features found in dataframe, skipping one hot encoding\n")
        
    # --- Clean up df and assert no final NaN values ---

    processed_df[num_feats] = round(processed_df[num_feats], 4)
    processed_df = processed_df.drop(columns='Unnamed: 0')
    
    assert processed_df[num_feats].isna().sum().sum() == 0, \
        f"{processed_df[num_feats].isna().sum().sum()} NaN values found in numerical features after preprocessing.\n"
        
    # --- Plot standardised features as violin plot to verify success ---
    
    _plot_standardised_data_aligned(processed_df, random_seed, violin_plt_path)
    
    return processed_df, shared_scaler, encoder
