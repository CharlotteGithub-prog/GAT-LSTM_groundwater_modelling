# Copyright (c) 2025 Charlotte Wayment
# This file is part of the Dissertation project and is licensed under the MIT License.

### FULL PIPELINE ###

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================

# --- 1a. Library Imports ---
import sys
import torch
import random
import joblib
import logging
import numpy as np
import pandas as pd

# --- 1b. Project Imports ---
from src.utils.config_loader import load_project_config
from src.graph_building.graph_construction import build_mesh
from src.visualisation.mapped_visualisations import plot_interactive_mesh
from src.data_ingestion.gwl_data_ingestion import process_station_coordinates, \
    fetch_and_process_station_data, download_and_save_station_readings
from src.preprocessing.gwl_preprocessing import load_timeseries_to_dict, \
    outlier_detection, resample_daily_average, remove_spurious_data, \
    interpolate_short_gaps

# --- 1c. Logging Config ---
logging.basicConfig(
    level=logging.INFO,
    # format='%(levelname)s - %(message)s',  # Uncomment for short logging
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',  # Uncomment for full logging
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
config = load_project_config(config_path="config/project_config.yaml")
notebook = False

# --- 1d. Set up seeding to define global states ---
random_seed = config["global"]["pipeline_settings"]["random_seed"]
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 1e. Define catchment(s) and API calls to Process --
catchments_to_process = config["global"]["pipeline_settings"]["catchments_to_process"]
run_defra_API_calls = config["global"]["pipeline_settings"]["run_defra_api"]  # True to run API calls
run_camels_API_calls = config["global"]["pipeline_settings"]["run_camels_api"]  # True to run API calls
run_outlier_detection = config["global"]["pipeline_settings"]["run_outlier_detection"]

# Run full pipeline by catchment
try:
    for catchment in catchments_to_process:

        # ==============================================================================
        # SECTION 2: DATA INGESTION
        # ==============================================================================

        # --- 2a. Load and convert gwl station location data (DEFRA) ---
        
        stations_with_coords_df = process_station_coordinates(
            os_grid_squares=config["global"]["paths"]["gis_os_grid_squares"],
            station_list_input=config[catchment]["paths"]["gwl_station_list"],
            station_list_output=config[catchment]["paths"]["gwl_station_list_with_coords"],
            catchment=catchment
        )

        logger.info(f"Pipeline step 'Process Station Coordinates for {catchment}' complete.\n")
        
        # Only run API calls as needed
        if run_defra_API_calls:  

            # --- 2b. Retrieve station measures and metadata from DEFRA API ---

            stations_with_metadata_measures = fetch_and_process_station_data(
                stations_df=stations_with_coords_df,
                base_url=config["global"]["paths"]["defra_station_base_url"],
                output_path=config[catchment]["paths"]["gwl_station_metadata_measures"]
            )

            logger.info(f"Pipeline step 'Pull Hydrological Station Metadata for {catchment}' complete.\n") 
            
            # --- 2c. Retrieve raw gwl timeseris data by station from DEFRA API ---

            download_and_save_station_readings(
                stations_df=stations_with_metadata_measures,
                start_date=config["global"]["data_ingestion"]["api_start_date"],
                end_date=config["global"]["data_ingestion"]["api_end_date"],
                gwl_data_output_dir=config[catchment]["paths"]["gwl_data_output_dir"]
            )

            logger.info(f"All timeseries groundwater level data saved for {catchment} catchment.\n")
        
        else:
            
            loaded_csv_path = config[catchment]["paths"]["gwl_station_metadata_measures"]
            stations_with_metadata_measures = pd.read_csv(loaded_csv_path)
        
        # Only run API calls as needed
        if run_camels_API_calls: 
            
            # --- 2d. load camels-gb data ---
            
            logging.info("skipping...")
            
        else:
            
            logging.info("skipping...")
        
        # --- 2x. load other data ---

        # ==============================================================================
        # SECTION 3: PREPROCESSING
        # ==============================================================================
        
        # --- 3a. gwl preprocessing ---
        
        # Load station df's into dict, dropping catchments with insufficient data
        
        # Load timeseries CSVs from API into reference dict
        gwl_time_series_dict = load_timeseries_to_dict(
            stations_df=stations_with_metadata_measures,
            col_order=config["global"]["data_ingestion"]["col_order"],
            data_dir=config[catchment]["paths"]["gwl_data_output_dir"],
            inclusion_threshold=config[catchment]["preprocessing"]["inclusion_threshold"]
        )

        logger.info(f"All timeseries data converted to dict for {catchment} catchment.\n")
        
        # # Remove outlying and incorrect (user defined: spurious) data points
        
        for station_name, df in gwl_time_series_dict.items():
            gwl_time_series_dict[station_name] = remove_spurious_data(
                target_df=df,
                station_name=station_name,
                path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
                notebook=True
            )
            
        logger.info(f"Pipeline step 'Remove spurious points' complete for {catchment} catchment.\n")
        
        if run_outlier_detection:   
            processed_gwl_time_series_dict = outlier_detection(
                gwl_time_series_dict=gwl_time_series_dict,
                output_path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
                dpi=config[catchment]["visualisations"]["ts_plots"]["dpi_save"],
                dict_output=config[catchment]["paths"]["gwl_outlier_dict"],
                notebook=notebook
            )
            
            logger.info(f"All outlying data processed for {catchment} catchment.\n")
        
        else:
            input_dict = config[catchment]["paths"]["gwl_outlier_dict"]
            processed_gwl_time_series_dict = joblib.load(input_dict)
        
        logger.info(f"Pipeline step 'Run outlier detection and cleaning' complete for {catchment} catchment.\n")
        
        # Aggregate to daily time 
        
        daily_data = resample_daily_average(
            dict=processed_gwl_time_series_dict,
            start_date=config["global"]["data_ingestion"]["api_start_date"],
            end_date=config["global"]["data_ingestion"]["api_end_date"],
            path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            notebook=notebook
        )
        
        logger.info(f"Pipeline step 'Resample to Daily Timesteps' complete for {catchment} catchment.\n")
        
        # Interpolate across small gaps in the ts data (define threshold n/o missing time steps for interpolation eligibility) + Add binary interpolation flag column
        
        for station_name, df in daily_data.items():
            gaps_list, daily_data[station_name] = interpolate_short_gaps(
                df=df,
                station_name=station_name,
                path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
                max_steps=config["global"]["data_ingestion"]["max_interp_length"],
                notebook=notebook
            )
            
        logger.info(f"Pipeline step 'Interpolate Short Gaps' complete for {catchment} catchment.\n")

        # Resolve larger gaps in data
        
        
        
        # Lagged: Add lagged features (by timestep across 7 days?) + potentially rolling averages (3-day/7-day?)
        
        # Temporal Encoding: Define sinasoidal features for seasonality (both sine and cosine for performance)

        # --- 3b. camels-gb preprocessing etc... to be defined for all other features (static then dynamic, all spatial) ---
        
        # --- 3x. Standardisation of all features ---

        # ==============================================================================
        # SECTION 4: GRAPH BUILDING
        # ==============================================================================

        # --- 4a. Build Catchment Graph Mesh ---

        mesh_nodes_table, mesh_nodes_gdf, catchment_polygon = build_mesh(
            shape_filepath=config[catchment]['paths']['gis_catchment_boundary'],
            output_path=config[catchment]['paths']['mesh_nodes_output'],
            grid_resolution=config[catchment]['preprocessing']['graph_construction']['grid_resolution']
        )

        logger.info(f"Pipeline step 'Build Mesh' complete for {catchment} catchment.\n")
        
        # --- 4b. Save interactive map of catchment mesh ---

        mesh_map = plot_interactive_mesh(
            mesh_nodes_gdf=mesh_nodes_gdf,
            catchment_polygon=catchment_polygon,
            map_blue=config['global']['visualisations']['maps']['map_blue'],
            esri=config['global']['visualisations']['maps']['esri'],
            esri_attr=config['global']['visualisations']['maps']['esri_attr'],
            static_output_path=config[catchment]['visualisations']['maps']['static_mesh_map_output'],
            interactive_output_path=config[catchment]['visualisations']['maps']['interactive_mesh_map_output'],
            grid_resolution=config[catchment]['preprocessing']['graph_construction']['grid_resolution'],
            interactive=config['global']['visualisations']['maps']['display_interactive_map']
        )

        logger.info(f"Pipeline step 'Interactive Mesh Mapping' complete for {catchment} catchment.\n")
        
        # --- 4c. Snap GWL monitoring stations to mesh nodes ---
        # Purpose: Assign observed GWL time series and associated metadata to the closest mesh nodes.
        # Action: Develop function to find nearest mesh node for each GWL station (e.g., using spatial indexing, k-d tree).
        # Action: Attach GWL time series data (and static GWL station features) to these specific mesh nodes.
        # Output: Updated mesh_nodes_gdf or a separate node feature tensor/dataframe.

        # --- 4d. Snap other features data to mesh nodes (e.g., CAMELS-GB data, static attributes) ---
        # Purpose: Integrate other relevant spatial and spatiotemporal features onto the mesh.
        # Action: For CAMELS-GB (e.g., rainfall gauges, river flow sites), snap to nearest mesh nodes.
        # Action: Assign CAMELS time series data (e.g., rainfall, temperature) and static attributes (e.g., elevation, soil type, geology) to all relevant mesh nodes.
        # Action: Ensure all node features (GWL, CAMELS, static) are aligned by time and node ID.
        # Output: Comprehensive node feature matrix/tensor (X).

        # --- 4e. Define Graph Adjacency Matrix (Edges) ---
        # Purpose: Establish connections between mesh nodes.
        # Action: Define criteria for edges (e.g., k-nearest neighbors, distance threshold, hydrological connectivity).
        # Action: Construct the adjacency matrix (A) for the graph.
        # Output: Adjacency matrix or edge_index for PyTorch Geometric.

        # --- 4f. Create Graph Data Object / Input Tensors ---
        # Purpose: Assemble all graph components (nodes, edges, features, targets) into a format suitable for the GNN framework.
        # Action: Split data into training, validation, and test sets based on time (e.g., 70/15/15 chronological split).
        # Action: Generate graph snapshots/sequences for the GNN's input.
        # Output: PyTorch Geometric Data objects or DGL graphs, including node features (X), edge index (edge_index), and target GWL values (Y) for observed nodes.

        # --- 4g. (Optional) Visualise complete mesh map with stations and other features ---
        # Purpose: Verify final graph structure and feature distribution visually.
        # Action: Create an interactive map showing the mesh, GWL stations, and other snapped data points.

        # ==============================================================================
        # SECTION 5: MODEL
        # ==============================================================================
        
        # --- 5x. Goal: GAT-LSTM ---
        
        # --- 5a. Define Graph Neural Network Architecture ---
        # Goal: Implement a GAT-LSTM (Graph Attention Network + Long Short-Term Memory).
        # Action: Define the GNN layers (e.g., GATConv for spatial message passing, LSTM for temporal learning).
        # Action: Specify input/output dimensions, hidden layer sizes, activation functions, dropout rates.
        # Output: A PyTorch nn.Module or similar model class.

        # --- 5b. Define Loss Function ---
        # Action: Choose an appropriate loss function for regression (e.g., Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss).
        # Action: Consider masking the loss calculation to only evaluate predictions at observed GWL stations if non-interpolated.

        # --- 5c. Define Optimiser ---
        # Action: Select an optimiser (e.g., Adam, SGD).
        # Action: Configure learning rate and other optimiser parameters.

        # ==============================================================================
        # SECTION 6: TRAINING
        # ==============================================================================
        
        # --- 6a. Implement Training Loop ---
        # Action: Iterate over epochs.
        # Action: For each epoch, iterate over mini-batches of graph data.
        # Action: Perform forward pass through the model.
        # Action: Calculate loss.
        # Action: Perform backward pass and optimize weights.
        # Action: Implement gradient clipping if necessary.

        # --- 6b. Implement Validation Loop ---
        # Action: Periodically evaluate model performance on the validation set.
        # Action: Monitor validation loss/metrics for early stopping.

        # --- 6c. Model Checkpointing and Logging ---
        # Action: Save best performing model weights based on validation metrics.
        # Action: Log training and validation metrics (e.g., using TensorBoard, MLflow, or custom logging).

        # ==============================================================================
        # SECTION 7: EVALUATION
        # ==============================================================================

        # --- 7a. Final Model Evaluation ---
        # Action: Load the best trained model.
        # Action: Evaluate its performance on the unseen test set.
        # Action: Calculate key metrics (e.g., RMSE, MAE, R-squared, Nash-Sutcliffe Efficiency for hydrology).
        # Output: Quantitative evaluation results.

        # --- 7b. Visualisation of Predictions ---
        # Action: Plot actual vs. predicted GWL time series for selected stations/nodes.
        # Action: Create animated maps showing predicted GWL changes over time across the mesh (if using interpolated pseudo-labels).

        # --- 7c. Error Analysis ---
        # Action: Identify systematic errors or biases in predictions (e.g., over/under-prediction during peaks/troughs).
        # Action: Explore reasons for poor performance at specific stations/nodes or time periods.
        
        # ==============================================================================
        # SECTION 8: INTERPRETATION & HYDROLOGICAL INSIGHTS
        # ==============================================================================

        # --- 8a. Feature Importance Analysis (Global & Local) ---
        # Purpose: Understand which input features (climate, static, lagged GWL) drive predictions.
        # Action: Apply SHAP (SHapley Additive exPlanations) or similar methods (e.g., Permutation Importance) to identify global feature importance.
        # Action: For specific predictions or events, apply SHAP/LIME to understand local feature contributions.
        # Output: Feature importance plots, tables.

        # --- 8b. Spatial Relationship Interpretation (Attention/GNNExplainer) ---
        # Purpose: Uncover how the GNN leverages spatial connections and neighbors.
        # Action: If using GAT, analyze learned attention weights to understand influence of neighboring nodes.
        # Action: Apply GNNExplainer (or similar graph-specific XAI methods) to identify critical subgraphs for specific node predictions.
        # Output: Visualizations of influential neighbors/connections on maps, attention heatmaps.

        # --- 8c. Analysis of Pseudo-Ungauged Generalisation ---
        # Purpose: Evaluate the model's ability to predict at nodes masked during training.
        # Action: Compare predictions at "held-out" (proxy-ungauged) nodes against their interpolated pseudo-ground-truth.
        # Action: Analyze performance metrics specifically for these nodes.
        # Action: Visualise the spatial distribution of errors for these ungauged nodes.

        # --- 8d. Hydrological Interpretation and Discussion ---
        # Purpose: Translate model insights back into hydrogeological understanding.
        # Action: Discuss consistency of feature importances and spatial influences with known hydrological principles (e.g., lag times, flow paths, aquifer properties).
        # Action: Provide explanations for observed model behaviors during specific hydrological events (droughts, floods).
        # Output: Written analysis and discussion points for dissertation.

        # --- 8e. Design Trade-offs Analysis (Optional, if time allows) ---
        # Purpose: Evaluate the impact of choices in graph construction.
        # Action: Compare results/interpretations from different graph resolutions (e.g., 500m vs 1000m) or edge definitions (e.g., KNN vs distance threshold).
        # Output: Comparative analysis.

# If critical pipeline error, exit with an error code
except Exception as e:
    logger.critical(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)
    sys.exit(1)