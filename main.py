# Copyright (c) 2025 Charlotte Wayment
# This file is part of the Dissertation project and is licensed under the MIT License.

### FULL PIPELINE ###

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================

# --- 1a. Library Imports ---
import sys
import logging

# --- 1b. Project Imports ---
from src.utils.config_loader import load_project_config
from src.graph_building.graph_construction import build_mesh
from src.visualisation.mapped_visualisations import plot_interactive_mesh
from src.data_ingestion.gwl_data_ingestion import process_station_coordinates, \
    fetch_and_process_station_data, download_and_save_station_readings

# --- 1c. Logging Config ---
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',  # Uncomment for short logging
    # format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',  # Uncomment for full logging
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
config = load_project_config(config_path="config/project_config.yaml")

# --- 1d. Define catchment(s) to Process --
catchments_to_process = config["global"]["pipeline_settings"]["catchments_to_process"]

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

        logger.info(f"All timeseries groundwater level data saved for {catchment} catchment.")
        
        # --- 2d. load camels-gb data ---
        
        # --- 2x. load other data ---

        # ==============================================================================
        # SECTION 3: PREPROCESSING
        # ==============================================================================
        
        # --- 3a. gwl preprocessing ---
        
        # Load station df's into dict, dropping catchments with insufficient data
        
        # Remove outlying and incorrect data points
        
        # Aggregate to daily time steps
        
        # Interpolate across small gaps in the ts data (define threshold n/o missing time steps for interpolation eligibility) + Add binary interpolation flag column
        
        # Lagged: Add lagged features (by timestep across 7 days?) + potentially rolling averages (3-day/7-day?)
        
        # Temporal Encoding: Define sinasoidal features for seasonality (both sine and cosine for performance)

        # --- 3b. camels-gb preprocessing etc... to be defined for all other features (static then dynamic, all spatial) ---
        
        # --- 3x. Standardisation of all features ---

        # ==============================================================================
        # SECTION 4: GRAPH BUILDING
        # ==============================================================================

        # --- 3a. Build Catchment Graph Mesh ---

        mesh_nodes_table, mesh_nodes_gdf, catchment_polygon = build_mesh(
            shape_filepath=config[catchment]['paths']['gis_catchment_boundary'],
            output_path=config[catchment]['paths']['mesh_nodes_output'],
            grid_resolution=config[catchment]['preprocessing']['graph_construction']['grid_resolution']
        )

        logger.info(f"Pipeline step 'Build Mesh' complete for {catchment} catchment.")
        
        # --- 3b. Save interactive map of catchment mesh ---

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

        logger.info(f"Pipeline step 'Interactive Mesh Mapping' complete for {catchment} catchment.")
        
        # --- c. Snap gwl monitoring stations to mesh ---
        
        # --- 3d. Snap other features data to mesh

        # --- 3f. Complete mesh map (interactive map from 3a with stations marked etc) ---

        # ==============================================================================
        # SECTION 5: MODEL
        # ==============================================================================

        # ==============================================================================
        # SECTION 6: TRAINING
        # ==============================================================================

        # ==============================================================================
        # SECTION 7: EVALUATION
        # ==============================================================================


# If critical pipeline error, exit with an error code
except Exception as e:
    logger.critical(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)
    sys.exit(1)