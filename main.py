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
from src.preprocessing.graph_construction import build_mesh
from src.visualisation.mapped_visualisations import plot_interactive_mesh
from src.data_ingestion.gwl_data_ingestion import process_station_coordinates, \
    fetch_and_process_station_data, download_and_save_station_readings

# --- 1c. Logging Config ---
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',  # Uncomment for full logging
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

        # --- 2b. Load station measures and metadata from DEFRA API ---

        stations_with_metadata_measures = fetch_and_process_station_data(
            stations_df=stations_with_coords_df,
            base_url=config["global"]["paths"]["defra_station_base_url"],
            output_path=config[catchment]["paths"]["gwl_station_metadata_measures"]
        )

        logger.info(f"Pipeline step 'Pull Hydrological Station Metadata for {catchment}' complete.\n")

        # --- 2c. Load raw gwl timeseris data from DEFRA API ---

        download_and_save_station_readings(
            stations_df=stations_with_metadata_measures,
            start_date=config["global"]["data_ingestion"]["api_start_date"],
            end_date=config["global"]["data_ingestion"]["api_end_date"],
            gwl_data_output_dir=config[catchment]["paths"]["gwl_data_output_dir"]
        )

        logger.info(f"All timeseries groundwater level data saved for {catchment} catchment.")

        # ==============================================================================
        # SECTION 3: PREPROCESSING
        # ==============================================================================

        # ==============================================================================
        # SECTION 4: GRAPH BUILDING
        # ==============================================================================

        # --- 3x. Build Catchment Graph Mesh ---

        output_file_paths = {
            'mesh_nodes_csv_output': config[catchment]['paths']['mesh_nodes_csv_output'],
            'mesh_nodes_gpkg_output': config[catchment]['paths']['mesh_nodes_gpkg_output'],
            'mesh_nodes_shp_output': config[catchment]['paths']['mesh_nodes_shp_output']
        }

        mesh_nodes_table, mesh_nodes_gdf, catchment_polygon = build_mesh(
            shape_filepath=config[catchment]['paths']['gis_catchment_boundary'],
            grid_resolution=config[catchment]['preprocessing']['graph_construction']['grid_resolution'],
            output_paths=output_file_paths
        )

        logger.info(f"Pipeline step 'Build Mesh' complete for {catchment} catchment.")

        # --- 3x. Save interactive map of catchment mesh ---

        mesh_map = plot_interactive_mesh(
            mesh_nodes_gdf=mesh_nodes_gdf,
            catchment_polygon=catchment_polygon,
            map_blue=config['global']['visualisations']['maps']['map_blue'],
            esri=config['global']['visualisations']['maps']['esri'],
            esri_attr=config['global']['visualisations']['maps']['esri_attr'],
            static_output_path=config[catchment]['visualisations']['maps']['static_mesh_map_output'],
            interactive_output_path=config[catchment]['visualisations']['maps']['interactive_mesh_map_output'],
            interactive=config['global']['visualisations']['maps']['display_interactive_map']
        )

        logger.info(f"Pipeline step 'Interactive Mesh Mapping' complete for {catchment} catchment.")

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