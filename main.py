# Copyright (c) 2025 Charlotte Wayment
# This file is part of the Dissertation project and is licensed under the MIT License.

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================

# --- 1a. Library Imports ---
import os
import sys
import logging

# --- 1b. Project Imports ---
from src.utils.config_loader import load_project_config
from src.preprocessing.graph_construction import build_mesh
from src.visualisation.mapped_visualisations import plot_interactive_mesh

# --- 1c. Logging Config ---
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',  # Uncomment for full logging
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
config = load_project_config(config_path="config/eden_project_config.yaml")

# ==============================================================================
# SECTION 2: DATA INGESTION
# ==============================================================================

# --- 1d. Data Loading ---

# ==============================================================================
# SECTION 3: PREPROCESSING (Inc. GRAPH BUILDING)
# ==============================================================================

# --- 3x. Build Graph Mesh ---

shape_filepath = config['paths']['gis_eden_catchment_boundary']
grid_resolution = config['preprocessing']['graph_construction']['grid_resolution']

output_file_paths = {
    'mesh_nodes_csv_output': config['paths']['mesh_nodes_csv_output'],
    'mesh_nodes_gpkg_output': config['paths']['mesh_nodes_gpkg_output'],
    'mesh_nodes_shp_output': config['paths']['mesh_nodes_shp_output']
}

mesh_nodes_table, mesh_nodes_gdf, catchment_polygon = build_mesh(
    shape_filepath=shape_filepath,
    grid_resolution=grid_resolution,
    output_paths=output_file_paths
)

logger.info("Pipeline step 'Build Mesh' complete.")

# --- 3x. Save interactive map of catchment mesh ---

mesh_map = plot_interactive_mesh(
    mesh_nodes_gdf=mesh_nodes_gdf,
    catchment_polygon=catchment_polygon,
    map_blue=config['visualisations']['maps']['map_blue'],
    esri=config['visualisations']['maps']['esri'],
    esri_attr=config['visualisations']['maps']['esri_attr'],
    static_output_path=config['visualisations']['outputs']['static_mesh_map_output'],
    interactive_output_path=config['visualisations']['outputs']['interactive_mesh_map_output'],
    interactive=config['visualisations']['maps']['display_interactive_map']
)

logger.info("Pipeline step 'Interactive Mesh Mapping' complete.")

# ==============================================================================
# SECTION 4: MODEL
# ==============================================================================

# ==============================================================================
# SECTION 5: TRAINING
# ==============================================================================

# ==============================================================================
# SECTION 6: EVALUATION
# ==============================================================================