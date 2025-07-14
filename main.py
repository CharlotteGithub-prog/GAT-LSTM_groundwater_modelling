# Copyright (c) 2025 Charlotte Wayment
# This file is part of the Dissertation project and is licensed under the MIT License.

### FULL PIPELINE ###

# Expected Processing Time (with API calls = False): ## hrs ## minutes

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
from src.visualisation.mapped_visualisations import plot_interactive_mesh
from src.preprocessing import gwl_preprocessing, gap_imputation, gwl_feature_engineering, \
    hydroclimatic_feature_engineering
from src.data_ingestion import gwl_data_ingestion, static_data_ingestion, \
    timeseries_data_ingestion
from src.graph_building import graph_construction, data_merging

# --- 1c. Logging Config ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # Ensure logging config is respected (override any module logs)
    
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
run_defra_API_calls = config["global"]["pipeline_settings"]["run_defra_api"]
run_camels_API_calls = config["global"]["pipeline_settings"]["run_camels_api"]
run_outlier_detection = config["global"]["pipeline_settings"]["run_outlier_detection"]

# Run full pipeline by catchment
try:
    for catchment in catchments_to_process:

        # ==============================================================================
        # SECTION 2: API DATA INGESTION
        # ==============================================================================

        # --- 2a. Load and convert gwl station location data (DEFRA) ---
        
        stations_with_coords_df = gwl_data_ingestion.process_station_coordinates(
            os_grid_squares=config["global"]["paths"]["gis_os_grid_squares"],
            station_list_input=config[catchment]["paths"]["gwl_station_list"],
            station_list_output=config[catchment]["paths"]["gwl_station_list_with_coords"],
            catchment=catchment
        )

        logger.info(f"Pipeline step 'Process Station Coordinates for {catchment}' complete.\n")
        
        # Only run API calls as needed
        if run_defra_API_calls:  

            # --- 2b. Retrieve station measures and metadata from DEFRA API ---

            stations_with_metadata_measures = gwl_data_ingestion.fetch_and_process_station_data(
                stations_df=stations_with_coords_df,
                base_url=config["global"]["paths"]["defra_station_base_url"],
                output_path=config[catchment]["paths"]["gwl_station_metadata_measures"]
            )

            logger.info(f"Pipeline step 'Pull Hydrological Station Metadata for {catchment}' complete.\n") 
            
            # --- 2c. Retrieve raw gwl timeseris data by station from DEFRA API ---

            gwl_data_ingestion.download_and_save_station_readings(
                stations_df=stations_with_metadata_measures,
                start_date=config["global"]["data_ingestion"]["api_start_date"],
                end_date=config["global"]["data_ingestion"]["api_end_date"],
                gwl_data_output_dir=config[catchment]["paths"]["gwl_data_output_dir"]
            )

            logger.info(f"All timeseries groundwater level data saved for {catchment} catchment.\n")
        
        else:
            
            loaded_csv_path = config[catchment]["paths"]["gwl_station_metadata_measures"]
            stations_with_metadata_measures = pd.read_csv(loaded_csv_path)
            
        # ==============================================================================
        # SECTION 3: MESH BUILDING
        # ==============================================================================

        # --- 3a. Build Catchment Graph Mesh ---
        
        # Select Catchment area from country wide gdf
        graph_construction.define_catchment_polygon(
            england_catchment_gdf_path=config[catchment]['paths']['gis_catchment_boundary'],
            target_mncat=config[catchment]['target_mncat'],
            catchment=catchment,
            polygon_output_path=config[catchment]['paths']['gis_catchment_dir']
        )

        # NB: mesh_nodes_gdf are the centroid coords, mesh_cells_gdf_polygons are polygons for e.g. averaging area
        (mesh_nodes_table, mesh_nodes_gdf, mesh_cells_gdf_polygons,
         catchment_polygon) = graph_construction.build_mesh(
            shape_filepath=config[catchment]['paths']['gis_catchment_dir'],
            output_path=config[catchment]['paths']['mesh_nodes_output'],
            catchment=catchment,
            grid_resolution=config[catchment]['preprocessing']['graph_construction']['grid_resolution']
        )

        logger.info(f"Pipeline step 'Build Mesh' complete for {catchment} catchment.\n")
        
        # --- 3b. Save interactive map of catchment mesh ---
        
        # TODO: ADD STATIONS AS HIGHLIGHTED NODES

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
        
        # ==============================================================================
        # SECTION 4: PREPROCESSING
        # ==============================================================================
        
        # --- 4a. gwl preprocessing ---
        
        # Load timeseries CSVs from API into reference dict, dropping stations with insuffient data
        gwl_time_series_dict = gwl_preprocessing.load_timeseries_to_dict(
            stations_df=stations_with_metadata_measures,
            col_order=config["global"]["data_ingestion"]["col_order"],
            data_dir=config[catchment]["paths"]["gwl_data_output_dir"],
            inclusion_threshold=config[catchment]["preprocessing"]["inclusion_threshold"],
            station_list_output=config[catchment]["paths"]["gwl_station_list_output"],
            catchment=catchment
        )

        logger.info(f"All timeseries data converted to dict for {catchment} catchment.\n")
        
        # Remove outlying and incorrect (user defined: spurious) data points
        
        for station_name, df in gwl_time_series_dict.items():
            gwl_time_series_dict[station_name] = gwl_preprocessing.remove_spurious_data(
                target_df=df,
                station_name=station_name,
                path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
                notebook=True
            )
            
        logger.info(f"Pipeline step 'Remove spurious points' complete for {catchment} catchment.\n")
        
        # Run initial outlier detection and removal
        
        if run_outlier_detection:   
            processed_gwl_time_series_dict = gwl_preprocessing.outlier_detection(
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
        
        daily_data = gwl_preprocessing.resample_daily_average(
            dict=processed_gwl_time_series_dict,
            start_date=config["global"]["data_ingestion"]["api_start_date"],
            end_date=config["global"]["data_ingestion"]["api_end_date"],
            path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            notebook=notebook
        )
        
        logger.info(f"Pipeline step 'Resample to Daily Timesteps' complete for {catchment} catchment.\n")
        
        # Interpolate across small gaps in the ts data (define threshold n/o missing time steps for interpolation
        # eligibility) + Add binary interpolation flag column
        
        daily_data, gaps_list, station_max_gap_lengths_calculated = gwl_preprocessing.handle_short_gaps(
            daily_data=daily_data,
            path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            max_steps=config["global"]["data_ingestion"]["max_interp_length"],
            start_date=config["global"]["data_ingestion"]["api_start_date"],
            end_date=config["global"]["data_ingestion"]["api_end_date"],
            notebook=notebook
        )
            
        logger.info(f"Pipeline step 'Interpolate Short Gaps' complete for {catchment} catchment.\n")

        # Resolve larger gaps in data through a more considered donor imputation process
        
        synthetic_imputation_performace, cleaned_df_dict = gap_imputation.handle_large_gaps(
            df_dict=daily_data,
            gaps_list=gaps_list,
            catchment=catchment,
            spatial_path=config[catchment]["paths"]["gwl_station_list_with_coords"],
            path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            threshold_m=config[catchment]["preprocessing"]["large_catchment_threshold_m"],
            radius=config["global"]["preprocessing"]["radius"],
            output_path=config[catchment]["visualisations"]["corr_dist_score_scatters"],
            threshold=config[catchment]["preprocessing"]["dist_corr_score_threshold"],
            predefined_large_gap_lengths=config["global"]["preprocessing"]["gap_lengths_days"] ,
            max_imputation_length_threshold=config["global"]["preprocessing"]["max_imputation_threshold"],
            min_around=config["global"]["preprocessing"]["min_data_points_around_gap"],
            station_max_gap_lengths=station_max_gap_lengths_calculated,
            k_decay=config[catchment]["preprocessing"]["dist_corr_score_k_decay"],
            random_seed=config["global"]["pipeline_settings"]["random_seed"]
        )
            
        logger.info(f"Pipeline step 'Interpolate Long Gaps' complete for {catchment} catchment.\n")
        
        # Add lagged ground water measurement features (1-7 days, lagged before trimming for full coverage)
        
        df_with_lags = gwl_feature_engineering.build_lags(
            df_dict=cleaned_df_dict,
            catchment=catchment
        )

        # define sinusoidal features for seasonality (both sine and cosine for performance)
        
        df_with_seasons = gwl_feature_engineering.build_seasonality_features(
            df_dict=df_with_lags,
            catchment=catchment
        )

        logger.info(f"Pipeline step 'Build Seasons and Lags' complete for {catchment} catchment.\n")
        
        # Clean up final dataframes and trim to the temporal bounds of the GAT-LSTM model
        
        trimmed_df_dict = gwl_feature_engineering.trim_and_save(
            df_dict=df_with_seasons,
            model_start_date=config['global']['data_ingestion']['model_start_date'],
            model_end_date=config['global']['data_ingestion']['model_end_date'],
            trimmed_output_dir=config[catchment]["paths"]["trimmed_output_dir"],
            ts_path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            notebook=notebook
        )
        
        logger.info(f"Pipeline step 'Trim GWL to Model Bounds' complete for {catchment} catchment.\n")
        
        # --- 4b. Preprocess Static Features ---
        
        # Land Cover [UKCEH LCM2023]
        
        agg_land_cover_df = static_data_ingestion.load_land_cover_data(
            tif_path=config[catchment]['paths']['raw_land_cover_path'],
            csv_path=config[catchment]['paths']['land_cover_csv_path'],
            catchment=catchment,
            shape_filepath=config[catchment]['paths']['gis_catchment_dir']
        )
        
        logger.info(f"1km granularity land use data processed for {catchment} catchment.\n")
        
        # Elevation [DIGIMAPS (via OS Terrain 5 / Terrain 50)]
        
        elevation_gdf_polygon, clipped_dtm = static_data_ingestion.load_process_elevation_data(
            dir_path=config[catchment]['paths']['elevation_dir_path'],
            csv_path=config[catchment]['paths']['elevation_tif_path'],
            catchment_gdf=catchment_polygon,
            mesh_cells_gdf_polygons=mesh_cells_gdf_polygons,
            catchment=catchment,
            elev_max=config[catchment]['preprocessing']['catchment_max_elevation'],
            elev_min=config[catchment]['preprocessing']['catchment_min_elevation'],
            output_geojson_dir=config[catchment]['paths']['output_polygon_dir'],
            grid_resolution=config[catchment]['preprocessing']['graph_construction']['grid_resolution']
        )
        
        logger.info(f"Elevation data aggregated to node level for {catchment} catchment.\n")
        
        # Slope [Derived from DEMS] + Edge Direction Weights (Derived from Slope -> modularise?)
        
        slope_gdf, directional_edge_weights = static_data_ingestion.derive_slope_data(
            high_res_raster=clipped_dtm,
            mesh_cells_gdf_polygons=mesh_cells_gdf_polygons,
            catchment=catchment,
            direction_output_path=config[catchment]['paths']['direction_edge_weights_path'],
            slope_output_path=config[catchment]['paths']['slope_path']
        )
        
        logger.info(f"Slope and aspect data derived at node level for {catchment} catchment.\n")
        
        # Soil type [CEH's Grid-to-Grid soil maps / HOST soil classes / CAMELS-GB / BFIHOST]
        
        # Aquifer Properties (tbd - depth? type? transmissivity? storage coefficientet?) [DEFRA/BGS]
        
        # Geological Maps [DIGIMAPS (BGS data via Geology Digimap / Direct)]
        
        # Permeability [BGS]
        
        # Distance from River (Derived) [DEFRA/DIGIMAP]
        
        # --- 4c. Preprocess Time Series Features ---
        
        # Precipitation (Daily Rainfall, mm, catchment total) [HadUK-GRID]
        
        timeseries_data_ingestion.load_rainfall_data(
            rainfall_dir=config[catchment]["paths"]["rainfall_filename_dir"],
            shape_filepath=config[catchment]["paths"]["gis_catchment_dir"],
            processed_output_dir=config[catchment]["paths"]["rainfall_processed_output_dir"],
            fig_path=config[catchment]["paths"]["rainfall_fig_path"],
            required_crs=27700,
            catchment=catchment
        )
            
        logger.info(f"Pipeline step 'Load Rainfall Data' complete for {catchment} catchment.\n")
        
        # Surface Pressure (Daily Mean, hPa, catchment average) [HadUK-Grid]
        
        timeseries_data_ingestion.load_era5_land_data(
            catchment=catchment,
            shape_filepath=config[catchment]['paths']['gis_catchment_dir'],
            required_crs=27700,
            cdsapi_path=config["global"]["paths"]["CDSAPI_path"],
            start_date=config["global"]["data_ingestion"]["model_start_date"],
            end_date=config["global"]["data_ingestion"]["model_end_date"],
            run_era5_land_api=config["global"]["pipeline_settings"]["run_era5_land_api"],
            raw_output_dir=config[catchment]["paths"]["sp_raw_output_dir"],
            processed_output_dir=config[catchment]["paths"]["sp_processed_output_dir"],
            csv_path=config[catchment]["paths"]["sp_csv_path"],
            fig_path=config[catchment]["paths"]["sp_fig_path"],
            era5_feat='sp',
            era5_long='surface_pressure',
            feat_name='surface_pressure',
            aggregation_type='mean'
        )
            
        logger.info(f"Pipeline step 'Load Surface Pressure Data' complete for {catchment} catchment.\n")
        
        # Actual Evapotranspiration [ERA5-Land AET]
        
        timeseries_data_ingestion.load_era5_land_data(
            catchment=catchment,
            shape_filepath=config[catchment]['paths']['gis_catchment_dir'],
            required_crs=27700,
            cdsapi_path=config["global"]["paths"]["CDSAPI_path"],
            start_date=config["global"]["data_ingestion"]["model_start_date"],
            end_date=config["global"]["data_ingestion"]["model_end_date"],
            run_era5_land_api=config["global"]["pipeline_settings"]["run_era5_land_api"],
            raw_output_dir=config[catchment]["paths"]["aet_raw_output_dir"],
            processed_output_dir=config[catchment]["paths"]["aet_processed_output_dir"],
            csv_path=config[catchment]["paths"]["aet_csv_path"],
            fig_path=config[catchment]["paths"]["aet_fig_path"],
            era5_feat='e',
            era5_long='total_evaporation',
            feat_name='aet',
            aggregation_type='sum'
        )
            
        logger.info(f"Pipeline step 'Load Actual Evapotranspiration Data' complete for {catchment} catchment.\n")
        
        # 2m Surface Temperature (Daily Mean Temperature, °C, catchment average) [HadUK-GRID]
        
        timeseries_data_ingestion.load_era5_land_data(
            catchment=catchment,
            shape_filepath=config[catchment]['paths']['gis_catchment_dir'],
            required_crs=27700,
            cdsapi_path=config["global"]["paths"]["CDSAPI_path"],
            start_date=config["global"]["data_ingestion"]["model_start_date"],
            end_date=config["global"]["data_ingestion"]["model_end_date"],
            run_era5_land_api=config["global"]["pipeline_settings"]["run_era5_land_api"],
            raw_output_dir=config[catchment]["paths"]["2t_raw_output_dir"],
            processed_output_dir=config[catchment]["paths"]["2t_processed_output_dir"],
            csv_path=config[catchment]["paths"]["2t_csv_path"],
            fig_path=config[catchment]["paths"]["2t_fig_path"],
            era5_feat='2t',
            era5_long='2m_temperature',
            feat_name='2m_temp',
            aggregation_type='mean'
        )
            
        logger.info(f"Pipeline step 'Load 2m Surface Temp Data' complete for {catchment} catchment.\n")
        
        # River Flow / Streamflow / River Stage [DEFRA / NRFA]
        
        
        # Others from HAD-UK (CEDA)?
        
        """
        ERA5-Land:
            - snowLying: snow depth / presence
        """
        
        # --- 4d. Derived Hydrogeological Feature Engineering ---

        # 30/60 day rainfall rolling average + 7 day rainfall lags [DERIVED]
        
        rainfall_df = hydroclimatic_feature_engineering.derive_rainfall_features(
            csv_dir=config[catchment]["paths"]["rainfall_processed_output_dir"],
            processed_output_dir=config[catchment]["paths"]["rainfall_processed_output_dir"],
            start_date=config["global"]["data_ingestion"]["model_start_date"],
            end_date=config["global"]["data_ingestion"]["model_end_date"],
            config_path="config/project_config.yaml",
            catchment=catchment
        )
            
        logger.info(f"Pipeline step 'Derive Rainfall Lag and Averages' complete for {catchment} catchment.\n")
        
        # Calculate ET / temperature rolling averages? [DERIVED]
        
        # Pour point (catchment) by node -> see notion notes (important to consider)
        
            # - Use flow accumulation from the DEM (e.g., richdem, whitebox, or TauDEM)
            # - Aggregate this to mesh by zonal mean/sum (most likely sum? Decide + Justify).
        
        # ==============================================================================
        # SECTION 5: GRAPH BUILDING
        # ==============================================================================
        
        # --- 5a. Snap GWL monitoring station features to mesh nodes ---
        
        station_node_mapping = data_merging.snap_stations_to_mesh(
            station_list_path=config[catchment]["paths"]["gwl_station_list_output"],
            polygon_geometry_path=config[catchment]['paths']['output_polygon_dir'],
            output_path=config[catchment]["paths"]["snapped_station_node_mapping"],
            mesh_nodes_gdf=mesh_nodes_gdf,
            catchment=catchment
        )
        
        # mesh_nodes_gdf = mesh_nodes_gdf.merge(
        #     # Merge in all gwl data (from: data/02_processed/eden/gwl_timeseries/{station_name}_trimmed.csv)
        #     # Will only merge in with nodes wiht a station (approx. 15/2750), rest will be NaN
        # )

        # --- 5b. Snap static features to mesh nodes ---
        
        # Snap Land Cover to Mesh
        
        merged_gdf_nodes_landuse = mesh_nodes_gdf.merge(
            agg_land_cover_df[['easting', 'northing', 'land_cover_code']],
            on=['easting', 'northing'],
            how='left'  # left join to keep all centroids, even NaN
        )
        
        logger.info(f"Land cover data snapped to mesh nodes (centroids).\n")
        
        # Snap Elevation to Mesh
        
        merged_gdf_nodes_elevation = merged_gdf_nodes_landuse.merge(
            elevation_gdf_polygon[['node_id', 'mean_elevation', 'polygon_geometry']],
            on='node_id',
            how='left'  # left join to keep all centroids, even NaN
        )
        
        logger.info(f"Elevation data snapped to mesh nodes (centroids).\n")
        
        # Snap Slope to Mesh [TODO: THIS IS CURRENTLY ACTING AS FINAL STATIC DF, UPDATE IN FUTURE]
        
        # merged_gdf_nodes_slope = merged_gdf_nodes_elevation.merge(
        static_features = merged_gdf_nodes_elevation.merge(
            slope_gdf[['node_id', 'mean_slope_degrees', 'mean_aspect_sin', 'mean_aspect_cos']],
            on='node_id',
            how='left'  # left join to keep all centroids, even NaN
        )

        logger.info(f"Slope degrees and sinusoidal aspect data snapped to mesh nodes (centroids).\n")
        
        # [FUTURE] Snap Soil type to Mesh
        
        # [FUTURE] Snap Aquifer Properties to Mesh
        
        # [FUTURE] Snap Geology Maps to Mesh
        
        # [FUTURE] Snap Permeability to Mesh
        
        # [FUTURE] Snap Distance from River to Mesh
        
        # Finalise final_static_df for merge
        
        final_static_df = data_merging.reorder_static_columns(static_features)
        logger.info(f"Full static feature dataframe finalised and ready to merge into main model dataframe.\n")
        
        # --- 5c. Snap dynamic (timeseries) features to mesh nodes (equal across all for catchment) and daily timestep ---
        
        # Snap Precipitation, Lags and Averages to timestep

        merged_ts_precipitation = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["rainfall_csv_path"],
            feature='all_precipitation'
        )
        
        logger.info(f"Precipitation and derived data snapped to graph timesteps (daily aggregates).\n")
        
        # Snap 2m Temperature to timestep

        merged_ts_tsm = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["2t_csv_path"],
            feature='2m_temperature',
            timeseries_df=merged_ts_precipitation
        )
        
        logger.info(f"2m Temperature Data snapped to graph timesteps (daily aggregate).\n")
        
        # Snap AET to timestep

        merged_ts_aet = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["aet_csv_path"],
            feature='aet',
            timeseries_df=merged_ts_tsm
        )
        
        merged_ts_aet = hydroclimatic_feature_engineering.transform_aet_data(merged_ts_aet, catchment)
        
        logger.info(f"Actual evapotranspiration data snapped to graph timesteps (daily aggregate).\n")
        
        # Snap Surface Pressure to timestep

        final_merged_ts_df = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["sp_csv_path"],
            feature='surface_pressure',
            timeseries_df=merged_ts_aet
        )
        
        logger.info(f"Surface pressure data snapped to graph timesteps (daily aggregate).\n")
        
        # --- 5d. Merge static and timeseries df's into main df ---
        
        save_path = config[catchment]["paths"]["final_df_path"] + 'final_timeseries_df.csv'
        final_merged_ts_df.to_csv(save_path)
        
        logger.info(f"Final merged time series dataframe saved to {save_path}")
        logger.info(f"Full timeseries feature dataframe finalised and ready to merge into main model dataframe.\n")

        # --- 5e. KEY: HANDLE GWL MASKS ---
        
        # --- 5f. Incorporate Edge Weighting? (maybe move later) ---

        # --- 5g. (Optional) Visualise complete mesh map with stations and other features ---
        # Purpose: Verify final graph structure and feature distribution visually.
        # Action: Create an interactive map showing the mesh, GWL stations, and other snapped data points.
        
        # ==============================================================================
        # SECTION 6: TRAINING/TESTING SPLIT
        # ==============================================================================
        
        # --- 6a. Split the data temporally ---
        
        """ 6. THEN DO FULL FIRST ITERATION OF MODEL FOR WEDNESDAY... """
        
        # E.g., 70/15/15 train/val/test by date -> Key:Must be segregated by time, not mixed
        # All subsequent steps must be done AFTER split to avoid data leakage
        
        # --- 6b. Standardisation of all numeric features and round all numeric to 3-4dp ---
        
        
        # Using from sklearn.preprocessing import StandardScaler
        
        # --- 6c. One-Hot Encode Categorical Features ---
        
        # Using from sklearn.preprocessing import OneHotEncoder
        
        # --- 6d. Weight imbalanced classes (land_use) ---
        
        # Using from sklearn.utils.class_weight import compute_class_weight

        # --- 6e. Define Graph Adjacency Matrix (edge_index -> 8 nearest neighbours) ---
        # Purpose: Establish connections between mesh nodes.
        # Action: Define criteria for edges (e.g., k-nearest neighbors, distance threshold, hydrological connectivity).
        # Action: Construct the adjacency matrix (A) for the graph.
        # Output: Adjacency matrix or edge_index for PyTorch Geometric.

        # --- 6f. Create Graph Data Object / Input Tensors ---
        # Purpose: Assemble all graph components (nodes, edges, features, targets) into a format suitable for the GNN framework.
        # Action: Split data into training, validation, and test sets based on time (e.g., 70/15/15 chronological split).
        # Action: Generate graph snapshots/sequences for the GNN's input.
        # Output: PyTorch Geometric Data objects or DGL graphs, including node features (X), edge index (edge_index), and target GWL values (Y) for observed nodes.

        # ==============================================================================
        # SECTION 7: MODEL
        # ==============================================================================
        
        # --- 7. Goal: GAT-LSTM (using PyTorch Geometric) ---
        # When constructing the PyTorch Geometric Data objects (one per timestep), pass edge_index and edge_attr as separate arguments to the Data
        # constructor, alongside the x (node features) and y (targets).
        
        # --- 7a. Define Graph Neural Network Architecture ---
        # Goal: Implement a GAT-LSTM (Graph Attention Network + Long Short-Term Memory).
        # Action: Define the GNN layers (e.g., GATConv for spatial message passing, LSTM for temporal learning).
        # Action: Specify input/output dimensions, hidden layer sizes, activation functions, dropout rates.
        # Output: A PyTorch nn.Module or similar model class.

        # --- 7b. Define Loss Function ---
        # Action: Choose an appropriate loss function for regression (e.g., Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss).
        # Action: Consider masking the loss calculation to only evaluate predictions at observed GWL stations if non-interpolated.

        # --- 7c. Define Optimiser ---
        # Action: Select an optimiser (e.g., Adam, SGD).
        # Action: Configure learning rate and other optimiser parameters.

        # ==============================================================================
        # SECTION 8: TRAINING
        # ==============================================================================
        
        # --- 8a. Implement Training Loop ---
        # Action: Iterate over epochs.
        # Action: For each epoch, iterate over mini-batches of graph data.
        # Action: Perform forward pass through the model.
        # Action: Calculate loss.
        # Action: Perform backward pass and optimize weights.
        # Action: Implement gradient clipping if necessary.

        # --- 8b. Implement Validation Loop ---
        # Action: Periodically evaluate model performance on the validation set.
        # Action: Monitor validation loss/metrics for early stopping.

        # --- 8c. Model Checkpointing and Logging ---
        # Action: Save best performing model weights based on validation metrics.
        # Action: Log training and validation metrics (e.g., using TensorBoard, MLflow, or custom logging).

        # ==============================================================================
        # SECTION 9: EVALUATION
        # ==============================================================================

        # --- 9a. Final Model Evaluation ---
        # Action: Load the best trained model.
        # Action: Evaluate its performance on the unseen test set.
        # Action: Calculate key metrics (e.g., RMSE, MAE, R-squared, Nash-Sutcliffe Efficiency for hydrology).
        # Output: Quantitative evaluation results.

        # --- 9b. Visualisation of Predictions ---
        # Action: Plot actual vs. predicted GWL time series for selected stations/nodes.
        # Action: Create animated maps showing predicted GWL changes over time across the mesh (if using interpolated pseudo-labels).

        # --- 9c. Error Analysis ---
        # Action: Identify systematic errors or biases in predictions (e.g., over/under-prediction during peaks/troughs).
        # Action: Explore reasons for poor performance at specific stations/nodes or time periods.
        
        # ==============================================================================
        # SECTION 10: INTERPRETATION & HYDROLOGICAL INSIGHTS
        # ==============================================================================

        # --- 10a. Feature Importance Analysis (Global & Local) ---
        # Purpose: Understand which input features (climate, static, lagged GWL) drive predictions.
        # Action: Apply SHAP (SHapley Additive exPlanations) or similar methods (e.g., Permutation Importance) to identify global feature importance.
        # Action: For specific predictions or events, apply SHAP/LIME to understand local feature contributions.
        # Output: Feature importance plots, tables.

        # --- 10b. Spatial Relationship Interpretation (Attention/GNNExplainer) ---
        # Purpose: Uncover how the GNN leverages spatial connections and neighbors.
        # Action: If using GAT, analyze learned attention weights to understand influence of neighboring nodes.
        # Action: Apply GNNExplainer (or similar graph-specific XAI methods) to identify critical subgraphs for specific node predictions.
        # Output: Visualizations of influential neighbors/connections on maps, attention heatmaps.

        # --- 10c. Analysis of Pseudo-Ungauged Generalisation ---
        # Purpose: Evaluate the model's ability to predict at nodes masked during training.
        # Action: Compare predictions at "held-out" (proxy-ungauged) nodes against their interpolated pseudo-ground-truth.
        # Action: Analyze performance metrics specifically for these nodes.
        # Action: Visualise the spatial distribution of errors for these ungauged nodes.

        # --- 10d. Hydrological Interpretation and Discussion ---
        # Purpose: Translate model insights back into hydrogeological understanding.
        # Action: Discuss consistency of feature importances and spatial influences with known hydrological principles (e.g., lag times, flow paths, aquifer properties).
        # Action: Provide explanations for observed model behaviors during specific hydrological events (droughts, floods).
        # Output: Written analysis and discussion points for dissertation.

        # --- 10e. Design Trade-offs Analysis (Optional, if time allows) ---
        # Purpose: Evaluate the impact of choices in graph construction.
        # Action: Compare results/interpretations from different graph resolutions (e.g., 500m vs 1000m) or edge definitions (e.g., KNN vs distance threshold).
        # Output: Comparative analysis.

# If critical pipeline error, exit with an error code
except Exception as e:
    logger.critical(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)
    sys.exit(1)