# This file is part of the Dissertation project and is licensed under the MIT License.

### FULL PIPELINE ###
# Expected Processing Time for Eden Catchment (with API calls = False): ## hrs ## minutes
#        - 00 hrs 05 minutes 58 seconds -> to end of section 6b

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================

# --- 1a. Library Imports ---
import os
import sys
import torch
import random
import joblib
import logging
import numpy as np
import pandas as pd

# --- 1b. Project Imports ---
from src.utils.config_loader import load_project_config
from src.visualisation import mapped_visualisations
from src.preprocessing import gwl_preprocessing, gap_imputation, gwl_feature_engineering, \
    hydroclimatic_feature_engineering, data_partitioning, model_feature_engineering
from src.data_ingestion import gwl_data_ingestion, static_data_ingestion, \
    timeseries_data_ingestion
from src.graph_building import graph_construction, data_merging
from src.model import model_building
from src.training import model_training

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

# --- 1d. Set up root directory paths in config ---

raw_data_root = config["global"]["paths"]["raw_data_root"]

# Update all values in global paths
for key, val in config["global"]["paths"].items():
    if isinstance(val, str):
        config["global"]["paths"][key] = val.format(raw_data_root=raw_data_root)

# Update all catchment paths
catchments_to_process = config["global"]["pipeline_settings"]["catchments_to_process"]
for catchment in catchments_to_process:
    for key, val in config[catchment]["paths"].items():
        if isinstance(val, str):
            config[catchment]["paths"][key] = val.format(raw_data_root=raw_data_root)
            
# --- 1e. Set up seeding to define global states ---
random_seed = config["global"]["pipeline_settings"]["random_seed"]
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 1f. Define catchment(s) and API calls to Process --
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

        mesh_map = mapped_visualisations.plot_interactive_mesh(
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
                pred_frequency=config["global"]["pipeline_settings"]["prediction_resolution"],
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
        
        # Aggregate to specified timestep 
        
        pred_frequency=config["global"]["pipeline_settings"]["prediction_resolution"]
        
        timestep_data = gwl_preprocessing.resample_timestep_average(
            gwl_data_dict=processed_gwl_time_series_dict,
            start_date=config["global"]["data_ingestion"]["api_start_date"],
            end_date=config["global"]["data_ingestion"]["api_end_date"],
            path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            pred_frequency=pred_frequency,
            notebook=notebook
        )
        
        logger.info(f"Pipeline step 'Resample to {pred_frequency} Timesteps' complete for {catchment} catchment.\n")
        
        # Interpolate across small gaps in the ts data (define threshold n/o missing time steps for interpolation
        # eligibility) + Add binary interpolation flag column
        
        timestep_data, gaps_list, station_max_gap_lengths_calculated = gwl_preprocessing.handle_short_gaps(
            timestep_data=timestep_data,
            path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            max_steps=config["global"]["data_ingestion"]["max_interp_length"],
            start_date=config["global"]["data_ingestion"]["api_start_date"],
            end_date=config["global"]["data_ingestion"]["api_end_date"],
            pred_frequency=pred_frequency,
            notebook=notebook
        )
            
        logger.info(f"Pipeline step 'Interpolate Short Gaps' complete for {catchment} catchment.\n")

        # Resolve larger gaps in data through a more considered donor imputation process
        
        synthetic_imputation_performace, cleaned_df_dict = gap_imputation.handle_large_gaps(
            df_dict=timestep_data,
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
            pred_frequency=pred_frequency,
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
            catchment=catchment,
            pred_frequency=config["global"]["pipeline_settings"]["prediction_resolution"]
        )

        logger.info(f"Pipeline step 'Build Seasons and Lags' complete for {catchment} catchment.\n")
        
        # Clean up final dataframes and trim to the temporal bounds of the GAT-LSTM model
        
        trimmed_df_dict = gwl_feature_engineering.trim_and_save(
            df_dict=df_with_seasons,
            model_start_date=config['global']['data_ingestion']['model_start_date'],
            model_end_date=config['global']['data_ingestion']['model_end_date'],
            trimmed_output_dir=config[catchment]["paths"]["trimmed_output_dir"],
            ts_path=config[catchment]["visualisations"]["ts_plots"]["time_series_gwl_output"],
            notebook=notebook,
            catchment=catchment,
            init_with_dip_value=config[catchment]['model']['architecture']['initialise_with_dipped']
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
            elevation_geojson_path=config[catchment]['paths']['elevation_geojson_path'],
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
        
        # Geological Maps (including bedrock and superficial permeability) [DIGIMAPS Geology]
        
        mesh_geology_df = static_data_ingestion.load_and_process_geology_layers(
            base_dir=config[catchment]["paths"]["geology_dir"],
            mesh_crs=mesh_cells_gdf_polygons.crs,
            columns_of_interest={"bedrock": ["RCS_ORIGIN"], "superficial": ["RCS_D"]},
            mesh_cells_gdf_polygons=mesh_cells_gdf_polygons,
            perm_dir=config[catchment]["paths"]["permeability_dir"],
            geo_output_dir=config[catchment]["paths"]["geology_df"],
            catchment=catchment
        )
        
        logger.info(f"Pipeline step 'Load and preprocess geology data' complete for {catchment} catchment.\n")
        
        # Plot geology features as interactive map
        
        feature_category_colors, feature_category_labels, layer_labels = static_data_ingestion.get_geo_feats()
        geology_map = mapped_visualisations.plot_geology_layers_interactive(
            mesh_geology_df=mesh_geology_df,
            catchment_polygon=catchment_polygon,
            esri=config['global']['visualisations']['maps']['esri'],
            esri_attr=config['global']['visualisations']['maps']['esri_attr'],
            output_path=config[catchment]["visualisations"]["maps"]["interactive_mesh_map_output"],
            feature_columns=['geo_superficial_type','geo_bedrock_type'],
            category_colors=feature_category_colors,
            category_labels=feature_category_labels,
            map_blue=config['global']['visualisations']['maps']['map_blue'],
            layer_labels=layer_labels
        )
        
        """ PREVIOUSLY TO mesh_geology_df """
        
        # Aquifer Productivity [BGS 625k Hydrogeological Data]

        productivity_gdf = static_data_ingestion.ingest_and_process_productivity(
            productivity_dir=config[catchment]["paths"]["productivity_dir"],
            csv_path=config[catchment]['paths']['productivity_csv_path'],
            mesh_cells_gdf_polygons=mesh_cells_gdf_polygons,
            catchment=catchment   
        )
        
        logger.info(f"Pipeline step 'Load and preprocess aquifer productvity data' complete for {catchment} catchment.\n")
        
        # Distance from River (Derived) [OS Open Rivers]

        dist_to_river_gdf, rivers_real = static_data_ingestion.derive_distance_to_river(
            rivers_dir=config[catchment]["paths"]["rivers_dir"],
            csv_path=config[catchment]['paths']['rivers_csv_path'],
            catchment=catchment,
            mesh_cells_gdf_polygons=mesh_cells_gdf_polygons,
            catchment_polygon=catchment_polygon,
            mesh_nodes_gdf=mesh_nodes_gdf
        )

        logger.info(f"Pipeline step 'Derive distance from river' complete for {catchment} catchment.\n")
        
        # Soil Hydrology [HOST, via Digimaps]

        soil_hydrology_df = static_data_ingestion.load_process_soil_hydrology(
            soil_dir=config[catchment]["paths"]["soil_dir"],
            csv_path=config[catchment]['paths']['soil_csv_path'],
            catchment=catchment,
            mesh_cells_gdf_polygons=mesh_cells_gdf_polygons,
            catchment_polygon=catchment_polygon,
            mesh_nodes_gdf=mesh_nodes_gdf
        )

        logger.info(f"Pipeline step 'Load Soil Hydrology Data' complete for {catchment} catchment.\n")
        
        # [FUTURE] Depth to bedrock [BGS]
        
        """ ADD IN ONCE GIVEN LICENCE """
        
        # [SKIPPED] Superficial Thickness [BGS] - Skipped due to insufficient data coverage
        # [SKIPPED] Soil Type [CEH's Grid-to-Grid soil maps] - Skipped due to insufficient data coverage
        # [FUTURE] Aquifer Properties (tbd - depth? type? transmissivity? storage coefficientet?) [DEFRA/BGS]
        # [FUTURE] Gridded infiltration rates / hydraulic conductivity - Skipped for now due to insufficient data
        
        # --- 4c. Preprocess Time Series Features ---
        
        # Precipitation (Timestep Frequency Rainfall, mm, catchment total) [HadUK-GRID]
        
        timeseries_data_ingestion.load_rainfall_data(
            rainfall_dir=config[catchment]["paths"]["rainfall_filename_dir"],
            shape_filepath=config[catchment]["paths"]["gis_catchment_dir"],
            processed_output_dir=config[catchment]["paths"]["rainfall_processed_output_dir"],
            fig_path=config[catchment]["paths"]["rainfall_fig_path"],
            required_crs=27700,
            pred_frequency=config["global"]["pipeline_settings"]["prediction_resolution"],
            catchment=catchment
        )
            
        logger.info(f"Pipeline step 'Load Rainfall Data' complete for {catchment} catchment.\n")
        
        # Surface Pressure (Timestep Frequency Mean, hPa, catchment average) [HadUK-Grid]
        
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
            csv_name=f'surface_pressure_{pred_frequency}_catchment_mean.csv',
            fig_path=config[catchment]["paths"]["sp_fig_path"],
            pred_frequency=pred_frequency,
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
            csv_name=f'aet_{pred_frequency}_catchment_sum.csv',
            fig_path=config[catchment]["paths"]["aet_fig_path"],
            pred_frequency=pred_frequency,
            era5_feat='e',
            era5_long='total_evaporation',
            feat_name='aet',
            aggregation_type='sum'
        )
            
        logger.info(f"Pipeline step 'Load Actual Evapotranspiration Data' complete for {catchment} catchment.\n")
        
        # [FUTURE] Add PET with almost identical pipeline for higher level climate demand
        # [FUTURE] Derive ET_deficit using PET - AET to capture cumulative drying and recharge patterns
        
        # 2m Surface Temperature (Timestep Frequency Mean Temperature, °C, catchment average) [HadUK-GRID]
        
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
            csv_name=f'2m_temp_{pred_frequency}_catchment_mean.csv',
            fig_path=config[catchment]["paths"]["2t_fig_path"],
            pred_frequency=pred_frequency,
            era5_feat='2t',
            era5_long='2m_temperature',
            feat_name='2m_temp',
            aggregation_type='mean'
        )
            
        logger.info(f"Pipeline step 'Load 2m Surface Temp Data' complete for {catchment} catchment.\n")
        
        # DEFRA Hydrology API timestep frequency total streamflow at station closest to catchment ouflow (lumped hydrological modelling, m^3/s)

        stream_flow_df = timeseries_data_ingestion.download_and_save_flow_data(
            station_csv=config[catchment]["paths"]["stream_flow_station"],
            start_date=config["global"]["data_ingestion"]["model_start_date"],
            end_date=config["global"]["data_ingestion"]["model_end_date"],
            output_dir=config[catchment]["paths"]["stream_flow_csv"],
            pred_frequency=pred_frequency,
            catchment=catchment
        )

        logger.info(f"Pipeline step 'Ingest Streamflow Data' complete for {catchment} catchment.\n")
        
        # [FUTURE] Others from HAD-UK (CEDA) / ERA5-Land?
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
            pred_frequency=config["global"]["pipeline_settings"]["prediction_resolution"],
            catchment=catchment
        )
            
        logger.info(f"Pipeline step 'Derive Rainfall Lag and Averages' complete for {catchment} catchment.\n")
        
        # [FUTURE] Calculate ET / temperature rolling averages? [DERIVED]
        # [FUTURE] Pour point (catchment) by node -> see notion notes (important to consider)
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
        
        # Load station and final df directories
        
        station_dir = config[catchment]["paths"]["trimmed_output_dir"]
        node_mapping_dir = config[catchment]["paths"]["snapped_station_node_mapping"]
        
        # Load and process gwl data for subsequent merge
        
        gwl_data = data_merging.load_gwl_data_for_merge(
            station_dir=station_dir,
            node_mapping_dir=node_mapping_dir
        )

        # --- 5b. Snap static features to mesh nodes ---
        
        # Snap Land Cover to Mesh
        
        merged_gdf_nodes_landuse = mesh_nodes_gdf.merge(
            agg_land_cover_df[['easting', 'northing', 'land_cover_code']],
            on=['easting', 'northing'],
            how='left'  # left join to keep all centroids, even if NaN
        )
        
        logger.info(f"Land cover data snapped to mesh nodes (centroids).\n")
        
        # Snap Elevation to Mesh
        
        merged_gdf_nodes_elevation = merged_gdf_nodes_landuse.merge(
            elevation_gdf_polygon[['node_id', 'mean_elevation', 'polygon_geometry']],
            on='node_id',
            how='left'  # left join to keep all centroids, even if NaN
        )
        
        logger.info(f"Elevation data snapped to mesh nodes (centroids).\n")
        
        # Snap Geology Maps to Mesh
        
        merged_gdf_nodes_geology = merged_gdf_nodes_elevation.merge(
            mesh_geology_df[['geo_bedrock_type', 'geo_superficial_type', 'bedrock_flow_type',
                             'bedrock_perm_avg', 'superficial_flow_type', 'superficial_perm_avg',
                             'node_id']],
            on='node_id',
            how='left'  # left join to keep all centroids, even if NaN
        )
        
        logger.info(f"Geology data snapped to mesh nodes (centroids).\n")
        
        # Snap Slope to Mesh
        
        merged_gdf_nodes_slope = merged_gdf_nodes_geology.merge(
            slope_gdf[['node_id', 'mean_slope_degrees', 'mean_aspect_sin', 'mean_aspect_cos']],
            on='node_id',
            how='left'  # left join to keep all centroids, even if NaN
        )

        logger.info(f"Slope degrees and sinusoidal aspect data snapped to mesh nodes (centroids).\n")
        
        # Snap Soil Hydrology to Mesh
        
        merged_gdf_nodes_soil = merged_gdf_nodes_slope.merge(
            soil_hydrology_df[['node_id', 'HOST_soil_class']],
            on='node_id',
            how='left'  # left join to keep all centroids, even if NaN
        )
        
        logger.info(f"Soil Hydrology data snapped to mesh nodes (centroids).\n")
        
        # Snap Aquifer Productivity to Mesh
        
        merged_gdf_nodes_productivity = merged_gdf_nodes_soil.merge(
            productivity_gdf[['node_id', 'aquifer_productivity']],
            on='node_id',
            how='left'  # left join to keep all centroids, even if NaN
        )
        
        logger.info(f"Aquifer Productivity data snapped to mesh nodes (centroids).\n")
        
        # Snap Distance from River to Mesh
        
        static_features = merged_gdf_nodes_productivity.merge(
            dist_to_river_gdf[['node_id', 'distance_to_river']],
            on='node_id',
            how='left'  # left join to keep all centroids, even if NaN
        )
        
        logger.info(f"Distance from river data snapped to mesh nodes (centroids).\n")
        
        # [FUTURE] Snap Infiltration Rate to Mesh
        # [FUTURE] Snap Soil Type Maps to Mesh
        # [FUTURE] Snap Depth to Groundwater to Mesh
        
        # Finalise final_static_df for merge
        
        final_static_df = data_merging.reorder_static_columns(static_features)  # TODO: Update as more features added
        static_data_ingestion.save_final_static_data(
            static_features=final_static_df,
            dir_path=config[catchment]["paths"]["final_df_path"]
        )
        
        logger.info(f"Full static feature dataframe finalised and ready to merge into main model dataframe.\n")
        
        # --- 5c. Snap dynamic (timeseries) features to mesh nodes (equal across all for catchment) and timestep frequency ---
        
        pred_frequency = config["global"]["pipeline_settings"]["prediction_resolution"]
        
        # Snap Precipitation, Lags and Averages to timestep

        merged_ts_precipitation = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["rainfall_csv_path"],
            csv_name=f'rainfall_{pred_frequency}_catchment_sum_log_transform.csv',
            feature='all_precipitation',
            pred_frequency=pred_frequency
        )
        
        logger.info(f"Precipitation and derived data snapped to graph timesteps ({pred_frequency} aggregates).\n")
        
        # Snap 2m Temperature to timestep

        merged_ts_tsm = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["2t_csv_path"],
            csv_name=f'2m_temp_{pred_frequency}_catchment_mean.csv',
            feature='2m_temperature',
            pred_frequency=pred_frequency,
            timeseries_df=merged_ts_precipitation
        )
        
        logger.info(f"2m Temperature Data snapped to graph timesteps ({pred_frequency} aggregate).\n")
        
        # Snap AET to timestep

        merged_ts_aet = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["aet_csv_path"],
            csv_name=f'aet_{pred_frequency}_catchment_sum.csv',
            feature='aet',
            pred_frequency=pred_frequency,
            timeseries_df=merged_ts_tsm
        )
        
        merged_ts_aet = hydroclimatic_feature_engineering.transform_aet_data(merged_ts_aet, catchment)
        
        logger.info(f"Actual evapotranspiration data snapped to graph timesteps ({pred_frequency} aggregate).\n")
        
        # Snap Surface Pressure to timestep

        merged_ts_sp = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=config[catchment]["paths"]["sp_csv_path"],
            csv_name=f'surface_pressure_{pred_frequency}_catchment_mean.csv',
            feature='surface_pressure',
            pred_frequency=pred_frequency,
            timeseries_df=merged_ts_aet
        )
        
        logger.info(f"Surface pressure data snapped to graph timesteps ({pred_frequency} aggregate).\n")
        
        # Snap Streamflow to timestep
        
        final_merged_ts_df = data_merging.merge_timeseries_data_to_df(
            model_start_date=config["global"]["data_ingestion"]["model_start_date"],
            model_end_date=config["global"]["data_ingestion"]["model_end_date"],
            feature_csv=os.path.join(config[catchment]["paths"]["stream_flow_csv"], f'{pred_frequency}_streamflow.csv'),
            feature='streamflow',
            pred_frequency=pred_frequency,
            timeseries_df=merged_ts_sp
        )
        
        logger.info(f"Streamflow data snapped to graph timesteps ({pred_frequency} aggregate).\n")
        
        save_path = config[catchment]["paths"]["final_df_path"] + 'final_timeseries_df.csv'
        final_merged_ts_df.to_csv(save_path)
        
        logger.info(f"Final merged time series dataframe saved to {save_path}")
        logger.info(f"Full timeseries feature dataframe finalised and ready to merge into main model dataframe.\n")
        
        # --- 5d. Merge static, timeseries and gwl df's into main df ---
        
        main_df = graph_construction.build_main_df(
            start_date=config["global"]["data_ingestion"]["model_start_date"],
            end_date=config["global"]["data_ingestion"]["model_end_date"],
            mesh_nodes_gdf=mesh_nodes_gdf,
            catchment=catchment,
            pred_frequency=pred_frequency
        )

        # Merge static data into main_df
        final_dir = config[catchment]["paths"]["final_df_path"]
        static_df = pd.read_csv(os.path.join(final_dir, 'final_static_df.csv'))
        main_df_static = main_df.merge(
            static_df,
            left_on='node_id',
            right_on='node_id',
            how='left'
        )
        
        logger.info(f"Static data successfully merged into main_df for {catchment} catchment.\n")

        # Merge timeseries data into main_df
        
        timeseries_df = pd.read_csv(os.path.join(final_dir, 'final_timeseries_df.csv'))
        timeseries_df['time'] = pd.to_datetime(timeseries_df['time'])
        main_df_timeseries = main_df_static.merge(
            timeseries_df,
            left_on='timestep',
            right_on='time',
            how='left'
        ).drop(columns='time')

        logger.info(f"Timeseries data successfully merged into main_df for {catchment} catchment.\n")
            
        # Load GWL station data in
        
        seasonal_df = gwl_data[['timestep', 'season_sin', 'season_cos']].drop_duplicates('timestep')
        main_df_full = (
            main_df_timeseries
            .merge(seasonal_df, on='timestep', how='left')
            .merge(gwl_data.drop(columns=['season_sin', 'season_cos']), on=['node_id', 'timestep'], how='left')
        )

        logger.info(f"Groundwater Level data successfully merged into main_df for {catchment} catchment.\n")
        
        # Save final dataframe to file - NB: TIME TO SAVE APPROX. 2.5 MINS - [FUTURE] SET FLAG?
        
        final_save_path = os.path.join(final_dir, 'final_df.csv')
        main_df_full.to_csv(final_save_path)
        
        logger.info(f"Final merged dataframe saved to {final_save_path}")
        
        # --- 5e. Incorporate Edge Weighting (edge_weight if simple or only edge_att, using adjacency grid) ---
        
        edge_attr_tensor, edge_index_tensor = graph_construction.define_graph_adjacency(
            directional_edge_weights=directional_edge_weights,
            elevation_geojson_path=config[catchment]['paths']['elevation_geojson_path'],
            graph_output_dir=config[catchment]["paths"]["graph_data_output_dir"],
            mesh_cells_gdf_polygons=mesh_cells_gdf_polygons,
            epsilon_path=config["global"]["graph"]["epsilon"],
            catchment=catchment
        )

        logger.info(f"Pipeline step 'Define Graph Adjacency' complete for {catchment} catchment.\n")

        # --- 5f. Visualise complete mesh map with stations and other features ---
        # [FUTURE] Create an interactive map showing the mesh, GWL stations, and other snapped data points.
        
        # ==============================================================================
        # SECTION 6: TRAINING / VALIDATION / TESTING PYG OBJECT CREATION
        # ==============================================================================
        
        # --- 6a. Define Spatial Split for Observed Stations ---
                
        train_station_ids, val_station_ids, test_station_ids = data_partitioning.define_station_id_splits(
            main_df_full=main_df_full,
            catchment=catchment,
            test_station_shortlist=config[catchment]["model"]["data_partioning"]["test_station_shortlist"],
            val_station_shortlist=config[catchment]["model"]["data_partioning"]["val_station_shortlist"],
            random_seed=config["global"]["pipeline_settings"]["random_seed"],
            perc_train=config[catchment]["model"]["data_partioning"]["percentage_train"],
            perc_val=config[catchment]["model"]["data_partioning"]["percentage_val"],
            perc_test=config[catchment]["model"]["data_partioning"]["percentage_test"]
        )

        logger.info(f"Pipeline Step 'define station splits' complete for {catchment} catchment.\n")
        
        # --- 6b. Preprocess (Standardise, one hot encode, round to 4dp) all shared features (not GWL) ---
        
        processed_df, shared_scaler, encoder, gwl_feats = model_feature_engineering.preprocess_shared_features(
            main_df_full=main_df_full,
            catchment=catchment,
            random_seed=config["global"]["pipeline_settings"]["random_seed"],
            violin_plt_path=config[catchment]["visualisations"]["violin_plt_path"],
            scaler_dir = config[catchment]["paths"]["scalers_dir"]
        )

        logger.info(f"Pipeline Step 'Preprocess Final Shared Features' complete for {catchment} catchment.\n")
        
        # --- 6c. Preprocess all GWL features using training data only ---
        
        processed_df, gwl_scaler, gwl_encoder = model_feature_engineering.preprocess_gwl_features(
            processed_df=processed_df,
            catchment=catchment,
            train_station_ids=train_station_ids,
            val_station_ids=val_station_ids,
            test_station_ids=test_station_ids,
            sentinel_value = config["global"]["graph"]["sentinel_value"],
            scaler_dir = config[catchment]["paths"]["scalers_dir"]
        )

        logger.info(f"Pipeline Step 'Preprocess Final GWL Features' complete for {catchment} catchment.\n")
        
        # --- 6d. Creat PyG data object using partioned station IDs (from 6a) ---
        
        # Run time approx. 12.5 mins to build 4018 timesteps of objects (0.19s per Object)
        all_timesteps_list = data_partitioning.build_pyg_object(
            processed_df=processed_df,
            sentinel_value=config["global"]["graph"]["sentinel_value"],
            train_station_ids=train_station_ids,
            val_station_ids=val_station_ids,
            test_station_ids=test_station_ids,
            gwl_feats=gwl_feats,
            edge_index_tensor=edge_index_tensor,
            edge_attr_tensor=edge_attr_tensor,
            catchment=catchment
        )

        torch.save(all_timesteps_list, config[catchment]["paths"]["pyg_object_path"])
        logger.info(f"Pipeline Step 'Build PyG Data Objects' complete for {catchment} catchment.\n")

        # --- 6e. Define Graph Adjacency Matrix (edge_index -> 8 nearest neighbours) ---
        # Already generated in Step 5e and incorporated into PyG objects in step 6d.

        # ====================================================================================================
        # SECTION 7: MODEL
        # ----------------------------------------------------------------------------------------------------
        # Instantiate GAT-LSTM Model using PyTorch Geometric:
        #   - Construct PyTorch Geometric Data objects (one per timestep), passing edge_index and edge_attr as
        #     separate arguments to the Data constructor, alongside x (node features) and y (targets).
        # ====================================================================================================

        # --- 7a. Build Data Loaders by Timestep ---

        full_dataset_loader = model_building.build_data_loader(
            all_timesteps_list=all_timesteps_list,
            batch_size = config["global"]["model"]["data_loader_batch_size"],
            shuffle = config["global"]["model"]["data_loader_shuffle"],
            catchment=catchment
        )

        logger.info(f"Pipeline Step 'Create PyG DataLoaders' complete for {catchment} catchment.\n")
        
        # --- 7b. Define Graph Neural Network Architecture including loss and optimiser definition ---

        # Adjust model architecture and params in catchment-specific config. TODO: Further optimise hyperparams.
        model, device, optimizer, criterion = model_building.instantiate_model_and_associated(
            all_timesteps_list=all_timesteps_list,
            config=config,
            catchment=catchment
        )

        logger.info(f"Pipeline Step 'Instantiate GAT-LSTM Model' complete for {catchment} catchment.\n")

        # ==============================================================================
        # SECTION 8: TRAINING
        # ==============================================================================
        
        # --- 8a. Implement Training Loop ---
        
        # train_losses, val_losses = model_training.run_training_and_validation(
        #     num_epochs=config[catchment]["training"]["num_epochs"],
        #     early_stopping_patience=config[catchment]["training"]["early_stopping_patience"],
        #     lr_scheduler_factor=config[catchment]["training"]["lr_scheduler_factor"],
        #     lr_scheduler_patience=config[catchment]["training"]["lr_scheduler_patience"],
        #     min_lr=config[catchment]["training"]["min_lr"],
        #     gradient_clip_max_norm=config[catchment]["training"]["gradient_clip_max_norm"],
        #     model_save_path=config[catchment]["paths"]["best_model_path"],
        #     loss_delta=config[catchment]["training"]["loss_delta"],
        #     verbose=config[catchment]["training"]["verbose"],
        #     catchment=catchment,
        #     model=model,
        #     device=device,
        #     optimizer=optimizer,
        #     criterion=criterion,
        #     all_timesteps_list=all_timesteps_list,
        #     scalers_dir=config[catchment]["paths"]["scalers_dir"]
        # )

        # logger.info(f"Pipeline Step 'Train and Validate Model' complete for {catchment} catchment.")

        # model_training.save_train_val_losses(
        #     output_analysis_dir=config[catchment]["paths"]["model_dir"],
        #     train_losses=train_losses,
        #     val_losses=val_losses
        # )

        # logger.info(f"Pipeline Step 'Save Training and Validation Losses' complete for {catchment} catchment.")

        # --- 8b. Model Checkpointing and Logging ---
        # Action: Save best performing model weights based on validation metrics.
        # Action: Log training and validation metrics (e.g., using TensorBoard, MLflow, or custom logging).
        
        # --- 8c. Hyperparameter Tuning Here? ---

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