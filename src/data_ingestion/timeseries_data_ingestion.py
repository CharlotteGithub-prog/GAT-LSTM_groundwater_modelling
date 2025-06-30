# Import Libraries
import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

# from src.data_ingestion.spatial_transformations import easting_northing_to_lat_long, \
#     find_catchment_boundary
    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

