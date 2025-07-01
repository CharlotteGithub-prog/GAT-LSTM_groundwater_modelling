# Import Libraries
import os
import sys
import logging
import pandas as pd
    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def reorder_static_columns(df: pd.DataFrame):
    """ 
    Reorder columns in df to put geometry before features
    """
    desired_order = ['node_id', 'geometry', 'polygon_geometry', 'easting', 'northing',
                    'lon', 'lat', 'land_cover_code', 'mean_elevation',
                    'mean_slope_degrees', 'mean_aspect_sin', 'mean_aspect_cos']

    return df[desired_order]

def add_polygon_to_centroids(df: pd.DataFrame):
    pass