# Import Libraries
import os
import sys
import logging
import numpy as np
import xarray as xr
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

HAD_UK_rainfall_url = "https://dap.ceda.ac.uk/thredds/dodsC/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.3.0.ceda/1km/rainfall/day/v20240514/rainfall_hadukgrid_uk_1km_day_20230101-20230131.nc"

# Find HADUK file names
def find_haduk_file_names(start_date: str, end_date: str, base_url: str):
    
    start_year = int(start_date[0:4])
    end_year = int(end_date[0:4])

    # Initialise list to hold urls
    urls = []

    for y in range(start_year, end_year + 1):
        year, nMonths = y, 12
        year_start_str = f"{year}-01-01"
        
        # CEDA files use %Y%m%d format for file name identification
        month_start = pd.date_range(year_start_str, periods=nMonths, freq='MS').strftime("%Y%m%d")
        month_end = pd.date_range(year_start_str, periods=nMonths, freq='ME').strftime("%Y%m%d")

        months = [(base_url + start + '-' + end + '.nc') for start, end in zip(month_start, month_end)]
        
        # Add to main list, doesn't need to be divided by year
        urls.extend(months)
    
    return urls

# HADUK Gridded Rainfall
# def load_main_rainfall_data(HAD_UK_rainfall_url):
#     """
#     Loads rainfall data from...
#     """
#     logger.info(f"Loading HADUK rainfall data via CEDA OPeNDAP service...")

#     ds = xr.open_dataset(url)
#     print(ds)
    

# def load_prelim_rainfall_data():