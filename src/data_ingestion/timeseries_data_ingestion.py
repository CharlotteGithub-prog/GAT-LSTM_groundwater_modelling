# Import Libraries
import os
import sys
import time
import cdsapi
import logging
import calendar
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from src.data_ingestion.spatial_transformations import easting_northing_to_lat_long, \
    find_catchment_boundary
    
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

def load_aet_data(catchment: float, shape_filepath: float, required_crs: int, cdsapi_path: str,
                  start_date: str, end_date: str, run_era5_land_api: bool, raw_output_dir: str,
                  processed_output_dir: str):    
    """
    Downloads, preprocesses, and saves Actual Evapotranspiration (AET) data
    from the ERA5-Land reanalysis dataset.
    """
    # Get Total Evapotranspiration from ERA5-Land dataset
    os.environ["CDSAPI_RC"] = os.path.expanduser(cdsapi_path)
    c = cdsapi.Client()

    # Suppress cdsapi INFO and WARNING messages
    # logging.getLogger("cdsapi").setLevel(logging.ERROR)

    # Get date range
    start_year = int(start_date[0:4])
    end_year = int(end_date[0:4])
    
    ## --- Find catchment bounding box --- 
    
    _, _, minx, miny, maxx, maxy = find_catchment_boundary(
        catchment=catchment,
        shape_filepath=shape_filepath,
        required_crs=required_crs
    )

    # Get catchment bounding box
    bbox_df = pd.DataFrame({
        'easting': [minx, maxx],
        'northing': [miny, maxy]
    })

    latlon_bbox = easting_northing_to_lat_long(bbox_df)

    south = latlon_bbox['lat'].min()
    north = latlon_bbox['lat'].max()
    west = latlon_bbox['lon'].min()
    east = latlon_bbox['lon'].max()

    ## --- Call CDS API monthyl to retrieve AET data --- 

    # List to hold processed daily xarray DataArrays from each year
    all_daily_aet_dataarrays = []

    # Ensure the output directories exist
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)

    if run_era5_land_api:
        
        # Get daily AET data (mm/day)
        for year in range(start_year, end_year + 1):
            logging.info(f"Retrieving AET data from ERA5-Land for year {year}...")
            
            # Get month range:
            month_start = int(start_date[5:7]) if year == start_year else 1
            month_end = int(end_date[5:7]) if year == start_year else 12
            
            for month in range(month_start, month_end + 1):
                logging.info(f"Retrieving AET data from ERA5-Land data for month {month}...")
                
                # Define number of days in month
                num_days = calendar.monthrange(year, month)[1]
                
                # Define filename to save
                grib_filename = f"{raw_output_dir}aet_{year}_{month:02d}_era5land.grib"
                
                # TODO: Update this to handle a call failing after c.retrieve before end
                # Skip download if already there (allows retrieval to be split up when long)
                # if os.path.exists(grib_filename):
                #     logging.info(f"File already exists: {grib_filename}. Skipping download.")
                #     continue
                
                # Try retrieving data, return error if unable to for month
                try:
                    # Initialise timing
                    start_time = time.time()
                    
                    c.retrieve(
                        'reanalysis-era5-land',
                        {
                            'variable': 'total_evaporation',
                            'year': str(year),
                            'month': month,
                            'day': [f"{d:02d}" for d in range(1, num_days + 1)],
                            'time': [f'{h:02d}:00' for h in range(24)],
                            'format': 'grib',
                            'area': [north, west, south, east]
                        },
                        grib_filename
                    )

                    # Complete timing and log
                    elapsed = time.time() - start_time
                    logging.info(f"Successfully downloaded GRIB for {month:02d}/{year} in {elapsed:.2f} seconds.")
                
                    # Open .nc file and convert to daily timestep (from hourly)
                    if os.path.exists(grib_filename) and os.path.getsize(grib_filename) > 0:
                        ds = xr.open_dataset(grib_filename, engine='cfgrib')
                        logging.info(f"Converting AET for {month:02d}/{year} to daily timestep...")
                        
                        # Sum 24 hourly values to get daily total, then convert to millimeters
                        daily_mm = ds['e'].resample(time='1D').sum() * 1000
                        all_daily_aet_dataarrays.append(daily_mm)
                        
                    else:
                        logging.warning(f"File {grib_filename} appears empty or corrupted. Skipping.")
                        continue

                except Exception as e:
                    logging.error(f"Error retrieving or processing AET for {month:02d}/{year}: {e}")

    # --- Aggregate to daily values for catchment ---

    if all_daily_aet_dataarrays:
        # Concatenate all monthly DataArrays into a single xarray.DataArray along the time dimension
        full_aet_da = xr.concat(all_daily_aet_dataarrays, dim='time')

        # Save the combined data to a single NetCDF file for the entire period
        final_nc_path = f"{processed_output_dir}/aet_daily_{start_year}-{end_year}_era5land.nc"
        logging.info(f"Saving combined daily AET from {start_year}-{end_year} to: {final_nc_path}")
        full_aet_da.to_netcdf(final_nc_path)
        
        logging.info("ERA5-Land AET data retrieval and processing complete.")

        # --- Save processed data as csv ---

        # Save as csv
        final_csv_path = f"{processed_output_dir}aet_daily_catchment_mean.csv"
        logging.info(f"Saving catchment-averaged daily AET to CSV: {final_csv_path}")
        
        catchment_mean_aet = full_aet_da.mean(dim=["longitude", "latitude"])
        catchment_mean_aet.to_dataframe().to_csv(final_csv_path)
        
        return catchment_mean_aet

    else:
        
        logging.info(f"No AET data was retrieved or processed.")
        return None