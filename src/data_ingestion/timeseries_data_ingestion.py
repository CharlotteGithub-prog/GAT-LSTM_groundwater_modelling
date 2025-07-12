# Import Libraries
import os
import sys
import time
import cdsapi  # ERA5-Land API
import logging
import zipfile
import datetime
import calendar
import regionmask
import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Geod
import geopandas as gpd
import concurrent.futures
import matplotlib.pyplot as plt

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

def _get_bbox_and_polygon(catchment, shape_filepath, required_crs):
    # Derive full path
    temp_geojson_path = f"{catchment}_combined_boundary.geojson"
    path = shape_filepath + temp_geojson_path
    
    catchment_polygon, _, minx, miny, maxx, maxy = find_catchment_boundary(
        catchment=catchment,
        shape_filepath=path,
        required_crs=required_crs
    )

    # Get catchment bounding box
    bbox_df = pd.DataFrame({
        'easting': [minx, maxx],
        'northing': [miny, maxy]
    })

    # Convert to lat, lon
    latlon_bbox = easting_northing_to_lat_long(bbox_df)

    south = latlon_bbox['lat'].min()
    north = latlon_bbox['lat'].max()
    west = latlon_bbox['lon'].min()
    east = latlon_bbox['lon'].max()
    
    return north, south, east, west, catchment_polygon

def _estimate_cell_area(lat, lon, dlat, dlon, g):
    """
    Estimate area in m² of a lat/lon cell using pyproj.
    
    Args:
        lat, lon: the center point of a grid cell.
        dlat, dlon: the height and width of the grid cell in degrees.
    """
    # Construct vertices of closed rectangular polygon representing each grid cell
    lons = [lon - dlon/2, lon + dlon/2, lon + dlon/2, lon - dlon/2, lon - dlon/2]
    lats = [lat - dlat/2, lat - dlat/2, lat + dlat/2, lat + dlat/2, lat - dlat/2]
    
    # Use Geod.polygon_area_perimeter to compute the signed area (in m²) of the polygon
    area, _ = g.polygon_area_perimeter(lons, lats)
    return abs(area)

def _compute_weighted_aggregation(full_da, feat_name, aggregation_type):
    # Debug prints:
    logging.info(f"DEBUG: Entering _compute_weighted_aggregation for {feat_name}")
    logging.info(f"DEBUG: full_da dims: {full_da.dims}, coords: {list(full_da.coords.keys())}")
    logging.info(f"DEBUG: full_da latitude: {full_da.latitude.values}")
    logging.info(f"DEBUG: full_da longitude: {full_da.longitude.values}")

    # Extract coordinate arrays (1D) from the DataArray
    lats = full_da.latitude.values
    lons = full_da.longitude.values
    
    # Calcualating resolution by feature (0.1 or 0.05 by type in ERA5-Land)
    lat_resolution = abs(lats[1] - lats[0])
    lon_resolution = abs(lons[1] - lons[0])

    logging.info(f"Grid cell resolution (deg): {lat_resolution:.2f}, {lon_resolution:.2f}")
    logging.info(f"Total grid cells: {len(lats)*len(lons)}")
    
    # Define g to allow for geodetic (curved-earth) calcs using the WGS84 ellipsoid (next section)
    g = Geod(ellps="WGS84")

    # Loop over each grid cell's center point and calculate the area in m^2
    area_array = np.zeros((len(lats), len(lons)))  # stored in 2D array matching spatial grid
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            area_array[i, j] = _estimate_cell_area(lat, lon, lat_resolution, lon_resolution, g)

    # converts NumPy array of cell areas to xarray.DataArray for alignment with full_aet_da
    area_da = xr.DataArray(
        area_array,
        coords={"latitude": lats, "longitude": lons},
        dims=["latitude", "longitude"]
    )

    # Convert mm to m^3 (as standard for sum total for area) for pre-masked (polygon) area
    if aggregation_type == 'sum':
        weighted_data = full_da * area_da
        catchment_aggregated_data = weighted_data.sum(dim=["latitude", "longitude"])
        catchment_aggregated_data.name = f"{feat_name}_total_volume_m3"
        
    # For 'mean' need area-weighted average (sum of value * area) / (sum of area)
    elif aggregation_type == 'mean':
        
        # Build mask to only the catchment cells, not whole bbox (as causes scaling issue)
        mask = ~np.isnan(full_da.isel(time=0))
        area_catch = area_da.where(mask)
        
        # Calculate weighted sum 
        weighted_sum = (full_da * area_da).sum(dim=["latitude","longitude"])
        total_catchment_area = area_catch.sum(dim=["latitude","longitude"])
        catchment_aggregated_data = weighted_sum / total_catchment_area
        catchment_aggregated_data.name = f"{feat_name}_area_weighted_mean"
    
    # If incorrect aggregation type raise value error
    else:
        raise ValueError(f"Unsupported aggregation_type: {aggregation_type}.")
    
    return catchment_aggregated_data

def _combine_and_aggregate_daily_data(all_daily_dataarrays, processed_output_dir,start_year, end_year,
                                      feat_name, aggregation_type, catchment):
    # Concatenate all monthly DataArrays into a single xarray.DataArray along the time dimension
    full_da = xr.concat(all_daily_dataarrays, dim='time')

    # Save the combined data to a single NetCDF file for the entire period
    final_nc_path = f"{processed_output_dir}{feat_name}_daily_{start_year}-{end_year}_era5land.nc"
    logging.info(f"Saving combined daily {feat_name} from {start_year}-{end_year} to: {final_nc_path}")
    
    full_da.to_netcdf(final_nc_path)
    
    logging.info(f"ERA5-Land {feat_name} data retrieval and processing complete.")
    

    # --- Save processed data as csv ---

    # Save as csv
    final_csv_path = f"{processed_output_dir}{feat_name}_daily_catchment_{aggregation_type}.csv"
    logging.info(f"Saving catchment-summed daily {feat_name} to CSV: {final_csv_path}")
    
    # Compute total volume loss for area using weighted area per grid cell (not uniform)
    catchment_sum_data = _compute_weighted_aggregation(full_da, feat_name, aggregation_type)
    
    # Handle outlying values if needed (manual confirmation)
    if feat_name == 'surface_pressure' and catchment == 'eden':
        catchment_sum_data = _handle_outlying_values(catchment_sum_data, catchment)
    
    # Drop duplicate (overlapping) timestamps before saving
    time_index = catchment_sum_data.indexes["time"]
    if time_index.has_duplicates:
        logging.warning("Duplicate dates found in catchment_sum_data -> will be dropped.")
        catchment_sum_data = catchment_sum_data.sel(
            time=~time_index.duplicated()
        )

    # Save as csv to merge into main model df    
    catchment_sum_data.to_dataframe().to_csv(final_csv_path)
    
    return catchment_sum_data

def _extract_zip_file(zip_filename, raw_output_dir, year, month, feat_name):
    if zipfile.is_zipfile(zip_filename):
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            zip_ref.extractall(raw_output_dir)

            extracted_grib_name = next((f for f in extracted_files if f.endswith('.grib')), None)
            if not extracted_grib_name:
                logging.error(f"No GRIB file found in ZIP {zip_filename}")
                return None

            extracted_grib_path = os.path.join(raw_output_dir, extracted_grib_name)
            renamed_grib_path = f"{raw_output_dir}{feat_name}_{year}_{month:02d}_era5land.grib"

            os.rename(extracted_grib_path, renamed_grib_path)
            grib_filename = renamed_grib_path  # Update reference for loading

            logging.info(f"Renamed extracted file to: {renamed_grib_path}")

        # Delete the actual ZIP
        os.remove(zip_filename)
        logging.info(f"Deleted ZIP archive: {zip_filename}")
        return renamed_grib_path

    return None

def _apply_catchment_mask(catchment_polygon, ds, daily_data_sliced):
    """
    Mask from whole bounding box to catchment polygon before aggregating to catchment
    totals to avoid external bounding box data leaking into polygon.
    """
    # Reset catchment mask to avoid leakage between features causing issues
    # regionmask.defined_regions.clear()
    
    # Ensure the polygon is in the correct CRS for masking against lon/lat data
    polygon_4326 = catchment_polygon.to_crs("EPSG:4326").copy()
    
    mask_da = regionmask.mask_geopandas(
        polygon_4326,
        daily_data_sliced.longitude.copy(), # Pass xarray.DataArray for coordinates, not just values
        daily_data_sliced.latitude.copy()   # Pass xarray.DataArray for coordinates, not just values
    )
    
    # Ensure consistent naming ('latitude', 'longitude') before applying the mask
    if 'lat' in mask_da.dims and 'lon' in mask_da.dims:
        mask_da = mask_da.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif 'latitude' not in mask_da.dims and 'longitude' not in mask_da.dims:
        pass # Or raise an error if expected dims are missing
    
    # Apply the 2D mask to the daily_mm df (Xarray handles broadcasting)
    masked_data = daily_data_sliced.where(~np.isnan(mask_da))
    
    return masked_data

def _call_era5_api(start_year, end_year, start_date, end_date, total_months, raw_output_dir, north,
                  west, south, east, c, era5_feat, feat_name, era5_long, aggregation_type):
    # Initialise API call counter
    call_count = 0
    
    # Get daily feature data (aet: mm/day, temp: av degrees, snow: ?)
    for year in range(start_year, end_year + 1):
        
        # Get month range:
        month_start = int(start_date[5:7]) if year == start_year else 1
        month_end = int(end_date[5:7]) if year == end_year else 12
        
        for month in range(month_start, month_end + 1):
            
            # Track overall API call progress
            call_count += 1
            logging.info(f"API PROGRESS: Starting call {call_count} of {len(total_months)}")
            logging.info(f"Retrieving {feat_name} data from ERA5-Land data for {year} -> month {month}...\n")
            
            # Define number of days in month
            num_days = calendar.monthrange(year, month)[1]
            
            # Define filename to save
            zip_filename = f"{raw_output_dir}{feat_name}_{year}_{month:02d}_era5land.zip"
            grib_filename = f"{raw_output_dir}{feat_name}_{year}_{month:02d}_era5land.grib"
            
            # TODO: Update this to handle a call failing after c.retrieve before end
            # Skip if GRIB already downloaded (to allow retrieval to be split up when long calls)
            logging.info(f"Checking if GRIB file exists: {grib_filename}")
            if os.path.exists(grib_filename):
                logging.info(f"GRIB file already exists: {grib_filename} — skipping download.")
                continue
            logging.info(f"{grib_filename} does not exist in {raw_output_dir}")  
            
            # Try retrieving data, return error if unable to for month
            try:
                # Initialise timing
                start_time = time.time()
                
                area = [round(float(north), 2), round(float(west), 2), round(float(south), 2), round(float(east), 2)]
                
                c.retrieve(
                    'reanalysis-era5-land',
                    {
                        'variable': era5_long,
                        'year': str(year),
                        'month': month,
                        'day': [f"{d:02d}" for d in range(1, num_days + 1)],
                        'time': [f'{h:02d}:00' for h in range(24)],
                        'format': 'grib',
                        'area': area  # Initially load data for bounding box not polygon
                    },
                    zip_filename
                )

                # Complete timing and log
                elapsed = time.time() - start_time
                logging.info(f"Successfully downloaded GRIB for {month:02d}/{year} in {elapsed:.2f} seconds.")
                
                # If file is a ZIP, extract it
                grib_filename = _extract_zip_file(zip_filename, raw_output_dir, year, month, feat_name)

                if grib_filename is None:
                    continue
                
                file_size = os.path.getsize(grib_filename)
                if file_size == 0:
                    logging.error(f"Downloaded file {grib_filename} is empty. Retrying or skipping.\n")
                    os.remove(grib_filename)
                    continue # Skip processing this month and try next

            except Exception as e:
                logging.error(f"Error retrieving or processing {feat_name} data for {month:02d}/{year}: {e}\n")

def _resample_by_time_and_step(ds, era5_feat, aggregation_type):
    # Mapping from  internal era5_feat to actual variable name in the xarray dataset loaded by cfgrib
    grib_variable_map = {
        '2t': 't2m',
        'e': 'e',
        'sp': 'sp'
    }
    
    # Get the actual variable name from the dataset based on era5_feat (fallback to era5_feat if not mapped)
    actual_grib_variable_name = grib_variable_map.get(era5_feat, era5_feat)
    
    # GRIB formatting means resample must be done over both time and step, regular 1D fails
    feat = ds[actual_grib_variable_name]
    feat_flat = feat.stack(time_step=("time", "step"))

    valid_times = ds['valid_time'].stack(time_step=("time", "step"))
    feat_flat = feat_flat.assign_coords(valid_time=("time_step", valid_times.data))

    valid_mask = ~np.isnan(feat_flat['valid_time'].values.astype('datetime64[ns]'))
    feat_flat =feat_flat.isel(time_step=valid_mask)
    feat_flat = feat_flat.swap_dims({"time_step": "valid_time"}).sortby("valid_time")
    
    # Apply transformation for data to standard units and range for AET data
    if era5_feat == 'e':  # AET
        transformed_data = -1 * feat_flat  # Invert only
        transformed_data = transformed_data.where(transformed_data >= 0, 0)
    elif era5_feat == '2t':  # 2m temperature
        transformed_data = feat_flat - 273.15  # Convert from ºK to ºC
    elif era5_feat == 'sp':  # surface pressure
        transformed_data = feat_flat / 100  # Convert Pa to hPa
    else:
        raise ValueError(f"No specific transformation defined for feature: {era5_feat}")
        
    # Return daily data aggregated by type
    if aggregation_type == 'sum':
        return transformed_data.resample(valid_time="1D").sum()
    elif aggregation_type == 'mean':
        return transformed_data.resample(valid_time="1D").mean()
    else:
        raise ValueError(f"No specific aggregation defined for type: {aggregation_type}")
        
def _save_era5_graph(csv_path, fig_path, feat_name, catchment, aggregation_type):
    # Load the processed CSV
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    print(f"Max value in dataset: {df.max().values}")
    print(f"Min value in dataset: {df.min().values}")
    
    plot_label = "Catchment Data"
    y_label = "Value"
    title_suffix = "Data"
    
    # Specific handling for known features and aggregation types
    if feat_name == 'aet' and aggregation_type == 'sum':
        plot_label = f"Catchment {feat_name.upper()} Volume"
        y_label = "Volume (m³)"
        title_suffix = "AET Total Volume Loss"
        
    elif feat_name == '2m_temp' and aggregation_type == 'mean':
        plot_label = "Catchment 2m Temperature"
        y_label = "Temperature (°C)"
        title_suffix = "Average Temperature"
        
    elif feat_name == 'surface_pressure' and aggregation_type == 'mean':
        plot_label = "Catchment Surface Pressure"
        y_label = "Pressure (hPa)"
        title_suffix = "Average Pressure"

    # Plot time series
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df.values, label=plot_label, color='tab:blue')
    plt.title(f"Daily {catchment} Catchment - {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    # Save figure to results
    plt.savefig(fig_path)

def _process_local_grib_files(raw_output_dir, catchment_polygon, all_daily_dataarrays,
                              era5_feat, aggregation_type, feat_name):
    
    for fname in sorted(os.listdir(raw_output_dir)):
        if fname.endswith(".grib"):
            grib_path = os.path.join(raw_output_dir, fname)
            logging.info(f"Processing GRIB: {fname}")
            
            # Aggregate to daily totals and mask to polygon boundings not bbox
            try:
                ds = xr.open_dataset(grib_path, engine='cfgrib')
                daily_data = _resample_by_time_and_step(ds, era5_feat, aggregation_type)
                
                # Extract year and month from the filename (e.g., aet_2024_04_era5land.grib)
                parts = fname.split('_')
                current_year = int(parts[-3])
                current_month = int(parts[-2])
                
                # # Define the start and end of the actual data for the current month (to avoid introducing 0's)
                last_day_of_month = calendar.monthrange(current_year, current_month)[1]
                month_start_date = datetime.datetime(current_year, current_month, 1)
                month_end_date = datetime.datetime(current_year, current_month, last_day_of_month)
                
                # Slice daily_data using these dates
                daily_data_sliced = daily_data.sel(valid_time=slice(month_start_date, month_end_date))
                masked = _apply_catchment_mask(catchment_polygon, ds, daily_data_sliced)
                
                # Clean and ensure consistent structure before appending to all_daily_aet_dataarrays
                masked = masked.drop_vars(['number', 'surface'], errors='ignore')
                masked = masked.rename({'valid_time': 'time'})
                masked = masked.astype('float32')
                
                all_daily_dataarrays.append(masked)  # Append masked daily AET DataArray to monthly list
                
            except Exception as e:
                logging.error(f"Error processing {fname}: {e}")

def _handle_outlying_values(catchment_agg, catchment):
    """
    Adjust misaligned surface pressure values (manual not generalised outlier detection).
    """
    logging.info(f"Adjusting outlying values in {catchment} catchment.")
    catchment_clean = catchment_agg.copy()
    
    n_adjusted = int((catchment_clean > 1100).sum())
    logging.info(f"Adjusted {n_adjusted} values over 1100 hPa in surface pressure.")

    # Adjust value (hard coded to correct surface_pressure)
    catchment_clean = xr.where(catchment_clean > 1100, catchment_clean - 175, catchment_clean)
    return catchment_clean

def load_era5_land_data(catchment: str, shape_filepath: float, required_crs: int, cdsapi_path: str,
                  start_date: str, end_date: str, run_era5_land_api: bool, raw_output_dir: str,
                  processed_output_dir: str, csv_path: str, fig_path: str, era5_feat: str = 'e',
                  era5_long: str = 'total_evaporation', feat_name: str = 'aet', aggregation_type: str = 'sum'):    
    """
    Downloads, preprocesses, aggregates and saves specified data
    from the ERA5-Land reanalysis dataset.
    
    Retrieve feature 'era5_feat' str using shortName param from documentation > parameter listings:
        https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-parameterlistingParameterlistings
    """
    # Get specified features from ERA5-Land dataset
    os.environ["CDSAPI_RC"] = os.path.expanduser(cdsapi_path)
    c = cdsapi.Client()

    # Get date range
    start_year = int(start_date[0:4])
    end_year = int(end_date[0:4])
    total_months = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    ## --- Find catchment bounding box --- 
    
    north, south, east, west, catchment_polygon = _get_bbox_and_polygon(
        catchment, shape_filepath, required_crs)
    
    logging.info(f"Catchment BBox (lat/lon): {north, south, east, west}")
    logging.info(f"Catchment polygon CRS: {catchment_polygon.crs}")
    logging.info(f'Catchment polygon area (approx.): {catchment_polygon.to_crs("EPSG:3857").area.sum()/1e6} km^2')
    
    ## --- Call CDS API monthyl to retrieve AET data --- 

    # List to hold processed daily xarray DataArrays from each year
    all_daily_dataarrays = []

    # Ensure the output directories exist
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)

    # Only run API call if specified (Warning: Multi-hour call)
    if run_era5_land_api:
        
        # NOTE: Calling API in parallel does not speed up performance due to ERA5 internal queuing.
        _call_era5_api(start_year, end_year, start_date, end_date, total_months, raw_output_dir,
                       north, west, south, east, c, era5_feat, feat_name, era5_long, aggregation_type)
        
    
    
    # Always process data
    _process_local_grib_files(raw_output_dir, catchment_polygon, all_daily_dataarrays,
                              era5_feat, aggregation_type, feat_name)
    
    if all_daily_dataarrays:
        
        # --- Aggregate to daily values for catchment ---
        catchment_agg = _combine_and_aggregate_daily_data(
            all_daily_dataarrays, processed_output_dir,start_year, end_year,
            feat_name, aggregation_type, catchment)
        
        # --- Save time series data to results ---
        _save_era5_graph(csv_path, fig_path, feat_name, catchment, aggregation_type)
        
        return catchment_agg

    else:
        
        logging.info(f"No {feat_name} data was retrieved or processed.\n")
        return None
