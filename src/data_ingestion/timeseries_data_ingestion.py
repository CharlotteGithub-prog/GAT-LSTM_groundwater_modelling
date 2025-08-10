# Import Libraries
import os
import sys
import time
import cdsapi  # ERA5-Land API
import logging
import zipfile
import datetime
import requests
import calendar
import regionmask
import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Geod
import concurrent.futures
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime as dt

from src.data_ingestion.static_data_ingestion import _transform_skewed_data
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

# Find HADUK file names

def find_haduk_file_names(start_date: str, end_date: str, base_url: str):
    """
    NOTE: HADUK Data should be ingested through API, but ongoing credential issues
    necessitated manual download for current pipeline. This should be adjusted back
    to use the API when these CEDA issues are resoled.
    """
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

# Load in HAD-UK 1km gridded rainfall Data

def _save_haduk_graph(csv_path, fig_path, pred_frequency, catchment):
    # Load the processed CSV
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    print(f"Max value in dataset: {df.max().values}")
    print(f"Min value in dataset: {df.min().values}")

    # Plot time series
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df.values, label="Catchment Rainfall Volume", color='tab:blue')
    plt.title(f"{pred_frequency} {catchment} Catchment - Total Rainfall Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume (m³)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    # Save figure to results
    plt.savefig(fig_path)

def _slice_and_mask_data(ds, north, south, east, west, catchment, mask_da):
    # --- slice data to bounding box ---
            
    # Generate boolean mask for lat/lon bounds
    lat2d = ds.latitude
    lon2d = ds.longitude

    within_bbox = (lat2d >= south) & (lat2d <= north) & (lon2d >= west) & (lon2d <= east)

    # Get bounding indices of valid region
    valid_y, valid_x = np.where(within_bbox)

    if valid_y.size == 0 or valid_x.size == 0:
        raise ValueError("No valid points found within bounding box — check bounding box or data grid.")

    # Get index bounds to slice projection coords
    min_y, max_y = valid_y.min(), valid_y.max()
    min_x, max_x = valid_x.min(), valid_x.max()

    # Slice using projection coordinate dimensions
    ds = ds.isel(projection_y_coordinate=slice(min_y, max_y + 1), 
                projection_x_coordinate=slice(min_x, max_x + 1))
    
    # --- mask to polygon bounds ---
            
    # Apply the 2D mask to ds (xarray handles broadcasting)
    logging.info(f'    Masking rainfall data to {catchment} catchment.')
    masked_data = ds.where(~np.isnan(mask_da))
    
    return masked_data

def _process_rainfall_files(rainfall_dir, catchment, shape_filepath, required_crs):
    """
    Provess raw rainfall data files in directory. Slice data to bounding box and mask to
    catchment polygon bounds. Append masked data to list of all files and return.
    """
    # --- Get bounding box for catchment

    north, south, east, west, catchment_polygon =  _get_bbox_and_polygon(
        catchment, shape_filepath, required_crs)
        
    logging.info(f"Catchment BBox (lat/lon): {north, south, east, west}")
    logging.info(f"Catchment polygon CRS: {catchment_polygon.crs}")
    logging.info(f'Catchment polygon area (approx.): {catchment_polygon.to_crs("EPSG:3857").area.sum()/1e6} km^2')
    
    # -- Loop through raw data files ---
    
    # Initialise data list and catchment polygon
    rainfall_data_all = []
    polygon_4326 = catchment_polygon.to_crs("EPSG:4326").copy()

    # Loop through file in directory and append file to full data list
    for filename in sorted(os.listdir(rainfall_dir)):
        # if filename.startswith("rainfall_hadukgrid_uk_1km_day_"):
        if filename.startswith("rainfall_hadukgrid_uk_1km_day_"):
            ds = xr.open_dataset(os.path.join(rainfall_dir, filename), backend_kwargs={"decode_timedelta": True})
            
            year = filename[-11:-7]
            month = filename[-7:-5]
            logging.info(f'Processing rainfall data for month {month} year {year}...')
            
            # --- mask to polygon bounds ---

            # Only generate mask first time to reduce computation
            if not rainfall_data_all:
                logging.info("Generating regionmask for first file...")
                
                mask_da = regionmask.mask_geopandas(
                    polygon_4326,
                    ds.longitude.copy(),
                    ds.latitude.copy()
                )
                
                # Ensure consistent naming ('latitude', 'longitude') before applying the mask
                if 'lat' in mask_da.dims and 'lon' in mask_da.dims:
                    mask_da = mask_da.rename({'lat': 'latitude', 'lon': 'longitude'})
            
            masked_data = _slice_and_mask_data(ds, north, south, east, west, catchment, mask_da)
            
            # Drop uneeded vars to reduce computation
            masked_data = masked_data.drop_vars(['projection', 'crs'], errors='ignore')
            
            # Check for nan
            if 'rainfall' in masked_data:
                nan_count = masked_data['rainfall'].isnull().sum().item()
                logging.info(f"    Total NaN values in 'rainfall': {nan_count} (of {len(masked_data)})")
            else:
                logging.warning("'rainfall' variable not found in dataset!")
            
            logging.info(f'    Appending rainfall data to rainfall_data_all list.\n')
            rainfall_data_all.append(masked_data)
    
    return rainfall_data_all

def load_rainfall_data(rainfall_dir, shape_filepath, processed_output_dir, fig_path, required_crs,
                       pred_frequency, catchment):

    # --- open rainfall files across file time period ---

    rainfall_data_all = _process_rainfall_files(rainfall_dir, catchment,
                                                shape_filepath, required_crs)
        
    # Concatenate all monthly DataArrays into a single xarray.DataArray along the time dimension
    logging.info(f'Concatenating rainfall data along time dimension.')
    full_da = xr.concat(rainfall_data_all, dim='time')

    # Ensure no duplicates
    full_da = full_da.sortby('time')
    if full_da.indexes['time'].has_duplicates:
        logging.warning("Duplicate timestamps found — dropping duplicates.")
        full_da = full_da.sel(time=~full_da.indexes['time'].duplicated())
        
    # --- Aggregate grid data to catchment totals by timestep ---
    logging.info(f"Summing depth per unit area to total {catchment} catchment volume rainfall.")
    total_volume_m3 = (full_da["rainfall"] / 1000 * 1_000_000).sum(
            dim=["projection_y_coordinate", "projection_x_coordinate"])

    # Clarify column name for merging
    total_volume_m3.name = "rainfall_volume_m3"

    # --- Save processed data as csv ---

    # Save as csv
    final_csv_path = f"{processed_output_dir}rainfall_daily_catchment_sum.csv"
    logging.info(f"Saving catchment-summed daily rainfall to CSV: {final_csv_path}")
    
    # Save as csv to merge into main model df -> No return here, just saved to access later  
    total_volume_m3.to_dataframe().to_csv(final_csv_path)

    # Save as ties series graph to view
    _save_haduk_graph(final_csv_path, fig_path, pred_frequency, catchment)

# Load in various ERA5-Land API features

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

def _compute_weighted_aggregation_1D(full_da, feat_name, aggregation_type):
    """
    Aggregate 1D grid data from ERA5-Land datasets by sum or mean average.
    """
    # Debug prints:
    logging.info(f"DEBUG: Entering _compute_weighted_aggregation_1D for {feat_name}")
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

def _combine_and_aggregate_timestep_data(all_timestep_dataarrays, processed_output_dir,start_year, end_year,
                                      feat_name, aggregation_type, pred_frequency, catchment):
    # Concatenate all monthly DataArrays into a single xarray.DataArray along the time dimension
    full_da = xr.concat(all_timestep_dataarrays, dim='time')

    # Save the combined data to a single NetCDF file for the entire period
    final_nc_path = f"{processed_output_dir}{feat_name}_{pred_frequency}_{start_year}-{end_year}_era5land.nc"
    logging.info(f"Saving combined {pred_frequency} {feat_name} from {start_year}-{end_year} to: {final_nc_path}")
    full_da.to_netcdf(final_nc_path)
    
    logging.info(f"ERA5-Land {feat_name} data retrieval and processing complete.")

    # --- Save processed data as csv ---

    # Save as csv
    final_csv_path = f"{processed_output_dir}{feat_name}_{pred_frequency}_catchment_{aggregation_type}.csv"
    logging.info(f"Saving catchment-summed {pred_frequency} {feat_name} to CSV: {final_csv_path}")
    
    # Compute total volume loss for area using weighted area per grid cell (not uniform)
    catchment_sum_data = _compute_weighted_aggregation_1D(full_da, feat_name, aggregation_type)
    
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

def _apply_catchment_mask(catchment_polygon, timestep_data_sliced):
    """
    Mask from whole bounding box to catchment polygon before aggregating to catchment
    totals to avoid external bounding box data leaking into polygon.
    """
    # Ensure the polygon is in the correct CRS for masking against lon/lat data
    polygon_4326 = catchment_polygon.to_crs("EPSG:4326").copy()
    
    mask_da = regionmask.mask_geopandas(
        polygon_4326,
        timestep_data_sliced.longitude.copy(), # Pass xarray.DataArray for coordinates, not just values
        timestep_data_sliced.latitude.copy()   # Pass xarray.DataArray for coordinates, not just values
    )
    
    # Ensure consistent naming ('latitude', 'longitude') before applying the mask
    if 'lat' in mask_da.dims and 'lon' in mask_da.dims:
        mask_da = mask_da.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif 'latitude' not in mask_da.dims and 'longitude' not in mask_da.dims:
        pass # Or raise an error if expected dims are missing
    
    # Apply the 2D mask to the df (Xarray handles broadcasting)
    masked_data = timestep_data_sliced.where(~np.isnan(mask_da))
    
    return masked_data

def _call_era5_api(start_year, end_year, start_date, end_date, total_months, raw_output_dir, north,
                  west, south, east, c, era5_feat, feat_name, era5_long, aggregation_type):
    # Initialise API call counter
    call_count = 0
    
    # Get feature data (aet: mm/timestep, temp: av degrees, snow: ?)
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

def _resample_by_time_and_step(ds, era5_feat, aggregation_type, pred_frequency):
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
        
    # Get model frequency
    frequency_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}
    clean_pred_frequency = pred_frequency.lower().strip()
    
    frequency = frequency_map.get(clean_pred_frequency)
    if frequency is None:
        raise ValueError(f"Invalid prediction frequency: {pred_frequency}. Must be 'daily', "
                         f"'weekly', or 'monthly'.")
    
    # Return timestep frequency data aggregated by type
    if aggregation_type == 'sum':
        return transformed_data.resample(valid_time=frequency).sum()
    elif aggregation_type == 'mean':
        return transformed_data.resample(valid_time=frequency).mean()
    else:
        raise ValueError(f"No specific aggregation defined for type: {aggregation_type}")
        
def _save_era5_graph(csv_path, csv_name, fig_path, feat_name, catchment,
                     aggregation_type, pred_frequency):
    # Load the processed CSV
    filepath = os.path.join(csv_path, csv_name)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
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
    plt.title(f"{pred_frequency} {catchment} Catchment - {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    # Save figure to results
    plt.savefig(fig_path)

def _process_local_grib_files(raw_output_dir, catchment_polygon, all_timestep_dataarrays,
                              era5_feat, aggregation_type, feat_name, pred_frequency):
    
    for fname in sorted(os.listdir(raw_output_dir)):
        if fname.endswith(".grib"):
            grib_path = os.path.join(raw_output_dir, fname)
            logging.info(f"Processing GRIB: {fname}")
            
            # Aggregate to timestep frequency totals and mask to polygon boundings not bbox
            try:
                ds = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={"decode_timedelta": True})
                timestep_data = _resample_by_time_and_step(ds, era5_feat, aggregation_type, pred_frequency)
                
                # Extract year and month from the filename (e.g., aet_2024_04_era5land.grib)
                parts = fname.split('_')
                current_year = int(parts[-3])
                current_month = int(parts[-2])
                
                # # Define the start and end of the actual data for the current month (to avoid introducing 0's)
                last_day_of_month = calendar.monthrange(current_year, current_month)[1]
                month_start_date = dt.datetime(current_year, current_month, 1)
                month_end_date = dt.datetime(current_year, current_month, last_day_of_month)
                
                # Slice timestep_data using these dates and mask
                timestep_data_sliced = timestep_data.sel(valid_time=slice(month_start_date, month_end_date))
                masked = _apply_catchment_mask(catchment_polygon, timestep_data_sliced)
                
                # Clean and ensure consistent structure before appending to all_{pred_frequency}_aet_dataarrays
                masked = masked.drop_vars(['number', 'surface'], errors='ignore')
                masked = masked.rename({'valid_time': 'time'})
                masked = masked.astype('float32')
                
                all_timestep_dataarrays.append(masked)  # Append masked AET DataArray to monthly list
                
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
                  processed_output_dir: str, csv_path: str, csv_name: str, fig_path: str, pred_frequency: str,
                  era5_feat: str = 'e', era5_long: str = 'total_evaporation', feat_name: str = 'aet',
                  aggregation_type: str = 'sum'):    
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

    # List to hold processed xarray DataArrays from each year
    all_timestep_dataarrays = []

    # Ensure the output directories exist
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)

    # Only run API call if specified (Warning: Multi-hour call)
    if run_era5_land_api:
        
        # NOTE: Calling API in parallel does not speed up performance due to ERA5 internal queuing.
        _call_era5_api(start_year, end_year, start_date, end_date, total_months, raw_output_dir,
                       north, west, south, east, c, era5_feat, feat_name, era5_long, aggregation_type)
    
    # Always process data
    _process_local_grib_files(raw_output_dir, catchment_polygon, all_timestep_dataarrays,
                              era5_feat, aggregation_type, feat_name, pred_frequency)
    
    if all_timestep_dataarrays:
        
        # --- Aggregate to pred frequency values for catchment ---
        catchment_agg = _combine_and_aggregate_timestep_data(
            all_timestep_dataarrays, processed_output_dir,start_year, end_year,
            feat_name, aggregation_type, pred_frequency, catchment)
        
        # --- Save time series data to results ---
        _save_era5_graph(csv_path, csv_name, fig_path, feat_name, catchment, aggregation_type, pred_frequency)

    else:
        
        logging.info(f"No {feat_name} data was retrieved or processed.\n")

# DEFRA streamflow ingestion and preprocessing

def _get_daily_flow_measure_uri(station_id: str, station_name: str):
    """
    Call DEFRA API and get daily average flow specific URI for the targeet station.
    """
    measures_url = f"https://environment.data.gov.uk/hydrology/id/stations/{station_id}/measures"
    
    params = {
        'parameterName': 'Flow',
        'periodName': 'daily'
    }
    
    logger.info(f"Retrieving daily mean flow URI for station: {station_name}")
    
    try:
        response = requests.get(measures_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Look for the measure URI in the API response
        if data and 'items' in data and data['items']:
            # Iterate through the returned daily flow measures to find the mean
            for item in data['items']:
                measure_uri = item.get('@id', '')
                if "flow-mean" in measure_uri:
                    logger.info(f"  Found URI: {measure_uri}")
                    return measure_uri
                
        return measure_uri

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for station {station_id}. Error: {e}")
        return None
    except (ValueError, IndexError) as e:
        logger.error(f"Failed to process API response for station {station_id}. Error: {e}")
        return None
    
def _download_flow_readings(measure_uri: str, startdate_str: str, enddate_str: str,
                      max_per_request: int = 50000):
    """
    Download streamflow readings for each station from DEFRA Hydrology API within given dates.
    Max requests set at 50000 with pagination used when readings exceeed this.
    """
    # Initialise list to store reading and offset tracker for pagination
    all_readings = []
    offset = 0
    
    # Get final day
    end_date_obj = dt.datetime.strptime(enddate_str, "%Y-%m-%dT%H:%M:%S")
    final_day = end_date_obj + timedelta(days=1)
    final_day_str = final_day.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Define params to use in API call
    params = {
        '_limit': max_per_request,
        '_offset': offset,
        'min-dateTime': startdate_str,
        'max-dateTime': final_day_str
    }
    
    # Call API with defined params
    while True:
        try:
            response = requests.get(f"{measure_uri}/readings", params=params)
            response.raise_for_status()
            readings = response.json().get('items', [])
            
            # If readings are found append them to main list
            if readings:
                df_portion = pd.DataFrame(readings)
                all_readings.append(df_portion)
                logger.debug(f"        Downloaded {len(readings)} readings from offset {offset}.")

                # Check if the readings length == max_readings_per_request (meaning more data might exist)
                if len(readings) < max_per_request:
                    break
                
                # If maximum hit, adjust offset and continue
                else:
                    offset += max_per_request
            else:
                logger.info(f"        No more data found for {measure_uri} from offset {offset}.")
                break
        
        # Return errors for debugging
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {measure_uri} (offset {offset}). Error: {e}")
            return pd.DataFrame()
        except ValueError as e:
            logger.error(f"Failed to decode JSON response for {measure_uri} (offset {offset}). Error: {e}")
            return pd.DataFrame()
    
    # Required incase of pagination needing to be used
    if all_readings:
        final_df = pd.concat(all_readings, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

def _preprocess_final_data(readings_df: pd.DataFrame, output_dir: str, start_date: str,
                           end_date: str, station_name: str, pred_frequency: str, catchment: str):
    if readings_df.empty:
        logger.warning(f"No readings downloaded for station {station_name}. Check download logs.")
        return pd.DataFrame()
    
    # Save raw readings before preprocessing
    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, f"raw_streamflow.csv")
    readings_df['station_name'] = station_name.title().strip()
    readings_df.to_csv(raw_output_path, index=False)
    logger.info(f"Saving {len(readings_df)} raw readings for {station_name} to {raw_output_path}")
    
    # Convert to daily total streamflow
    readings_df['streamflow_total_m3'] = readings_df['value'] * 86400  # to daily total is correct here and aggregated later
    
    # Confirm data length is as expected
    expected_dates = pd.date_range(start=start_date, end=end_date)
    expected_days = len(expected_dates)
    
    # Sort and truncate readings before assigning the date
    readings_df = readings_df.sort_values("dateTime").reset_index(drop=True)
    readings_df = readings_df.iloc[:expected_days]
    
    if len(readings_df) != expected_days:
        logger.warning(f"Length mismatch after slicing. Expected {expected_days}, got {len(readings_df)}.")

    # Assigned expected dates as new index
    readings_df['date'] = expected_dates
    
    # Fill NaNsusing interpolation if only a few
    missing_count = readings_df['streamflow_total_m3'].isna().sum()
    total_count = len(readings_df)
    missing_ratio = missing_count / total_count

    # Check less than 1% missing before filling
    if missing_ratio < 0.01:
        logger.info(f"Filling {missing_count} missing values (<1%) using linear interpolation.")
        readings_df['streamflow_total_m3'] = readings_df['streamflow_total_m3'].interpolate(method='linear')
    else:
        logger.warning(f"Too many missing values in 'streamflow_total_m3' ({missing_ratio:.2%}). Skipping interpolation.")
        
    # Drop unneeded columns (keeping defensive to avoid crash)
    drop_cols = ['measure', 'valid', 'invalid', 'missing', 'completeness',
                'quality', 'station_name', 'value', 'dateTime']
    readings_df = readings_df.drop(columns=[col for col in drop_cols if col in readings_df.columns])
    
    # Ensure datetime index for resampling
    readings_df['date'] = pd.to_datetime(readings_df['date'])
    readings_df = readings_df.set_index('date').sort_index()
    
    # Aggregate to necessary timestep
    frequency_map = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}
    clean_pred_frequency = pred_frequency.lower().strip()
    frequency = frequency_map.get(clean_pred_frequency)
    if frequency is None:
        raise ValueError(f"Invalid prediction frequency: {pred_frequency}. Must be 'daily', "
                         f"'weekly', or 'monthly'.")
    
    # Apply resampling
    readings_df = readings_df.resample(frequency).sum()
    logger.info(f"Total streamflow aggregated to {pred_frequency} timestep.")
    
    # Boxcox transform skewed streamflow data
    readings_df = _transform_skewed_data(readings_df, catchment, 'streamflow_total_m3')
    
    # Rename date to time for merging
    readings_df.index.name = 'time'
    
    logger.info("Streamflow ingestion pipeline complete.")
    return readings_df

def download_and_save_flow_data(station_csv: str, start_date: str, end_date: str, output_dir: str,
                                pred_frequency: str, catchment: str):
    """
    Downloads file using DEFRA hydrology API for flow station specified in flow station csv.
    """
    logger.info("Starting streamflow data pipeline...")
    logging.info(f"Collecting data from {start_date[:-9]} to {end_date[:-9]}\n")

    # Read the station information from the CSV
    try:
        station_df = pd.read_csv(station_csv)
        station_id = station_df.iloc[0]["url_id"]
        station_name = station_df.iloc[0]["station_name"]
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error reading station CSV: {e}")
        return
    
    # Get tmeasure URI  and download selected time series for streamflow
    measure_uri = _get_daily_flow_measure_uri(station_id, station_name)
    if not measure_uri:
        logger.error(f"Could not retrieve a valid measure URI for station {station_name}.")
        return pd.DataFrame()

    readings_df = _download_flow_readings(measure_uri, start_date, end_date)
    readings_df = _preprocess_final_data(readings_df, output_dir, start_date, end_date,
                                         station_name, pred_frequency, catchment)
    
    # Save as csv
    save_path = os.path.join(output_dir, f"{pred_frequency}_streamflow.csv")
    readings_df.to_csv(save_path)
    logger.info(f"{pred_frequency} total streamflow saved to {save_path}.")
    
    return readings_df
