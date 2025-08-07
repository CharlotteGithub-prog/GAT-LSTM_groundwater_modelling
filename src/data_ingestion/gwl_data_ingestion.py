import os
import sys
import ast
import logging
import requests
import pandas as pd
from pyproj import Transformer

from src.data_ingestion.spatial_transformations import grid_ref_to_coords

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

# Process station catchments list into inital spatial data csv
def process_station_coordinates(os_grid_squares: str, station_list_input: str,
    station_list_output: str, catchment: str):
    """
    Reads a station list with OS grid references, converts them to coordinates,
    and saves the enriched dataframe.
    """
    logger.info(f"[{catchment}] Starting coordinate processing for station list...\n")

    # Use lookup file to create a dict of zipped grid letter and value lookup objects
    grid_lookup_df = pd.read_csv(os_grid_squares)
    grid_letters = dict(zip(grid_lookup_df['grid_letters'], zip(grid_lookup_df['easting_base'], grid_lookup_df['northing_base'])))
    logger.info(f"[{catchment}] Loaded OS grid squares lookup from: {os_grid_squares}")

    # Load station list
    stations_df = pd.read_csv(station_list_input)
    logger.info(f"[{catchment}] Loaded station list from: {station_list_input}")

    # Apply coordinate conversion using a lambda to pass grid_letters into the apply function's scope
    coords = stations_df['grid_ref'].apply(lambda ref: grid_ref_to_coords(ref, grid_letters))
    stations_df[['easting', 'northing', 'lat', 'lon']] = coords
    logger.info(f"[{catchment}] Converted OS grid references to coordinates for {len(stations_df)} stations.\n")

    # Save processed df as csv and check head and length
    stations_df.to_csv(station_list_output, index=False)
    logger.info(f"[{catchment}] Saved processed station list to: {station_list_output}")
    logger.info(f"Station location reference table head:\n\n{stations_df.head()}\n")
    logger.info(f"Total Stations: {len(stations_df)}")

    logger.info(f"[{catchment}] Coordinate processing for station list complete.\n")
    return stations_df

# Call DEFRA API and retrieve metadata by station
def _get_station_metadata(wiski_id: str, base_url: str):
    
    params = {'wiskiID': wiski_id}
    response = requests.get(base_url, params=params)
    
    # If response is good (200) return metadata items as single metadata column
    try:
        response.raise_for_status()
        data = response.json()
        
        if data and 'items' in data and data['items']:
            return data['items'][0]
        else:
            logger.warning(f"No metadata items found or unexpected response structure for station.")
            return None
       
    # Return appropriate errors for debugging
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None
    except ValueError as e: # Catches JSON decoding errors
        logger.error(f"Failed to decode JSON response: {e}")
        return None

# Call DEFRA API and retrieve measures data by station
def _get_station_measures(row: pd.Series):
    """
    See https://environment.data.gov.uk/hydrology/doc/reference#measures-summary for measures data
    """
    # Exract ID and metadata
    metadata = row.get('metadata')
    station_id = row.get('station_id')
    measures_url = f"{metadata['@id']}/measures"

    # Continue if the required things exist in the correct form
    if not isinstance(metadata, dict) or '@id' not in metadata:
        logger.warning(f"Station {station_id} has no valid metadata dictionary or missing '@id' for measures data.")
        return []
        
    # Request the measures data
    try:
        response = requests.get(measures_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('items', []) # Return the 'items' list, or an empty list if 'items' is missing
    
    # Return appropriate errors for debugging
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for station {station_id} at {measures_url}. Error: {e}")
        return []
    except ValueError as e:  # If response.json() fails
        logger.error(f"Failed to decode JSON response for measures of station {station_id}. Error: {e}")
        return []

# Process both API station data calls and flatten into csv
def fetch_and_process_station_data(stations_df: pd.DataFrame, base_url: str, output_path: str):
    """
    Fetches metadata and measures for a DataFrame of stations, processes it, and saves the result.
    """
    logger.info("Fetching station metadata from DEFRA API...")
    
    # --- API CALL 1: Get Metadata
    
    stations_df['metadata'] = stations_df['station_id'].apply(
        lambda id: _get_station_metadata(id, base_url)
    )
    
    # Convert metadata from string to dict using ast (API returns dict as str)
    stations_df['metadata'] = stations_df['metadata'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    logger.info("Fetching station measures data from DEFRA API...")
    
    # --- API CALL 2: Get Measures

    # Apply _get_station_measures using df.apply with axis=1 to pass the whole row
    stations_df['measures'] = stations_df.apply(lambda row: _get_station_measures(row), axis=1)
    logger.info("Extracting flattened columns from metadata and measures...\n")
    
    # Convert measures from string to dict using ast
    stations_df['metadata'] = stations_df['metadata'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Extract initial name and measure_uri info as flattened columns
    stations_df['station_name'] = stations_df['metadata'].apply(
        lambda x: x.get('label') if isinstance(x, dict) else None)
    stations_df['measure_uri'] = stations_df['measures'].apply(
        lambda x: x[0].get('@id') if x and isinstance(x, list) and x else None)
    
    # --- Save flattened df to csv

    logger.info(f"Saving processed station data to: {output_path}")
    stations_df.to_csv(output_path, index=False)

    return stations_df

# Call DEFRA API and retrieve raw gwl timeseries station data
def _download_gwl_ts_readings(measure_uri: str, startdate_str: str, enddate_str: str,
                      max_per_request: int = 50000):
    """
    Download hydrological readings for each station from DEFRA Hydrology API within given dates.
    Max requests set at 50000 with pagination used when readings exceeed this.
    """
    all_measure_readings = []
    offset = 0
    
    if not isinstance(measure_uri, str) or not measure_uri.startswith("http"):
        logger.error(f"Invalid URI provided: {measure_uri}. Must be a valid HTTP URL string.")
        return pd.DataFrame()
    
    while True:

        params = {
            '_limit': max_per_request,
            '_offset': offset,  # offset for when number of readings exceeds max_per_request (pagination)
            'min-dateTime': startdate_str,
            'max-dateTime': enddate_str
        }
        
        try:
            # Call API with defined parameters
            response = requests.get(f"{measure_uri}/readings", params=params)
            response.raise_for_status()
            
            readings = response.json().get('items', [])
            
            # If readings are found append them to all_measure_readings list
            if readings:
                df_portion = pd.DataFrame(readings)
                all_measure_readings.append(df_portion)
                logger.debug(f"        Downloaded {len(readings)} readings from offset {offset}.")

                # Check if the max_readings_per_request was received (indicating more data might exist)
                if len(readings) < max_per_request:
                    break  # No more readings
                else:
                    offset += max_per_request # Move offset for next chunk to retrieve
            else:
                logger.info(f"        No more data found for {measure_uri} from offset {offset}.")
                break
        
        # Return appropriate errors for debugging
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {measure_uri} (offset {offset}). Error: {e}")
            return pd.DataFrame() # Return empty DataFrame if failure
        except ValueError as e:
            logger.error(f"Failed to decode JSON response for {measure_uri} (offset {offset}). Error: {e}")
            return pd.DataFrame() # Return empty DataFrame if failure
        
    if all_measure_readings:
        final_df = pd.concat(all_measure_readings, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

# Orchestrate API calls for all stations in catchment
def download_and_save_station_readings(stations_df: pd.DataFrame, start_date: str,
                                       end_date: str, gwl_data_output_dir: str):
    """
    Iterates through stations, downloads their readings, and saves them to individual CSV files.
    Returns the original stations_df.
    """
    logging.info(f"Collecting data from {start_date[:-9]} to {end_date[:-9]}...\n")
    count = 1
    
    # Ensure output directory exists in project architecutre
    os.makedirs(gwl_data_output_dir, exist_ok=True)
    
    # Use .copy() to avoid SettingWithCopyWarning
    processed_stations_df = stations_df.copy()

    # Pull gwl timeseries data station by station from DEFRA API
    for index, row in processed_stations_df.iterrows():
        uri = row['measure_uri']
        station_name = row['station_name']
        measure_id = uri.split("/")[-1]
        
        # Define the full output path for each CSV
        output_csv_path = os.path.join(gwl_data_output_dir, f"{measure_id}_readings.csv")
        
        logging.info(f"({count}/{len(processed_stations_df)}) Processing measure: {uri+'/readings'}")
        
        # Download timeseries data and assign name
        df_readings = _download_gwl_ts_readings(uri, start_date, end_date)
        
        # If valid response then save to csv
        if not df_readings.empty:
            df_readings['station_name'] = station_name.title().strip()
            logger.info(f"    Saving {len(df_readings)} readings for {station_name} to {output_csv_path}\n")
            df_readings.to_csv(output_csv_path)
        else:
            logger.warning(f"    No readings downloaded for station {station_name} ({uri}). Skipping save.\n")
        
        count += 1

