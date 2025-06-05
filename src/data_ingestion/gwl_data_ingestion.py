import ast
import logging
import requests
import pandas as pd
from pyproj import Transformer

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

# Process station catchments list into inital spatial data csv
def process_station_coordinates(os_grid_squares: str, station_list_input: str,
    station_list_output: str, catchment_name: str):
    """
    Reads a station list with OS grid references, converts them to coordinates,
    and saves the enriched dataframe.
    """
    logger.info(f"[{catchment_name}] Starting coordinate processing for station list...\n")

    # Use lookup file to create a dict of zipped grid letter and value lookup objects
    grid_lookup_df = pd.read_csv(os_grid_squares)
    grid_letters = dict(zip(grid_lookup_df['grid_letters'], zip(grid_lookup_df['easting_base'], grid_lookup_df['northing_base'])))
    logger.info(f"[{catchment_name}] Loaded OS grid squares lookup from: {os_grid_squares}")

    # Load station list
    stations_df = pd.read_csv(station_list_input)
    logger.info(f"[{catchment_name}] Loaded station list from: {station_list_input}")

    # Apply coordinate conversion using a lambda to pass grid_letters into the apply function's scope
    stations_df[['easting', 'northing', 'lat', 'lon']] = stations_df['grid_ref'].apply(
        lambda grid_ref: grid_ref_to_coords(grid_ref, grid_letters)
    )
    logger.info(f"[{catchment_name}] Converted OS grid references to coordinates for {len(stations_df)} stations.\n")

    # Save processed df as csv and check head and length
    stations_df.to_csv(station_list_output, index=False)
    logger.info(f"[{catchment_name}] Saved processed station list to: {station_list_output}")
    logger.info(f"Station location reference table head:\n\n{stations_df.head()}\n")
    logger.info(f"Total Stations: {len(stations_df)}")

    logger.info(f"[{catchment_name}] Coordinate processing for station list complete.\n")
    return stations_df

# Convert alphanumeric OS grid ref to easting, northing, lat, lon
def grid_ref_to_coords(grid_ref: str, grid_letters: dict):
    """
    Convert OS Grid references into espg 27700 easting and northing values, then into
    espg 4326 coordinate ref system (longitude and latitude) for future visualisaations.
    """
    # Transformer to WGS84 (for lat/long coordinate transformations)
    transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
    
    # Clean grid references
    grid_ref = grid_ref.strip().upper()
    
    # Split to letter and numeric
    letter_only = grid_ref[:2]  # e.g. "NY" (from NY123456)
    numeric_only = grid_ref[2:]  # e.g. "123456" (from NY123456)
    
    # Check expected form (paired values)
    if len(numeric_only) % 2 != 0:
        raise ValueError(f"Invalid grid reference: {grid_ref}")
    
    easting_base, northing_base = grid_letters[letter_only]  # e.g. "300000, 500000" from NY (check ref file)
    half = len(numeric_only) // 2  # Seperate easting and northing (e.g. "123", "456")
    easting_offset = int(numeric_only[:half].ljust(5, '0'))  # e.g. "123" -> 12300 m
    northing_offset = int(numeric_only[half:].ljust(5, '0'))  # e.g. "456" -> 45600 m
    
    easting = int(easting_base + easting_offset)  # e.g. 300000 + 12300 = 312300 m
    northing = int(northing_base + northing_offset)  # e.g. 500000 + 45600 = 545600 m
    lon, lat = transformer.transform(easting, northing)  # Transform 312300, 545600 to lat, long (epsg:4326)
    
    return pd.Series([easting, northing, lat, lon])

# Call DEFRA API and retrieve metadata by station
def get_station_metadata(wiski_id: str, base_url: str):
    
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
def get_station_measures(row: pd.Series):
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
        lambda id: get_station_metadata(id, base_url)
    )
    
    # Convert metadata from string to dict using ast (API returns dict as str)
    stations_df['metadata'] = stations_df['metadata'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    logger.info("Fetching station measures data from DEFRA API...")
    
    # --- API CALL 2: Get Measures

    # Apply get_station_measures using df.apply with axis=1 to pass the whole row
    stations_df['measures'] = stations_df.apply(lambda row: get_station_measures(row), axis=1)
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
