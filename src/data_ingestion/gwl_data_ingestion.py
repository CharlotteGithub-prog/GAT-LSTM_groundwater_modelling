import logging
import pandas as pd
from pyproj import Transformer

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

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
