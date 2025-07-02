import sys
import base64
import folium
import logging
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

def plot_interactive_mesh(mesh_nodes_gdf: gpd.geodataframe, catchment_polygon: gpd.GeoDataFrame, map_blue: str,
                          esri: str, esri_attr: str, static_output_path: str, interactive_output_path: str,
                          grid_resolution: int = 1000, interactive: bool = True):
    """
    Build a visualisation of the catchment mesh, as either an interactive Folium map
    with optional base map tiles and layer toggles, or a static Matplotlib plot.

    Args:
        mesh_nodes_gdf (gpd.GeoDataFrame): Node ID, coordinates, and geometry (Point).
        catchment_polygon (gpd.GeoDataFrame): Catchment area polygon (GeoJson).
        map_blue (str): Consisten visualisation colours as defined in config.
        esri (str): esri tile path.
        esri_attr (str): esri tile attribution (required for use).
        interactive (bool): A boolean. True (default) generates an interactive folium
        map.
                            False generates a static matplotlib map.
    """
    logger.info(f"PLOT_INTERACTIVE_MESH: Plotting catchment mesh overlaid on map.")

    # Static map when select
    if not interactive:
        fig, ax = plt.subplots(figsize=(10, 10))
        mesh_nodes_gdf.plot(ax=ax, color=map_blue, markersize=1.5)
        catchment_polygon.plot(ax=ax, facecolor='none', edgecolor=map_blue, linewidth=1)

        plt.xlabel("EPSG:27700 Easting (m)")
        plt.ylabel("EPSG:27700 Northing (m)")
        plt.title("Mesh Centroids for Eden Catchment (Resolution: 1km x 1km)")
        
        full_static_output_path = static_output_path + f'_{grid_resolution}.png'
        
        plt.savefig(full_static_output_path, dpi=300)
        logger.info(f"Static map file saved to: {full_static_output_path}\n")
        return fig
        
    # Interactive map otherwise
    else:
        # Create base map centered on centre of mesh
        map_center = [mesh_nodes_gdf['lat'].mean(), mesh_nodes_gdf['lon'].mean()]

        # Define map tile layers
        map = folium.Map(location=map_center, zoom_start=10, tiles=None)

        # Add tile layers (esri visible by default, others in toggle)
        folium.TileLayer(tiles=esri, attr=esri_attr, name='Topo', show=True).add_to(map)
        folium.TileLayer('CartoDB positron', name='Light', show=False).add_to(map)
        folium.TileLayer('CartoDB dark_matter', name='Dark', show=False).add_to(map)

        # Add all node centroids as circle markers
        mesh_layer = folium.FeatureGroup(name="Mesh Nodes")
        for col, row in mesh_nodes_gdf.iterrows():
            folium.CircleMarker(location=[row['lat'], row['lon']], radius=1, color=map_blue,
                                fill=True, fill_opacity=0.6).add_to(mesh_layer)
            
        # TODO: Add ground-truth data stations as different shape / colour nodes

        # Add solid catchment boundary polygon to the map
        folium.GeoJson(catchment_polygon, name='Catchment Boundary', 
                    style_function=lambda x: {'color': map_blue, 'weight': 2, 'fillColor': map_blue,
                                                'fillOpacity': 0.15}).add_to(map)

        # Add layer control to toggle catchment boundary, mesh and map tiles
        mesh_layer.add_to(map)
        folium.LayerControl().add_to(map)

        full_interactive_output_path = interactive_output_path + f'_{grid_resolution}.html'
        
        # Save to html (unique by timestamp) and display in notebook
        map.save(full_interactive_output_path)
        logger.info(f"Interactive map file saved to: {full_interactive_output_path}\n")
        return map
    
def plot_interactive_mesh_colour_coded(mesh_nodes_gdf: gpd.geodataframe, catchment_polygon: gpd.GeoDataFrame, map_blue: str,
                          esri: str, esri_attr: str, static_output_path: str, interactive_output_path: str,
                          category_colors: dict, category_labels: dict, grid_resolution: int = 1000, interactive: bool = True):
    """
    Build a visualisation of the catchment mesh, as either an interactive Folium map
    with optional base map tiles and layer toggles, or a static Matplotlib plot.
    Nodes are colour coded by catogorical feature.

    """
    logger.info(f"PLOT_INTERACTIVE_MESH: Plotting catchment mesh overlaid on map.")

    # Static map when select
    if not interactive:
        fig, ax = plt.subplots(figsize=(10, 10))
        mesh_nodes_gdf.plot(ax=ax, color=map_blue, markersize=1.5)
        catchment_polygon.plot(ax=ax, facecolor='none', edgecolor=map_blue, linewidth=1)

        plt.xlabel("EPSG:27700 Easting (m)")
        plt.ylabel("EPSG:27700 Northing (m)")
        plt.title("Mesh Centroids for Eden Catchment (Resolution: 1km x 1km)")
        
        full_static_output_path = static_output_path + f'_{grid_resolution}.png'
        
        plt.savefig(full_static_output_path, dpi=300)
        logger.info(f"Static map file saved to: {full_static_output_path}\n")
        return fig
        
    # Interactive map otherwise
    else:
        # Create base map centered on centre of mesh
        map_center = [mesh_nodes_gdf['lat'].mean(), mesh_nodes_gdf['lon'].mean()]

        # Define map tile layers
        map = folium.Map(location=map_center, zoom_start=10, tiles=None)

        # Add tile layers (esri visible by default, others in toggle)
        folium.TileLayer(tiles=esri, attr=esri_attr, name='Topo', show=True).add_to(map)
        folium.TileLayer('CartoDB positron', name='Light', show=False).add_to(map)
        folium.TileLayer('CartoDB dark_matter', name='Dark', show=False).add_to(map)

        # Add all node centroids as circle markers
        mesh_layer = folium.FeatureGroup(name="Mesh Nodes")
        for _, row in mesh_nodes_gdf.iterrows():
            land_code = row.get('land_cover_code', None)
            color = category_colors.get(land_code, "gray")  # fallback if code missing
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=1.5,
                color=color,
                fill=True,
                fill_opacity=0.6,
                tooltip=category_labels.get(land_code, "Unknown")
            ).add_to(mesh_layer)
            
        # TODO: Add ground-truth data stations as different shape / colour nodes

        # Add solid catchment boundary polygon to the map
        folium.GeoJson(catchment_polygon, name='Catchment Boundary', 
                    style_function=lambda x: {'color': map_blue, 'weight': 2, 'fillColor': map_blue,
                                                'fillOpacity': 0.15}).add_to(map)

        # Add layer control to toggle catchment boundary, mesh and map tiles
        mesh_layer.add_to(map)
        folium.LayerControl().add_to(map)

        full_interactive_output_path = interactive_output_path + f'_{grid_resolution}.html'
        
        # Save to html (unique by timestamp) and display in notebook
        map.save(full_interactive_output_path)
        logger.info(f"Interactive map file saved to: {full_interactive_output_path}\n")
        return map

def plot_directional_mesh(directional_edge_weights_gdf: gpd.GeoDataFrame, catchment_polygon: gpd.GeoDataFrame,
                          output_path: str, esri: str, esri_attr: str, grid_resolution: int = 1000):
    """
    Plots an interactive Folium map with each node represented by an arrow indicating
    the dominant slope direction.
    """
    logger.info(f"PLOT_DIRECTIONAL_MESH: Plotting catchment mesh with directional arrows.")

    map_center = [directional_edge_weights_gdf['lat'].mean(), directional_edge_weights_gdf['lon'].mean()]
    map = folium.Map(location=map_center, zoom_start=10, tiles=None)

    folium.TileLayer(tiles=esri, attr=esri_attr, name='Topo', show=True).add_to(map)
    folium.TileLayer('CartoDB positron', name='Light', show=False).add_to(map)
    folium.TileLayer('CartoDB dark_matter', name='Dark', show=False).add_to(map)

    arrow_layer = folium.FeatureGroup(name="Slope Directions")

    # Define a simple SVG arrow icon (Path drawing a thin upward arrow)
    arrow_svg_template = """
    <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <g transform="rotate({angle}, 12, 12)">
        <path d="M12 2 L17 8 H14 V22 H10 V8 H7 L12 2 Z" fill="{color}" />
    </g>
    </svg>
    """
    
    # Get a cyclical (hue, saturation, value) colourmap instance to define using circular angle
    cmap = cm.get_cmap('hsv')

    # Loop through mesh nodes in catchmend
    for _, row in directional_edge_weights_gdf.iterrows():
        lat, lon = row['lat'], row['lon']
        dx, dy = row['mean_slope_dx'], row['mean_slope_dy']
        
        # ND: +Y is North, +X is East
        angle_rad_for_folium_rotation = np.arctan2(dx, dy) # Angl efrom north to east
        angle_deg_for_folium_rotation = np.degrees(angle_rad_for_folium_rotation)

        # Ensure angle is 0-360 (could be negative from arctan2)
        normalised_angle = (angle_deg_for_folium_rotation + 360) % 360
        
        # Map the angle (0-360) to a colour in the colourmap (0-1 range)
        color_rgb = cmap(normalised_angle / 360.0) # RGBA tuple
        hex_color = mcolors.rgb2hex(color_rgb) # Convert to hex str

        # Create SVG string with dynamic colour and direction weighting rotation
        svg = arrow_svg_template.format(color=hex_color, angle=angle_deg_for_folium_rotation)
        encoded_svg = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        
        icon = folium.CustomIcon(icon_image=f"data:image/svg+xml;base64,{encoded_svg}",
                                 icon_size=(12, 12), icon_anchor=(6, 6))

        folium.Marker(location=[lat, lon], icon=icon).add_to(arrow_layer)

    # Catchment Outline
    folium.GeoJson(catchment_polygon, name='Catchment Boundary', 
                   style_function=lambda x: {'color': 'black', 'weight': 2, 'fillColor': 'grey',
                                            'fillOpacity': 0.15}).add_to(map)
    
    # Turn layer on and off to check map under clearly
    arrow_layer.add_to(map)
    folium.LayerControl().add_to(map)
    
    # Save by mesh resolution
    full_interactive_output_path = output_path + f'_{grid_resolution}.html'
    map.save(full_interactive_output_path)
    logger.info(f"Interactive directional map file saved to: {full_interactive_output_path}\n")
    
    return map

def plot_interactive_mesh_with_stations(mesh_nodes_gdf: gpd.GeoDataFrame, catchment_polygon: gpd.GeoDataFrame,
                          map_blue: str, esri: str, esri_attr: str, static_output_path: str,
                          interactive_output_path: str, catchment: str, grid_resolution: int = 1000,
                          interactive: bool = True, stations_gdf: gpd.GeoDataFrame = None,
                          station_color: str = 'orange', station_marker_icon: str = 'info-sign'):
    """
    Build a visualisation of the catchment mesh, as either an interactive Folium map
    with optional base map tiles and layer toggles, or a static Matplotlib plot.
    """
    logger.info(f"PLOT_INTERACTIVE_MESH: Plotting catchment mesh overlaid on map.")

    # Static map when select
    if not interactive:
        fig, ax = plt.subplots(figsize=(10, 10))
        mesh_nodes_gdf.plot(ax=ax, color=map_blue, markersize=1.5)
        catchment_polygon.plot(ax=ax, facecolor='none', edgecolor=map_blue, linewidth=1)

        # Add stations to static map if provided
        if stations_gdf is not None:
            stations_gdf.plot(ax=ax, color=station_color, marker='o', markersize=20, label='Stations')
            ax.legend() # Show legend if stations are plotted

        plt.xlabel("EPSG:27700 Easting (m)")
        plt.ylabel("EPSG:27700 Northing (m)")
        plt.title(f"Mesh Centroids for {catchment} Catchment (Resolution: {grid_resolution}m x {grid_resolution}m)")
        
        full_static_output_path = static_output_path + f'_{grid_resolution}.png'
        
        plt.savefig(full_static_output_path, dpi=300)
        logger.info(f"Static map file saved to: {full_static_output_path}\n")
        return fig
        
    # Interactive map otherwise
    else:
        # Create base map centered on centre of mesh
        map_center = [mesh_nodes_gdf['lat'].mean(), mesh_nodes_gdf['lon'].mean()]

        # Define map tile layers
        map_obj = folium.Map(location=map_center, zoom_start=10, tiles=None) # Renamed 'map' to 'map_obj' to avoid conflict

        # Add tile layers (esri visible by default, others in toggle)
        folium.TileLayer(tiles=esri, attr=esri_attr, name='Topo', show=True).add_to(map_obj)
        folium.TileLayer('CartoDB positron', name='Light', show=False).add_to(map_obj)
        folium.TileLayer('CartoDB dark_matter', name='Dark', show=False).add_to(map_obj)

        # Add all node centroids as circle markers
        mesh_layer = folium.FeatureGroup(name="Mesh Nodes")
        for idx, row in mesh_nodes_gdf.iterrows():
            folium.CircleMarker(location=[row['lat'], row['lon']], radius=1, color=map_blue,
                                fill=True, fill_opacity=0.6,
                                tooltip=f"Node ID: {row['node_id']}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}"
                                ).add_to(mesh_layer)
            
        # Add mesh to map as layer
        mesh_layer.add_to(map_obj)
        
        # Select marker type
        station_marker_icon = 'map-marker'
        
        # Add stations as clearly differentiated markers
        if stations_gdf is not None:
            stations_gdf_wgs84 = stations_gdf.to_crs(epsg=4326) # Convert to Lat/Lon

            station_layer = folium.FeatureGroup(name="Groundwater Stations")
            for idx, row in stations_gdf_wgs84.iterrows():

                folium.Marker(
                    location=[row.geometry.y, row.geometry.x], # Latitude, Longitude
                    icon=folium.Icon(color=station_color, icon=station_marker_icon, prefix='fa'), # Using a different icon/color from https://fontawesome.com/v4/cheatsheet/
                    tooltip=f"Station: {row['station_name']}<br>ID: {row['station_id']}<br>Easting: {row['easting']}<br>Northing: {row['northing']}"
                ).add_to(station_layer)
                
            station_layer.add_to(map_obj)

        # Add solid catchment boundary polygon to the map
        catchment_polygon_wgs84 = catchment_polygon.to_crs(epsg=4326)
        folium.GeoJson(catchment_polygon_wgs84, name='Catchment Boundary',
                    style_function=lambda x: {'color': map_blue, 'weight': 2, 'fillColor': map_blue,
                                                'fillOpacity': 0.15}).add_to(map_obj)

        # Add layer control to toggle catchment boundary, mesh and map tiles
        folium.LayerControl().add_to(map_obj)

        full_interactive_output_path = interactive_output_path + f'_{grid_resolution}.html'
        
        # Save to html (unique by timestamp) and display in notebook
        # map_obj.save(full_interactive_output_path)
        # logger.info(f"Interactive map file saved to: {full_interactive_output_path}\n")
        
        return map_obj
    