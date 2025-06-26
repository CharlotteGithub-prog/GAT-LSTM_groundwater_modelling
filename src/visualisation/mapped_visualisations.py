import sys
import folium
import logging
import geopandas as gpd
import matplotlib.pyplot as plt

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
    