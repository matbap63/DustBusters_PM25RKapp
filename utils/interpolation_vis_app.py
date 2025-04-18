import matplotlib.pyplot as plt
import seaborn as sns
import os
import folium
from folium.raster_layers import ImageOverlay
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd

def get_color(value, norm=None):
    """
    Determines a color based on a given value of PM2.5 using a shared normalization.

    (Inputs):
    - value: the value to visualize
    - norm: the color scale to use. Defaults to (3, 90) if none is provided

    (Outputs):
    A color based on the value
    """
    if norm is None:
        norm = mcolors.Normalize(vmin=3, vmax=90) 
    cmap = cm.get_cmap('magma_r')
    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba)


def plot_interpolation(pm25_df, date, core_data, long, lat):
    """
    Plots the Regression Kriging interpolation results on a map in an HTML file.

    (Inputs):
    - pm25_df: dataframe with PM2.5 values
    - interp: interpolation results

    (Outputs):
    - map_hstn: a map of Houston with interpolation overlaid
    """
    core_data['datetime'] = pd.to_datetime(core_data['datetime'], errors='coerce')
    core_data = core_data[core_data['datetime'].dt.date == date]

    interp = pm25_df['predictions'].values.reshape(100, 100)

    min_long = -96.09033
    max_long = -94.51128290000001
    min_lat = 28.910556
    max_lat = 30.450355000000002

    # Plot PM2.5 values. We will overlay this image on the folium map
    # Note: the minimum and maximum arguments should match that of the "get_color" function
    norm = mcolors.Normalize(vmin=3, vmax=90)

    fig, ax = plt.subplots(figsize=(8,8))

    cax = ax.imshow(interp,
                    extent=[min_long,
                            max_long,
                            min_lat,
                            max_lat],
                    origin='lower',
                    cmap='magma_r',
                    alpha=1,
                    norm = norm)
    
    # Add color bar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.036, pad=0.04)
    cbar.set_label('PM2.5 (µg/m³)')

    # Remove axes
    ax.axis('off')

    # Save the image
    fig.savefig('kriging_interpolation.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.036, pad=0.04).set_label('PM2.5 (µg/m³)')
    ax.axis('off')

    # Create map over Houston
    map_hstn = folium.Map(location=[29.76, -95.37], zoom_start=10)
    bounds = [[min_lat, min_long], [max_lat, max_long]]
    ImageOverlay(image='kriging_interpolation.png', bounds=bounds, opacity=.5, zindex=1).add_to(map_hstn)

    #Add sensors
    for _, row in core_data.iterrows():
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(html=f'<div style="font-size:12px; color:black; \
                                background-color:{get_color(row["pm2.5"])}; \
                                border-radius:50%; width:30px; height:30px; display:flex; \
                                justify-content:center; align-items:center;">\
                                {int(row["pm2.5"])}</div>'),
            popup=f"{row['sensor_name']}<br>{row['monitor_type']}<br>PM2.5: {row['pm2.5']:.2f}"
        ).add_to(map_hstn)

    #Add user's coordinates
    folium.Marker(
        location=[lat, long],
        popup=f"Your Location:<br>({lat:.4f}, {long:.4f})",
        icon=folium.Icon(color='red', icon='user')
    ).add_to(map_hstn)

    filename = f'./houston_kriging_map.html'
    map_hstn.save(filename)
    return map_hstn

def plot_interpolation_relative(pm25_df, date, core_data, long, lat):
    """
    Plots the Regression Kriging interpolation results on a map in an HTML file.

    (Inputs):
    - pm25_df: dataframe with PM2.5 values
    - interp: interpolation results

    (Outputs):
    - map_hstn: a map of Houston with interpolation overlaid
    """
    core_data['datetime'] = pd.to_datetime(core_data['datetime'], errors='coerce')
    core_data = core_data[core_data['datetime'].dt.date == date]

    interp = pm25_df['predictions'].values.reshape(100, 100)

    min_long = -96.09033
    max_long = -94.51128290000001
    min_lat = 28.910556
    max_lat = 30.450355000000002

    # Plot PM2.5 values. We will overlay this image on the folium map
    # Note: the minimum and maximum arguments should match that of the "get_color" function
    vmin = pm25_df['predictions'].min()
    vmax = pm25_df['predictions'].max() + (2 * pm25_df['predictions'].std())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8,8))

    cax = ax.imshow(interp,
                    extent=[min_long,
                            max_long,
                            min_lat,
                            max_lat],
                    origin='lower',
                    cmap='magma_r',
                    alpha=1,
                    norm = norm)
    
    # Add color bar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.036, pad=0.04)
    cbar.set_label('PM2.5 (µg/m³)')

    # Remove axes
    ax.axis('off')

    # Save the image
    fig.savefig('kriging_interpolation.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.036, pad=0.04).set_label('PM2.5 (µg/m³)')
    ax.axis('off')

    # Create map over Houston
    map_hstn = folium.Map(location=[29.76, -95.37], zoom_start=10)
    bounds = [[min_lat, min_long], [max_lat, max_long]]
    ImageOverlay(image='kriging_interpolation.png', bounds=bounds, opacity=.5, zindex=1).add_to(map_hstn)

    #Add sensors
    for _, row in core_data.iterrows():

        background_color = get_color(row["pm2.5"], norm=norm)

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(html=f'<div style="font-size:12px; color:black; \
                                background-color:{background_color}; \
                                border-radius:50%; width:30px; height:30px; display:flex; \
                                justify-content:center; align-items:center;">\
                                {int(row["pm2.5"])}</div>'),
            popup=f"{row['sensor_name']}<br>{row['monitor_type']}<br>PM2.5: {row['pm2.5']:.2f}"
        ).add_to(map_hstn)

    #Add user's coordinates
    folium.Marker(
        location=[lat, long],
        popup=f"Your Location:<br>({lat:.4f}, {long:.4f})",
        icon=folium.Icon(color='red', icon='user')
    ).add_to(map_hstn)

    filename = f'./houston_kriging_map.html'
    map_hstn.save(filename)
    return map_hstn

def add_hotspots(map_hstn, hotspots, coldspots):
    """
    Given a map and locations of hotspots and coldspots, will add the important spots to the
    map.

    (Inputs):
    - map_hstn: a map of the predictions
    - hotspots: a dataframe of hotspots
    - coldspots: a dataframe of coldspots

    (Outputs):
    - map_hstn: a map with the spots added
    """

    #Add markers for the hotspots (in red)
    for idx, row in hotspots.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.Icon(color='red', icon='fire', prefix='glyphicon'),
            radius=8,
            color='red',
            popup=f"{row['sensor_name']}<br>Hotspot ({round(row['local_G'], 3)})"
        ).add_to(map_hstn)

    #Add markers for the coldspots (in blue)
    for idx, row in coldspots.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.Icon(color='blue', icon='snowflake', prefix='fa'),
            radius=8,
            color='blue',
            popup=f"{row['sensor_name']}<br>Coldspot ({round(row['local_G'], 3)})"
        ).add_to(map_hstn)

    filename = f'./houston_kriging_map.html'
    map_hstn.save(filename)
    return map_hstn