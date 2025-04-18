import pandas as pd
from scipy.spatial import cKDTree
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import datetime
import yaml
import streamlit as st
import os

def load_pickle(model_type, date):
    """
    Given a model and date, will load the corresponding pickle
    
    (Inputs):
    - model_type: the model type selected (linear regression or random forest)
    - date: the datetime selected
    """

    model_dir = "./lr_pkl" if model_type == "Linear Regression" else "./rf_pkl"
    
    date_str = date.strftime('%Y-%m-%d')
    path = os.path.join(model_dir, f"RK_predictions_{date_str}.pkl")

    #Handle no path situation
    if not os.path.exists(path):
        raise FileNotFoundError(f"No pickle found for {model_type} on {date}")
    return pd.read_pickle(path)

def get_coordinates(location_input):
    """
    Will return the coordinates of an inputted location, whether it be a (lat, long) pair
    or a zip code

    (Inputs):
    -location_input: a location (either coordinates or zip code)

    (Outputs):
    -lat, lon: a latitude, longitude pair
    """

    #A (lat, long) coordinate pair will have a comma. Use this to determine if a pair is inputted
    if ',' in location_input:

        lat, lon = map(float, location_input.split(','))

    else: 
        location_input += ", USA"
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode(location_input)
        if location:
            lat, lon = location.latitude, location.longitude
        else:
            raise ValueError("Invalid ZIP or location.")
        
    return lat, lon

def check_within_grid(lat, lon):
    """
    Checks whether the inputted latitude, longitude fall within the grid of interpolation
    
    (Inputs):
    - lat, lon: inputted latitude and longitude
    """
    #Below are the dimensions of the grid of interpolation
    min_long = -96.09033
    max_long = -94.51128290000001
    min_lat = 28.910556
    max_lat = 30.450355000000002

    if (min_lat <= lat <= max_lat) and (min_long <= lon <= max_long):
        return True
    else:
        return False
    
def get_pred(lat, lon, pm25_df):
    """
    Finds the nearest point in the grid of interpolation.

    (Inputs):
    - lat, lon: inputted latitude nand longitude
    - pm25_df: PM25 dataframe from the loaded pickled file

    (Outputs):
    - A boolean specifying whether or not the neighbor was within 0.001º
    - pred: the prediction
    """
    
    #Find the closest point in the loaded pickle file
    tree = cKDTree(pm25_df[['latitude', 'longitude']])
    distance, index = tree.query([lat, lon])

    return distance < 0.001, pm25_df.iloc[index]

@st.cache_data
def load_hotspot_info():
    with open("./hotspot_info.yaml", "r") as f:
        return yaml.safe_load(f)