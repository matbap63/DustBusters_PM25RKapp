import streamlit as st
from datetime import datetime
from utils.predictions_utils import *
from utils.interpolation_vis_app import *
from utils.hotspot_analysis import *
from utils.regression_kriging import *
import streamlit.components.v1 as components
import pandas as pd
import yaml

core_data = pd.read_csv("./PM25_sensor_data")
st.sidebar.title("PM2.5 Dashboard")

model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"])
date = st.sidebar.date_input(
    "Select Date",
    value=datetime(2024, 3, 15), 
    min_value=datetime(2024, 1, 1),
    max_value=datetime(2024, 12, 31)
)
location_input = st.sidebar.text_input("Enter ZIP Code or 'lat,lon'")

###############################################
#
# Convert location input to coordinates
#
###############################################

try:
    lat, lon = get_coordinates(location_input)
    st.sidebar.success(f"Using coordinates: ({lat:.4f}, {lon:.4f})")
except Exception as e:
    st.sidebar.error(f"Location error: {e}")
    st.stop()

###############################################
#
# Select features
#
###############################################
#Normalized train and test sets
train_set_norm=pd.read_csv("./train_norm.csv")
test_set_norm=pd.read_csv("./test_norm.csv")
landuse_raster=pd.read_csv("./landuse_grid_4-10.csv")
auxiliary_master_norm=pd.read_csv("./master_auxiliary_4-13.csv")
income_raster=pd.read_csv("./grid_income.csv")

OK_features = st.multiselect(
    "Select features to include (engineered with Ordinary Kriging):",
    ['humidity', 'temp', 'sealevelpressure', 'windgust', 'precip', 'winddir', 'moonphase',
                'cloudcover', 'pm2.5_yesterday', 'pm2.5_lastweek'],
    default=['humidity', 'windgust', 'precip','winddir', 'moonphase','cloudcover','pm2.5_yesterday','pm2.5_lastweek']
)

IDW_features = st.multiselect(
    "Select features to include (engineered with Inverse Distance Weighting):",
    ['TPY_wind', 'traffic_volume', 'traffic_volume_1days', 'traffic_volume_2days', 'traffic_volume_3days'],
    default=['traffic_volume', 'traffic_volume_1days', 'traffic_volume_2days']
)

raster_features = st.multiselect(
    "Select features to include (raster-level data):",
    ['land_use_Developed, Low Intensity', 'land_use_Developed, open space',
    'land_use_Pasture/Hay', 'land_use_Developed High Intensity', 'land_use_Woody Wetlands',
    'land_use_Emergent Herbaceous Wetlands', 'land_use_Cultivated Crops', 'median_income'],
    default=['land_use_Pasture/Hay', 'land_use_Developed High Intensity']
)

###############################################
#
# Assess validity of latitude and longitude
#
###############################################

in_grid = check_within_grid(lat, lon)

prediction_df = None

if not in_grid:
    st.error("Your location is outside the interpolation grid. Please select a different point.")
    st.stop()
else:
    
    selected_features = OK_features + IDW_features + raster_features + ['latitude_norm']
    raster_list = []

    if 'median_income' in raster_features:
        raster_list.append(income_raster)
    if len(raster_features) > 0:
        raster_list.append(landuse_raster)

    #Define grid of interpolation
    gridx, gridy = create_grid(core_data, buffer_long=0.35, buffer_lat=0.1, num_points=100)

    #Train models
    lr = train_elasticnet(train_set_norm, selected_features, 'pm2.5', 
                        lr1_ratio=1, alpha=0.0001)
    rf = train_rf(train_set_norm, selected_features, 'pm2.5', 
                        n_estimators=100, max_depth=20, max_features='sqrt', 
                        min_samples_leaf=1, min_samples_split=2)

    if model_choice=="Linear Regression":
        
        model_type=lr

    else: 

        model_type=rf

    prediction_df = get_RK_interpolation_grid(date=date, 
                                    model=lr, 
                                    feature_list_OK=OK_features,
                                    feature_list_IDW=IDW_features, 
                                    feature_list_raster=raster_features, 
                                    train_set=train_set_norm, 
                                    interpolation_set=test_set_norm, 
                                    auxiliary_dataset=auxiliary_master_norm, 
                                    raster_list=raster_list, 
                                    target='pm2.5', 
                                    gridx=gridx, gridy=gridy,
                                    use_latitude=True, get_pickle=False)
    found, prediction = get_pred(lat, lon, prediction_df)

###############################################
#
# Find prediction
#
###############################################

your_prediction = round(prediction['predictions'], 5)

if your_prediction < 5.0:
    color = 'green'
elif 5.0 <= your_prediction < 15.0:
    color = 'yellow'
elif 15.0 <= your_prediction < 35.0:
    color = 'orange'
else:
    color = 'red'


st.download_button(
    label="Export Predictions as CSV",
    data=prediction_df.to_csv(index=False).encode('utf-8'),
    file_name=datetime.now().strftime("data_%Y-%m-%d_%H-%M-%S.csv"),
    mime='text/csv',
)

st.header('PM2.5 at your location')
st.markdown(
    f"<h3>Predicted PM2.5 (µg/m³):<br><br><span style='color:{color}'>{your_prediction}</span></h3>",
    unsafe_allow_html=True
)


st.subheader('What does this mean?')

st.write('PM2.5 refers to particulate matter under 2.5 micrometers in diameter (PM2.5). As a pollutant, these particles are so small ' \
'that they can be inhaled into the bloodstream through the lungs, presenting potential health consequences such as asthma, cardiovascular disease, ' \
'and even lung cancer.')

if your_prediction < 5.0:

    st.markdown('<span style="color:green"><strong>This concentration is considered to be generally safe for both long-term and short-term exposure.</span>', unsafe_allow_html=True)

elif 5.0 <= your_prediction < 9.0:

    st.markdown('<span style="color:yellow"><strong>The World Health organization notes that this concentration (>5.0 µg/m³) may be hazardous '
            'if present annually. However, it is not considered hazardous by the Texas Commission on Environmental Quality, who, according to EPA standards, recently '
            'changed their safety thresholds to 9.0 µg/m³.</span>', unsafe_allow_html=True)
    
elif 9.0 <= your_prediction < 15.0:

    st.markdown('<span style="color:yellow"><strong>The Texas Commission on Environmental Quality, according to EPA standards, considers this concentration '
            '(>9.0 µg/m³) to be hazardous at annual concentrations.</span>', unsafe_allow_html=True)

elif 15.0 <= your_prediction < 35.0:

    st.markdown('<span style="color:orange"><strong>This concentration (>15.0 µg/m³) is considered to be concerning at a 24-hour concentration according to the World Health Organization. '
            'However, it is not considered so by the Texas Commission on Environmental Quality, who recently changed their standards to >35 µg/m³.</span>', unsafe_allow_html=True)
    
elif 35.0 <= your_prediction < 50.0:

    st.markdown('<span style="color:orange"><strong>This concentration (>35.0 µg/m³) is considered to be hazardous by the Texas Commission on Environmental Quality.</span>', unsafe_allow_html=True)

elif 50.0 <= your_prediction:

   st.markdown('<span style="color:red"><strong>This concentration (>50.0 µg/m³) is considered to be hazardous, and prolonged exposure '
            'may lead to serious health issues and premature mortality</span>', unsafe_allow_html=True)

###############################################
#
# Visualize grid
#
###############################################

map_obj = plot_interpolation(prediction_df, date, core_data, lon, lat)

with open("houston_kriging_map.html", 'r', encoding='utf-8') as HtmlFile:
        source_code = HtmlFile.read()
        components.html(source_code, height=600, width=1000)

use_alternate_viz = st.checkbox("Visualize relative PM2.5 levels")

if use_alternate_viz:
     
     map_obj = plot_interpolation_relative(prediction_df, date, core_data, lon, lat)
     with open("houston_kriging_map.html", 'r', encoding='utf-8') as HtmlFile:
        source_code = HtmlFile.read()
        components.html(source_code, height=600, width=1000)

###############################################
#
# Find hotspots
#
###############################################

hotspots_info = st.checkbox("Want to know more about potential hotspots?")

if hotspots_info:

    core_data['datetime'] = pd.to_datetime(core_data['datetime'], errors='coerce')
    core_data_sub = core_data[core_data['datetime'].dt.date == date]

    localG, p_vals = get_localG(core_data_sub, threshold=0.01)

    core_data_sub.loc[:, 'local_G'] = localG
    core_data_sub.loc[:, 'p_val'] = p_vals

    spots = core_data_sub[core_data_sub['p_val'] < 0.05]

    hotspots = spots[spots['local_G'] > 0]

    if hotspots.empty:

        st.write("There are no detectable hotspots on this map.")

    else:

        st.write(f"There are {hotspots.shape[0]} potential hotspots on this map.")
        map_obj_HC = add_hotspots(map_obj, hotspots)
        with open("houston_kriging_map.html", 'r', encoding='utf-8') as HtmlFile:
            source_code = HtmlFile.read()
            components.html(source_code, height=600, width=1000)

        closest, distance = find_closest_hotspot(hotspots, lat, lon)

        if closest is not None:
            closest_name = closest['sensor_name']
            st.write(f"You are {distance:.2f} miles away from this hotspot: **{closest_name}**")

            hotspot_groups = load_hotspot_info()

            group_name = None
            site_name = closest['sensor_name']
            for group, data in hotspot_groups.items():
                if site_name in data["sensor_name"]:
                    group_name = group
                    break

            if group_name:
                group_data = hotspot_groups[group_name]
                st.markdown(f"### About this area: {group_name}")
                st.markdown(group_data["description"])
            else:
                st.info("No additional information available for this hotspot.")