import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from libpysal.weights import DistanceBand
from esda import G as getis_ord_G
from esda.getisord import G_Local
import scipy
scipy.inf=np.inf

def get_generalG(pm25_df, threshold=1500, target='pm2.5'):
    """
    Gets the Getis-Ord General G statistic for a day's PM2.5 values, which measures whether 
    there is a lopsided spatial distribution of PM2.5
    
    (Inputs):
    - pm25_df: dataframe with PM2.5 values, accompanied by longitude and latitude
    - threshold: how far two sensors can be considered as "neighbors". Given in kilometers (defaults to 1500 m)
    - target: name of target variable (defaults to pm2.5)
    
    (Outputs):
    - global_G: the global G statistic (values greater than 0 indicate similar values are spatially concentrated)
    - p_sim: p-value resulting from 999 simulations (if <0.05, indicates concentration is not random)
    """
    
    #Drop rows with missing target values
    pm25_df = pm25_df.dropna(subset=[target])

    #Create a weight matrix using the threshold
    coords = pm25_df[['longitude', 'latitude']].values
    weights = DistanceBand(coords, threshold=threshold)

    y = pm25_df[target].values

    # Apply the Getis-Ord general G statistic with 999 simulations
    g_result = getis_ord_G(y, weights, permutations=999)
    
    global_G = g_result.G
    p_sim = g_result.p_sim
    
    return global_G, p_sim

def get_localG(pm25_df, threshold=1.5, target='pm2.5'):
    """
    Returns a dataframe of each PM2.5 sensor's results after applying a Getis-Ord Local G* statistic test.
    
    (Inputs):
    - pm25_df: dataframe with PM2.5 values, accompanied by longitude and latitude
    - threshold: how far two sensors can be considered as "neighbors". Given in kilometers (defaults to 1500 m)
    - target: name of target variable (defaults to pm2.5)
    
    (Outputs):
    - local_G: the global G statistic (values greater than 0 indicate similar values are spatially concentrated)
    - p_sim: p-value resulting from 999 simulations (if <0.05, indicates concentration is not random)
    """
    
    #Drop rows with missing target values
    pm25_df = pm25_df.dropna(subset=[target])

    #Create a weight matrix using the threshold
    coords = pm25_df[['longitude', 'latitude']].values
    weights = DistanceBand(coords, threshold=threshold)
    
    y = pm25_df[target].values

    #Apply Getis-Ord Local G* test
    localG_star = G_Local(y, weights, transform='B', star=True)
    
    local_G_arr = localG_star.Zs
    p_sim_arr = localG_star.p_sim
    
    return local_G_arr, p_sim_arr

def get_hotspot_dates(pm25_df, dates, threshold=1.5, sensor_id_str='site_id', target='pm2.5'):
    """
    Gets hotspots for each given day (i.e. test positive for Getis-Ord local G* test).
    
    (Inputs):
    - pm25_df: dataframe with PM2.5 values, accompanied by longitude and latitude
    - dates: dates that tested positive for the General G test
    - threshold: how far two sensors can be considered as "neighbors". Given in kilometers (defaults to 1500 m)
    - sensor_id_str: name of sensor index variable (defaults to site_id)
    - target: name of target variable (defaults to pm2.5)
    
    (Outputs):
    - hotspot_df: dataframe with sensors identified as hotspots under certain dates
    """

    hotspot_df = pd.DataFrame()

    for date in dates:

        date_str = date.strftime('%Y-%m-%d')
        print(f'------------------------------------\nRunning simulations for {date_str}')

        #Focus on the current day's PM2.5 values
        focus = pm25_df[pm25_df['Date'] == date_str]
        
        #Run local G
        local_G, psim = get_localG(focus, threshold)
        
        #Add results to the 'focus' dataframe
        focus.loc[:, 'local_G'] = local_G
        focus.loc[:, 'p_val'] = psim
        
        #Filter for hotspots
        hotspots = focus[focus['p_val'] < 0.001]
        
        #Selected relevant columns:
        hotspots = hotspots[[sensor_id_str, 'Date', 'longitude', 'latitude', 'p_val', 'local_G', target]]
        
        #Append to results dataframe
        hotspot_df = pd.concat([hotspot_df, hotspots])
        
    hotspot_df.reset_index(drop=True, inplace=True)

    return hotspot_df