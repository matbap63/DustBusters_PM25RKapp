"""
This python file showcases how to utilize the exported
predictions from this utility dashboard, and some initial
data exploration to assist with future plots and interactables.

-demo_predictions.py:
    
    This is an example exported prediction dataframe. It contains PM2.5 predictions
    and interpolated features for a given day for the Houston area, in a pandas
    compatible dataframe format.
"""

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

pred_df = pd.read_csv("demo/demo_predictions.csv")

print(pred_df.columns)

"""
Here is the list of columns present in the dataset.

Most of these values are interpolated features. The lattitude 
and longitude values are presented in a grid with approximately 0.84 miles between
vertices.

Specifics of what these features represent can be accessed via our main project
repository.

['windgust', 'traffic_volume', 'humidity', 'traffic_volume_2days',
'pm2.5_lastweek', 'traffic_volume_1days', 'winddir', 'moonphase',
'precip', 'pm2.5_yesterday', 'cloudcover', 'longitude', 'latitude',
'latitude_norm', 'land_use_Barren Land (Rock/Sand/Clay)',
'land_use_Cultivated Crops', 'land_use_Deciduous Forest',
'land_use_Developed High Intensity',
'land_use_Developed, Low Intensity',
'land_use_Developed, Medium Intensity',
'land_use_Developed, open space',
'land_use_Emergent Herbaceous Wetlands', 'land_use_Evergreen Forest',
'land_use_Grassland/Herbaceous', 'land_use_Mixed Forest',
'land_use_Open Water', 'land_use_Pasture/Hay', 'land_use_Shrub/Scrub',
'land_use_Unknown', 'land_use_Woody Wetlands', 'predictions']

"""

pivoted = pred_df.pivot_table(index='latitude', columns='longitude', values='predictions')

plt.figure(figsize=(10, 8))
sns.heatmap(pivoted, cmap="viridis")  # or "coolwarm", "plasma", etc.
plt.title("Prediction Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

pivoted = pred_df.pivot_table(index='latitude', columns='longitude', values='traffic_volume')

plt.figure(figsize=(10, 8))
sns.heatmap(pivoted, cmap="viridis")  # or "coolwarm", "plasma", etc.
plt.title("Traffic Volume Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

"""
These are some visualization examples
--- CREATE FUNCTIONALITY FOR TIMESERIES PREDICTION ---

This would entail exporting a csv from the dashboard that has multiple days
to create an interactive map
"""