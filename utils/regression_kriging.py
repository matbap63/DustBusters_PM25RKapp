import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging
from sklearn.preprocessing import StandardScaler

def normalize_train(training_data, features):
    """
    Normalizes the data using z-normalization. 
    
    (Inputs):
    - training_data: a dataframe with training data
    - features: a list of features to normalize (typically numeric)
    
    (Outputs):
    - training_data_norm: the training data with z-normalization
    - scaler: a StandardScaler() object trained on the parameters of the training data
    """
    
    znorm = StandardScaler()
    
    #Obtain normalization parameters and normalize
    training_data_norm = training_data[features].copy()
    training_data_norm = znorm.fit_transform(training_data_norm)

    #Convert normalized data to a dataframe
    non_numeric_columns = list(set(training_data.columns)-set(features))
    training_data_norm = pd.DataFrame(training_data_norm, columns=features, index=training_data.index)
    training_data_norm = pd.concat([training_data[non_numeric_columns], training_data_norm], axis=1)
    
    return training_data_norm, znorm
    
def normalize_other(scaler, data, features):
    """
    Normalizes the data using z-normalization. 
    
    (Inputs):
    - scaler: a StandardScaler() object trained on the parameters of the training data
    - data: a dataframe with data to be normalized
    - features: a list of features to normalize (typically numeric)
    
    (Outputs):
    - data_norm: the data with z-normalization
    """
    
    #Obtain normalization parameters and normalize
    data_norm = data[features].copy()
    data_norm = scaler.transform(data_norm)

    #Convert normalized data to a dataframe
    non_numeric_columns = list(set(data.columns)-set(features))
    data_norm = pd.DataFrame(data_norm, columns=features, index=data.index)
    data_norm = pd.concat([data[non_numeric_columns], data_norm], axis=1)
    
    return data_norm   

def create_grid(df, buffer_long=0.35, buffer_lat=0.1, num_points=100):
    """
    Creates a grid on which to interpolate PM2.5 values (or other feature) based on inputted dataframe longitude, latitude values.

    (Inputs):
    - df: dataframe with (longitude, latitude) points associated with PM2.5 values (or other feature)
    - buffer_long: buffer of longitude values (defaults to 0.35 arcseconds)
    - buffer_lat: buffer of latitude values (defaults to 0.1 arcseconds)
    - num_points: number of points in the grid (defaults to 100)

    (Outputs):
    - gridx, gridy: longitude and latitude on which to interpolate PM2.5
    """
    gridx = np.linspace(min(df['longitude']) - buffer_long, max(df['longitude']) + buffer_long, num_points)
    gridy = np.linspace(min(df['latitude']) - buffer_lat, max(df['latitude']) + buffer_lat, num_points)
    return gridx, gridy

def train_OK_model(train_set, feature_col, var_method='exponential', show_plot=True, print_progress=False):
    """
    Trains an ordinary kriging model that interpolates PM2.5 concentration for a given day

    (Inputs):
    - train_set: dataframe with a given day's training set data (i.e. longitude and latitude with PM2.5 values)
    - feature_col: string indicating the name of the column to interpolate (e.g. 'pm2.5', 'temp')
    - var_method: variogram model to use ('spherical', 'gaussian', 'exponential', etc.). Defaults to "exponential"
    - show_plot: whether or not to show the variogram plot. Defaults to True
    - print_progress: if True, print statements will show progress.

    (Outputs):
    - OK_model: the trained ordinary kriging model. Returns None if there are not enough records in training set or there
    is no variation in the training set
    """
    # Check if the feature_col exists
    if feature_col not in train_set.columns:
        raise ValueError(f"Column '{feature_col}' not found in the dataframe")

    train_set = train_set.dropna(subset=[feature_col])

    #Check if there is variation in the coordinates. If not, OK fails.
    if np.all(train_set[feature_col].values == train_set[feature_col].values[0]):
        
        if print_progress:
            print(f"Skipping training — '{feature_col}' values are all equal to {train_set[feature_col].values[0]}")
        return None

    # Check if there are enough records for interpolation
    if len(train_set) >= 5:

        train_lat = train_set['latitude'].values
        train_long = train_set['longitude'].values
        train_target = train_set[feature_col].values

        OK_model = OrdinaryKriging(train_long, train_lat, train_target,
                                variogram_model=var_method,
                                coordinates_type='geographic',
                                verbose=False,
                                enable_plotting=show_plot)
        return OK_model

    else:

        raise ValueError("Training set consists of less than 5 records.")
def distance_matrix(x0, y0, x1, y1):
    """
    Calculate Euclidean distance matrix between observed and points to be interpolated.
    
    (Inputs):
    x0, y0: arrays of observed x, y coordinates
    x1, y1: arrays of grid x, y coordinates for interpolation
    
    (Outputs):
    Distance matrix (rows observation, cols interpolation)
    """
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    
    #Calculate (x, y) differences for Euclidean distance
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])
    return np.hypot(d0, d1)

def calculate_IDW_weights(distance, power=2):
    """
    Calculates weights for IDW.
    
    (Inputs):
    distance: distance matrix between observed and interpolation points
    power: exponent for weighting (default = 2)
    
    (Outputs):
    Weights for IDW interpolation
    """
    #Avoid division by zero by assigning a small distance 
    distance[distance == 0] = 1e-10

    #Weights are calculated as 1/distance^power
    return 1.0 / (distance**power)


def IDW(x, y, z, xi, yi, power=2):
    """
    Performs Inverse Distance Weighting interpolation.
    
    (Inputs):
    x, y: arrays of observed x, y coordinates
    z: observed feature
    xi, yi: grid coordinates to interpolate on
    power: exponent to which distance is raised (determines the "weight" of distance). Default is 2
    
    Returns:
    zi: interpolated values at specified points.
    """
    distance = distance_matrix(x, y, xi, yi)

    weights = calculate_IDW_weights(distance, power)

    #Normalize by the sum of the weights: weight/total weight, and scale observed feature values by weights
    zi = np.dot(weights.T, z) / np.sum(weights, axis=0) 
    return zi

def get_IDW_interpolation_grid(x, y, z, gridx, gridy, power=2):
    """
    Performs IDW interpolation over a specified grid.

    (Inputs):
    - x, y: arrays of observed x, y coordinates
    - z: observed feature
    - gridx: x coordinates of the grid
    - gridy: y coordinates of the grid
    - power: exponent for weighting (default = 2)
    
    (Outputs):
    - z_interp: 2D array of interpolated values on the grid
    """
    
    #Initialize the empty grid to store IDW-interpolated values
    z_interp = np.zeros((len(gridy), len(gridx)))

    #Loop through each point in the grid and calculate IDW value
    for i, yi in enumerate(gridy):
        for j, xi in enumerate(gridx):
            z_interp[i, j] = IDW(x, y, z, np.array([xi]), np.array([yi]), power)

    return z_interp

def get_OK_interpolation_grid(OK_model, gridx, gridy):
    """
    Perform interpolation using the trained OK model on a specified grid.

    (Inputs):
    - OK_model: trained ordinary kriging model
    - gridx: x coordinates of the grid
    - gridy: y coordinates of the grid

    (Outputs):
    - z_interp: interpolated values on the grid
    - ss: sum of squared residuals
    """
    z_interp, ss = OK_model.execute('grid', gridx, gridy)
    return z_interp, ss

def train_elasticnet(train_set, selected_features, target, lr1_ratio=1, alpha=0.0001):
   """
   Train an elastic net model to regularize features

   (Inputs):
   train_set: the data on which to train the model. Should not contain NA.
   selected_features: features selected
   target: name of the target column
   lr1_ratio: the ratio of L1:L2 (Lasso vs. Ridge). Defaults to 1
   alpha: regularization strength. Defaults to 0.0001

   (Outputs):
   elasticnet: the trained mode
   """
   X = train_set[selected_features].values

   #Check if X is a 1D array and reshape if necessary
   if X.ndim == 1:
       X = X.reshape(-1, 1)

   y = train_set[target].values
   linreg = ElasticNet(alpha = alpha, l1_ratio=lr1_ratio, max_iter=100000)
   linreg.fit(X, y)
   return linreg

def train_rf(train_set, selected_features, target, n_estimators=100, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2):
    """
    Train a random forest regression model. We use defaults here, but in the future, we will set the parameters
    to the best values using hyperparameter tuning.

    (Inputs):
    train_set: the data on which to train the model
    selected_features: features selected
    target: name of the target column

    (Outputs):
    linreg: the trained model
    """
    X = train_set[selected_features].values

    #Check if X is a 1D array and reshape if necessary
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = train_set[target].values
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    rf.fit(X, y)
    return rf

def get_residuals(model, interpolation_set, features, target):
  """
  Calculate residuals for kriging using a trained regression model

  (Inputs):
  - model: regression model that has been trained and fit on training set data
  - interpolation_set: data to run regression to obtain residuals
  - feature_list: list of features
  - target: name of the target variable

  (Outputs):
  - residuals: the residuals of the regression that will be kriged
  - trend_error: the RMSE of the model on the interpolation set (cf. training error)
  """
  X = interpolation_set[features].values
  ground_truth = interpolation_set[target]

  #Calculate the trend across the interpolation set by applying regression
  trend = model.predict(X)

  #Obtain residuals
  residuals = ground_truth - trend

  #Trend error
  trend_error = np.sqrt(mean_squared_error(ground_truth, trend))

  return residuals, float(trend_error)

def get_trend_component(model, feature_grid_df, feature_names):
  """
  Calculates the trend component of regression kriging by running regression over a
  feature grid dataframe

  (Inputs):
  - model: regression model that has been trained and fit on training set data
  - feature_grid_df: a dataframe of columns with interpolated features

  (Outputs):
  - trend_comp: the trended component (regression applied over the grid)
  """
  feature_grid_X = feature_grid_df[feature_names].values

  # Check if X is a 1D array and reshape if necessary
  if feature_grid_X.ndim == 1:
      X = X.reshape(-1, 1)

  trend_comp = model.predict(feature_grid_X)

  return trend_comp

def get_spatial_component(OK_model, gridx, gridy):
  """
  Calculates the trend component of regression kriging by running regression over a
  feature grid dataframe

  (Inputs):
  - OK_model: ordinary kriging model that has been trained on residuals
  - gridx, gridy: longitude and latitude on which to interpolate PM2.5

  (Outputs):
  - trend_comp: the trended component (regression applied over the grid)
  """
  interp = get_OK_interpolation_grid(OK_model, gridx, gridy)[0]
  SA_comp = interp.flatten()

  return SA_comp

def fill_grid(value, gridx, gridy):
    """
    If there is no variation in a feature over a grid, ordinary kriging will fail. Instead,
    fill the 'interpolated' values as a constant over the grid.

    (Inputs):
    - value: value to be filled over the grid
    - num_points: the number of points in the grid

    (Outputs):
    - interp: the interpolated grid (i.e. just a grid filled with inputted value)
    """

    M, N = len(gridx), len(gridy)
    interp = np.full((M, N), value)

    return interp

def get_feature_grids_OK(df_reference, features, gridx, gridy, print_progress=False):
    """
    Given a reference dataframe with an external feature (e.g. a dataframe containing temperature values across Houston),
    will provide a list of grids of interpolated values for features, suitable for regression kriging

    (Inputs):
    - df_reference: dataframe with external information (longitude, latitude pairs associated with a feature to be interpolated)
    - features: a list of features that will be interpolated
    - gridx, gridy: bounds of the grids obtained from create_grid()
    - print_progress: if True, print statements will show progress.

    (Outputs):
    - feature_grids_list: a list of grids with interpolated features

    """
    features_filled = 0
    features_skipped = 0
    features_success = 0

    feature_grid_list = []

    for category in features:

        if print_progress:
            print(f'Interpolating {category}....')
        df_reference_sub = df_reference.dropna(subset=['latitude', 'longitude', category])

        #Train the OK model using weather data
        OK_feature = train_OK_model(df_reference_sub, category, var_method='exponential', show_plot=False)

        #Check if OK_model failed or not
        if OK_feature is None:

          #Case 1: Constant feature values — fill with constant
          if df_reference_sub[category].nunique() == 1:

              feature_grid = fill_grid(df_reference_sub[category].values[0], gridx, gridy)
              feature_grid_list.append(feature_grid)
              features_filled += 1

          #Case 2: Insufficient data, not constant — skip
          else:
              if print_progress:
                print(f"Interpolation skipped for {category} — insufficient data or training failed.")
              features_skipped += 1
          continue

        #Case 3: Apply the model over the specified grid
        feature_grid = get_OK_interpolation_grid(OK_feature, gridx, gridy)[0]
        feature_grid_list.append(feature_grid)
        features_success += 1

    if print_progress:
        print(f'Final:\nFills: {features_filled}\nSkipped: {features_skipped}\nSuccessful: {features_success}')
    return feature_grid_list

def get_feature_grids_IDW(df_reference, features, gridx, gridy, print_progress=False):
    """
    Given a reference dataframe with an external feature (e.g. a dataframe containing temperature values across Houston),
    will provide a list of grids of interpolated values for features using IDW, suitable for regression kriging

    (Inputs):
    - df_reference: dataframe with external information (longitude, latitude pairs associated with a feature to be interpolated)
    - features: a list of features that will be interpolated
    - gridx, gridy: bounds of the grids obtained from create_grid()
    - print_progress: if True, print statements will show progress.

    (Outputs):
    - feature_grids_list: a list of grids with interpolated features

    """
    features_filled = 0
    features_skipped = 0
    features_success = 0

    feature_grid_list = []

    for category in features:

        if print_progress:
            print(f'Interpolating {category}....')
            
        df_reference_sub = df_reference.dropna(subset=['latitude', 'longitude', category])
        
        #Case 1: Constant feature values — fill with constant
        if df_reference_sub[category].nunique() == 1:

            feature_grid = fill_grid(df_reference_sub[category].values[0], gridx, gridy)
            feature_grid_list.append(feature_grid)
            features_filled += 1

        #Case 2: Apply the model over the specified grid
        ref_long = df_reference_sub['longitude']
        ref_lat = df_reference_sub['latitude']
        ref_val = df_reference_sub[category]
        feature_grid = get_IDW_interpolation_grid(ref_long, ref_lat, ref_val, gridx, gridy, power=2)
        feature_grid_list.append(feature_grid)
        features_success += 1

    if print_progress:
        print(f'Final:\nFills: {features_filled}\nSkipped: {features_skipped}\nSuccessful: {features_success}')
    return feature_grid_list

def convert_grid_to_df(feature_list, feature_grid_list, gridx, gridy):
  """
  Given a list of feature grids (A, B, ...), this function converts each feature grid into a column of a dataframe,
  with longitude, latitude pairs

  (Inputs):
    - feature_list: a list of features whose indices match those of the feature_grid_list
    - feature_grid_list: a list of grids with interpolated features
    - gridx, gridy: bounds of the grids obtained from create_grid()

  (Outputs):
  - feature_grid_df: a dataframe of columns with interpolated features
  """

  #Flatten all grids to arrays
  gridx_flat = np.tile(gridx, len(gridy))
  gridy_flat = np.repeat(gridy, len(gridx))

  feature_grid_flattened_list = []

  for feature_grid in feature_grid_list:

    feature_grid_flattened = np.array(feature_grid.data).flatten()

    feature_grid_flattened_list.append(feature_grid_flattened)

  feature_grid_arr = np.array(feature_grid_flattened_list).T

  #Create the feature grid dataframe
  feature_grid_df = pd.DataFrame(feature_grid_arr, columns=feature_list)
  feature_grid_df['longitude'] = gridx_flat
  feature_grid_df['latitude'] = gridy_flat

  return feature_grid_df

def get_RK_interpolation_grid(date, model, feature_list_OK, feature_list_IDW, feature_list_raster, train_set, interpolation_set, auxiliary_dataset, 
                                     raster_list, target, gridx, gridy, use_latitude=True, get_pickle=False):
    """
    Runs regression kriging for a single day over a specified grid.
     
    (Inputs):
    - date: a datetime object specifying the day to interpolate in the test set
    - model: regression model that has been trained and fit on training set data
    - feature_list_OK, feature_list_IDW, feature_list_raster: lists of the names of features to
    be interpolated through OK, features to be interpolated for IDW, and features with raster-level data
    - train_set, test_set: training data and testing data
    - auxiliary_dataset: dataset to be referenced when interpolating feature grids
    - raster_list: list of raster-level feature grids
    - gridx, gridy: arrays of longitude and latitude specifying grid of interpolation
    - use_latitude: use latitude as a predictor? Defaults to True
    - seed: seed for train/test split
    - get_pickle: if True, pickles the prediction results
    
    (Outputs):
    - predictions: the values over the grid of interpolation
    - a dataframe containing error metrics 
    """
    date_str = date.strftime('%Y-%m-%d')

    #Split current date data into interpolation and validation sets
    focus = interpolation_set[interpolation_set['datetime'] == date_str]
    if focus.empty:
        raise ValueError(f"No data found for date {date_str} in the test_set.")
        
    feature_list = feature_list_OK + feature_list_IDW + feature_list_raster
    
    if use_latitude:
        
        feature_list += ['latitude_norm']

    #Residuals from regression model
    residuals, trend_error = get_residuals(model, focus, feature_list, target)
    focus = focus.copy()
    focus['residuals'] = residuals

    #Krige residuals
    OK_res = train_OK_model(focus, 'residuals', var_method='exponential', show_plot=False)

    #Feature grid construction
    auxiliary_subset = auxiliary_dataset[auxiliary_dataset['datetime'] == date_str]
    feature_grid_IDW = get_feature_grids_IDW(auxiliary_subset, feature_list_IDW, gridx, gridy)
    feature_grid_OK = get_feature_grids_OK(auxiliary_subset, feature_list_OK, gridx, gridy)
    feature_grid_interp_list = list(set(feature_list) - set(['latitude_norm'] + feature_list_raster))
    feature_grid_list = feature_grid_IDW + feature_grid_OK
    feature_grid_df = convert_grid_to_df(feature_grid_interp_list,
                                                feature_grid_list, gridx, gridy)
    
    if use_latitude:
        
        #Normalize latitude
        latnormer = normalize_train(train_set, ['latitude'])[1]
        feature_grid_lat = normalize_other(latnormer, feature_grid_df, ['latitude'])
        feature_grid_lat.rename(columns={'latitude': 'latitude_norm'}, inplace=True)
        shared_cols = feature_grid_df.columns.intersection(feature_grid_lat.columns).tolist()
        feature_grid_df = pd.merge(
            feature_grid_df,
            feature_grid_lat,
            on=shared_cols,
            how='left'
        )
    
    feature_grid_df[['longitude', 'latitude']] = feature_grid_df[['longitude', 'latitude']].round(5)

    #Merge in raster layers
    if raster_list:
        for layer in raster_list:
            layer = layer.copy()
            layer[['longitude', 'latitude']] = layer[['longitude', 'latitude']].round(5)
            feature_grid_df = pd.merge(feature_grid_df, layer, on=['longitude', 'latitude'], how='left')
    
    trend = get_trend_component(model, feature_grid_df, feature_list)
    SA = get_spatial_component(OK_res, gridx, gridy)
    print(trend, SA)
    predictions = trend + SA

    feature_grid_df['predictions'] = predictions

    return feature_grid_df
