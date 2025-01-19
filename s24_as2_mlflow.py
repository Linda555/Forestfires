# -*- coding: utf-8 -*-
"""S24-AS2-mlflow (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-yDy_de46dXcx4vx612UegKAcO0Q5zdN

## 1. Libraries
"""

#!pip install pyngrok

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
import keras
from keras import models, layers


import mlflow
import subprocess
from pyngrok import ngrok, conf
import getpass
import mlflow.keras

"""## 2. Data cleaning and preprocessing"""

#Read dataset
df_raw = pd.read_csv("forestfires.csv")
df = df_raw.copy()

df.head()

df.info()

"""The data have 517 observations. I see no missing value. Target area is numeric. There are categorical features (month and day) and numerical (X,Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain) data."""

# Confirming data have no missing values.
print("Antal saknade värden per kolumn:")
print(df.isnull().sum())

# There are 4 duplicates
df.duplicated().sum()

# remove duplicate
df.drop_duplicates(inplace=True)
df.duplicated().sum()

# Summary statistics
df.describe()

"""Exept from rain all the features and the target area have mean differens from zero and std different from one.So the data seem to not follow a normal distribution."""

# See numeric columns
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

# Show outliers for numeric columns (I found outliers, rings.
for col in numeric_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot för {col} (Outliers)")
    plt.show()

"""Except from X there all the numeric features and the target have different levels of extreme outliers. I will normalize them"""

# Explore target
plt.figure(figsize=(10, 6))
plt.hist(df['area'], bins=50, edgecolor='k')
plt.title("Distribution of 'area'")
plt.xlabel("Area")
plt.ylabel("Frequency")
plt.yscale('log')
plt.show()

"""Target is extremely skewed. I will apply log-transformation."""

# Correlation between features and target.

numeric_features_df = df[['temp', 'DC', 'FFMC', 'DMC', 'ISI', 'wind', 'rain', 'RH','area']]
# Correlation matrix for numerical features
correlation_matrix = numeric_features_df.corr()

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Before Log Transformations)")
plt.show()

"""temp (0.10) and RH (-0.08) have the strongest correlation with area. I will select them as numerical features. for modeling."""

# data transformation

# Encode categorical features
label_encoders = {}
categorical_columns = ['month', 'day']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Scale numerical columns
scaler = MinMaxScaler()
numerical_columns = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

df.head()

# Feature selection

# Separate features and target
X = df.drop(columns=['area'])
y = df['area']

# 1. Tree-Based Feature Importance (Random Forest)
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X, y)
rf_importances = rf_model.feature_importances_

# 2. Mutual Information
mi_importances = mutual_info_regression(X, y, random_state=42)

# Combine results into a dataframe for comparison
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Random_Forest_Importance': rf_importances,
    'Mutual_Information': mi_importances
}).sort_values(by='Random_Forest_Importance', ascending=False)

# Sort by importance for better visualization
feature_importances_sorted = feature_importances.sort_values(by='Random_Forest_Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_sorted['Feature'], feature_importances_sorted['Random_Forest_Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.show()


# Display the feature importance results
feature_importances

"""The above feature importance supports our choice of working with temp and RH as features."""

# log-Transform target for modeling
df['area_log'] = np.log(df['area'] + 1)

df.head()

# final modeling Dataframe
model_df = df[["temp", "RH", "area_log"]]

model_df.head()

"""## 3. Setup MLflow"""

# Define the MLflow tracking URI with SQLite
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# Start the MLflow server using subprocess
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI, "--port", "5000"])

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set up ngrok for exposing the MLflow UI
print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass()

# Expose the MLflow UI on port 5000
port = 5000
public_url = ngrok.connect(port).public_url
print(f' * ngrok tunnel "{public_url}" -> "http://127.0.0.1:{port}"')

"""## 4. Neural network regressor modeling"""

# Split features and target
X = model_df[["temp","RH"]]
y = model_df["area_log"]

# Split data into training,validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

# Define parameters
neurons_numbers = [32, 12]
title = 'Experiment 1'

params = {
    'title': title,
    'Features': 'temp, RH',
    'optimizer': 'adam',
    'batch_size': 16,
    'epochs': 50
}

for i, neuron in enumerate(neurons_numbers):
  params[f'hidden_layer_{i + 1}'] = neuron

params

#Define model

def nn_model(neurons_numbers):
  hidden_layers = [layers.Dense(n, activation='relu') for n in neurons_numbers]
  model = models.Sequential([
      layers.Input(shape=(2,)),
      *hidden_layers,
      layers.Dense(1, activation='linear')
  ])
  return model

model.summary()



def train_model(neurons_numbers, params):
    model = nn_model(neurons_numbers)

    # Complicate model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])

    # Fitting training data to model
    epochs_hist = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                            validation_split=0.2)

train_model(neurons_numbers,params)

"""#### Model evaluation"""

def print_model_loss():
    plt.plot(epochs_hist.history['loss'])
    plt.plot(epochs_hist.history['val_loss'])
    plt.title('Model Loss Progress During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Training and Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

print_model_loss()

# Metrics calculation function
def calculate_metrics(x, y, model= model):

    rmse = model.evaluate(x, y, verbose=0)

    # Predictions
    predictions = model.predict(x)

    # Revert predictions and target back to original scale
    y_pred_original = np.exp(predictions) - 1
    y_actual_original = np.exp(y) - 1

    # Calculate metrics on original scale
    mse_original = mean_squared_error(y_actual_original, y_pred_original)
    mae_original = mean_absolute_error(y_actual_original, y_pred_original)
    rmse_original = np.sqrt(mse_original)
    r2_original = r2_score(y_actual_original, y_pred_original)

    # Adjusted R²
    n = len(y)
    p = x.shape[1]  # Number of predictors (features)
    adjusted_r2_original = 1 - (1 - r2_original) * (n - 1) / (n - p - 1)

    return {
      'RMSE': rmse_original,
      'R² Score': r2_original,
      'Adjusted R² Score': adjusted_r2_original
      }

    # Print Metrics Function
def print_metrics(metrics, epochs_hist, title):

    metrics = {
    'RMSE': float(f'{metrics['RMSE']:.4f}'),
    'R² Score': float(f'{metrics['R² Score']:.4f}'),
    'Adjusted R² Score': float(f'{metrics['Adjusted R² Score']:.4f}'),
    'final_training_loss': float(f"{epochs_hist.history['loss'][-1]:.4f}"),
    'final_validation_loss': float(f"{epochs_hist.history['val_loss'][-1]:.4f}")
    }
    print(f'{title}:')
    print('Metrics:', metrics)

# Log Experiment to MLflow
def log_to_mlflow(params, metrics, title, x = X_train):

    mlflow.set_experiment(title)
    with mlflow.start_run():

    # Log parameters
        mlflow.log_params(params)
    # Log results
        mlflow.log_metrics(metrics)
    # Log Model with Input Example
        input_example = np.array(x[:1])
        mlflow.sklearn.log_model(model, 'Neural Network Regressor', input_example=input_example)
        print("Run, logged to MLFlow successfully!")

# Evaluate model on validation set
metrics = calculate_metrics(X_val, y_val)
metrics

#
print_metrics(metrics, epochs_hist, "metrics of experiment 1")

log_to_mlflow(params, metrics, "experiment 1, validation set")

"""R² Score (-0.1405) and Adjusted R² Score (-0.1714) are negative, this suggests that the model dosen`t performe well."""

# Experiment 2
# Define parameters
neurons_numbers = [32, 12]
title = 'Experiment 2'

params = {
    'title': title,
    'Features': 'temp, RH',
    'optimizer': 'adam',
    'batch_size': 16,
    'epochs': 100
}

for i, neuron in enumerate(neurons_numbers):
  params[f'hidden_layer_{i + 1}'] = neuron

params

train_model(neurons_numbers, params)

metrics = calculate_metrics(X_val, y_val)

print_metrics(metrics, epochs_hist, "metrics of experiment 2")

"""Increasing Epoch doesn`t improve model performance"""

log_to_mlflow(params, metrics, "experiment 2, validation set")

# Experiment 3
# Define parameters
neurons_numbers = [64, 32, 12]
title = 'Experiment 3'

params = {
    'title': title,
    'Features': 'temp, RH',
    'optimizer': 'adam',
    'batch_size': 50,
    'epochs': 100
}

for i, neuron in enumerate(neurons_numbers):
  params[f'hidden_layer_{i + 1}'] = neuron

params

train_model(neurons_numbers, params)

metrics = calculate_metrics(X_val, y_val)

print_metrics(metrics, epochs_hist, "metrics of experiment 3")

log_to_mlflow(params, metrics, "experiment 3, validation set")

"""Increasing batchsiza and number of neurons didn´t improve the models performance."""

# Evaluation on testset

metrics = calculate_metrics(X_test, y_test)

print_metrics(metrics, epochs_hist, "metrics of experiment 4 on test set")

log_to_mlflow(params, metrics, "experiment 4, test set")

"""The model struggles to generalize on data. The RMSE shows that extreme outliers affekts the model."""

#save model for deployment
model = nn_model(neurons_numbers)
model.save('nns_regressor.keras')

