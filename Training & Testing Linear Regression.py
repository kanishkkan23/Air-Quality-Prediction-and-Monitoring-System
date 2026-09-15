# Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score )
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ----------------------------------
# Load Dataset
# ----------------------------------

df = pd.read_csv("Air_Quality_Feature_Engineered.csv")

# ----------------------------------
# Remove Non-Numeric Columns
# ----------------------------------

# Date is not directly useful for Random Forest
df = df.drop(columns=['Date'])

# Convert Season into numerical values if present
if 'Season' in df.columns:
    df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Convert City into numerical values if present
if 'City' in df.columns:
    df = pd.get_dummies(df, columns=['City'], drop_first=True)

# ----------------------------------
# Define Features and Target
# ----------------------------------

X = df.drop('AQI', axis=1)   
y = df['AQI']                

print("Min AQI:", y.min())
print("Max AQI:", y.max())
print("Mean AQI:", y.mean())

# ----------------------------------
# Train-Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training Records:", X_train.shape[0])
print("Testing Records :", X_test.shape[0])

# ----------------------------------
# Model Training
# ----------------------------------

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully")

joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)

# ----------------------------------
# Model Testing
# ----------------------------------

y_pred = lr_model.predict(X_test)

# ----------------------------------
# Evaluation Metrics
# ----------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))


joblib.dump(
    lr_model,
    "LinearRegression_AQI_Model.pkl"
)

print("Model Saved Successfully")

