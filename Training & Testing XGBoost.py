# Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# ----------------------------------
# Load Dataset
# ----------------------------------

df = pd.read_csv("Air_Quality_Feature_Engineered.csv")

# ----------------------------------
# Data Preparation
# ----------------------------------

# Remove Date column
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])

# Convert categorical columns
if 'Season' in df.columns:
    df = pd.get_dummies(df, columns=['Season'], drop_first=True)

if 'City' in df.columns:
    df = pd.get_dummies(df, columns=['City'], drop_first=True)

# ----------------------------------
# Features and Target
# ----------------------------------

X = df.drop('AQI', axis=1)
y = df['AQI']


# ----------------------------------
# Train-Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Records:", X_train.shape[0])
print("Testing Records :", X_test.shape[0])

# ----------------------------------
# XGBoost Model
# ----------------------------------

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# ----------------------------------
# Training
# ----------------------------------

xgb_model.fit(X_train, y_train)

print("\nXGBoost Model Trained Successfully")

# ----------------------------------
# Prediction
# ----------------------------------

y_pred = xgb_model.predict(X_test)

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



# ----------------------------------
# Feature Importance
# ----------------------------------

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features")

print(importance_df.head(10))
