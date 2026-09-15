
import pandas as pd
from sklearn.model_selection import train_test_split,  RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score )
import matplotlib.pyplot as plt
import numpy as np
import joblib


df = pd.read_csv("Air_Quality_Feature_Engineered.csv")


# Remove Non-Numeric Columns
df = df.drop(columns=['Date'])

# Convert Season into numerical values if present
if 'Season' in df.columns:
    df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Convert City into numerical values if present
if 'City' in df.columns:
    df = pd.get_dummies(df, columns=['City'], drop_first=True)


X = df.drop('AQI', axis=1)   
y = df['AQI']                

print("Min AQI:", y.min())
print("Max AQI:", y.max())
print("Mean AQI:", y.mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training Records:", X_train.shape[0])
print("Testing Records :", X_test.shape[0])


rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# ----------------------------------
# Hyperparameter Search
# ----------------------------------

param_grid = {
    "n_estimators": [200, 300, 500, 700],
    "max_depth": [None, 10, 15, 20, 25, 30],
    "min_samples_split": [2, 4, 5, 8, 10],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": [1.0, "sqrt", "log2", 0.7, 0.8],
    "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,
    scoring="r2",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

# ----------------------------------
# Best Model
# ----------------------------------

best_model = random_search.best_estimator_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross Validation R²:")
print(round(random_search.best_score_, 4))

# ----------------------------------
# Prediction
# ----------------------------------

y_pred = best_model.predict(X_test)

# ----------------------------------
# Evaluation
# ----------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nFinal Model Performance")

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully")

joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)


#y_pred = rf_model.predict(X_test)


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
    rf_model,
    "RandomForest_AQI_Model.pkl"
)

print("Model Saved Successfully")


