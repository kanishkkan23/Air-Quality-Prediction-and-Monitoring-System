import pandas as pd
import numpy as np

df = pd.read_csv("Air Quality Dataset.csv")

#Convert Date Column to Datetime

df['Date'] = pd.to_datetime(df['Date'])

#Check Missing Values

print("Missing Values Before Cleaning:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

#Remove Duplicate Records

print("\nRows Before Removing Duplicates:", len(df))

df = df.drop_duplicates(subset=['City', 'Date'], keep='first')

print("Rows After Removing Duplicates:", len(df))


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

df.to_csv("Air_Quality_Cleaned.csv", index=False)

print("\nPreprocessing Completed Successfully")
print("Cleaned Dataset Saved as Air_Quality_Cleaned.csv")

print(df[['AQI','CO','PM2.5','PM10','NO','NO2']].corr())
