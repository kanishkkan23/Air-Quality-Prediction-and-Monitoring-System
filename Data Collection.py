import pandas as pd

df = pd.read_csv("Air Quality Dataset.csv")

# Display first 5 rows
print("First 5 Records:")
print(df.head())

# Dataset Shape
print("\nShape of the Dataset:")
print(df.shape)

# Column Names
print("\nColumns:")
print(df.columns)

# Dataset Information
print("\nInformation of the Dataset:")
print(df.info())

#Dataset Description
print('\nDescription of the Dataset:')
print(df.describe())
