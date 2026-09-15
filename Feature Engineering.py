import pandas as pd

df = pd.read_csv("Air_Quality_Cleaned.csv")

df['Date'] = pd.to_datetime(df['Date'])


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "PostMonsoon"

df['Season'] = df['Month'].apply(get_season)

df.to_csv("Air_Quality_Feature_Engineered.csv", index=False)

print("Feature Engineering Completed Successfully")
print(df.head())
