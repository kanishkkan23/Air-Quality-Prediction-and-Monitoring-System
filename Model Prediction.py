import pandas as pd
import joblib
import matplotlib.pyplot as plt


model = joblib.load("RandomForest_AQI_Model.pkl")


input_data = pd.DataFrame(
    [[0.0] * len(model.feature_names_in_)],
    columns=model.feature_names_in_
)

city = input("Enter City Name: ")

date_input = input("Enter Date (YYYY-MM-DD): ")

pm25 = float(input("PM2.5: "))
pm10 = float(input("PM10: "))
no = float(input("NO: "))
no2 = float(input("NO2: "))
nox = float(input("NOx: "))
nh3 = float(input("NH3: "))
co = float(input("CO: "))
so2 = float(input("SO2: "))
o3 = float(input("O3: "))
benzene = float(input("Benzene: "))
toluene = float(input("Toluene: "))
xylene = float(input("Xylene: "))


date = pd.to_datetime(date_input)

year = date.year
month = date.month
day = date.day


if month in [12, 1, 2]:
    season = "Winter"
elif month in [3, 4, 5]:
    season = "Summer"
elif month in [6, 7, 8, 9]:
    season = "Monsoon"
else:
    season = "PostMonsoon"


feature_values = {
    'PM2.5': pm25,
    'PM10': pm10,
    'NO': no,
    'NO2': no2,
    'NOx': nox,
    'NH3': nh3,
    'CO': co,
    'SO2': so2,
    'O3': o3,
    'Benzene': benzene,
    'Toluene': toluene,
    'Xylene': xylene,
    'Year': year,
    'Month': month,
    'Day': day
}

for feature, value in feature_values.items():
    if feature in input_data.columns:
        input_data.loc[0, feature] = value


city_column = f"City_{city}"

if city_column in input_data.columns:
    input_data.loc[0, city_column] = 1
else:
    print(f"Warning: {city} not found in training data.")

#Set Season Dummy Variable


season_column = f"Season_{season}"

if season_column in input_data.columns:
    input_data.loc[0, season_column] = 1


predicted_aqi = model.predict(input_data)

aqi = predicted_aqi[0]

print("\nPredicted AQI:", round(aqi, 2))


if aqi <= 50:
    category = "Good"
elif aqi <= 100:
    category = "Satisfactory"
elif aqi <= 200:
    category = "Moderate"
elif aqi <= 300:
    category = "Poor"
elif aqi <= 400:
    category = "Very Poor"
else:
    category = "Severe"

print("AQI Category:", category)


