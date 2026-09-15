import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# -----------------------------
# Load Model and Features
# -----------------------------

model = joblib.load("RandomForest_AQI_Model.pkl")

feature_columns = joblib.load(
    "feature_columns.pkl"
)

import base64

def add_bg():

    with open("images/background2.jpg", "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
    <style>

    .stApp {{

        background-image: url("data:image/jpg;base64,{encoded}");

        background-size: cover;

        background-position: center;

        background-repeat: no-repeat;

        background-attachment: fixed;

    }}

    </style>
    """, unsafe_allow_html=True)

add_bg() 

# -----------------------------
# Page Title
# -----------------------------

st.title("🌍 AQI Prediction System")

st.write(
    "Enter Air Pollutant Values and Predict AQI"
)

# -----------------------------
# User Inputs
# -----------------------------

city = st.text_input("City")

date = st.date_input("Date")

pm25 = st.number_input("PM2.5", min_value=0.0)

pm10 = st.number_input("PM10", min_value=0.0)

no = st.number_input("NO", min_value=0.0)

no2 = st.number_input("NO2", min_value=0.0)

nox = st.number_input("NOx", min_value=0.0)

nh3 = st.number_input("NH3", min_value=0.0)

co = st.number_input("CO", min_value=0.0)

so2 = st.number_input("SO2", min_value=0.0)

o3 = st.number_input("O3", min_value=0.0)

benzene = st.number_input(
    "Benzene",
    min_value=0.0
)

toluene = st.number_input(
    "Toluene",
    min_value=0.0
)

xylene = st.number_input(
    "Xylene",
    min_value=0.0
)

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict AQI"):

    # -------------------------
    # Date Features
    # -------------------------

    year = date.year
    month = date.month
    day = date.day

    # -------------------------
    # Season Feature
    # -------------------------

    if month in [12, 1, 2]:
        season = "Winter"

    elif month in [3, 4, 5]:
        season = "Summer"

    elif month in [6, 7, 8, 9]:
        season = "Monsoon"

    else:
        season = "PostMonsoon"

    # -------------------------
    # Create Empty DataFrame
    # -------------------------

    input_data = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    # -------------------------
    # Fill Pollutants
    # -------------------------

    input_data.loc[0, 'PM2.5'] = pm25

    input_data.loc[0, 'PM10'] = pm10

    input_data.loc[0, 'NO'] = no

    input_data.loc[0, 'NO2'] = no2

    input_data.loc[0, 'NOx'] = nox

    input_data.loc[0, 'NH3'] = nh3

    input_data.loc[0, 'CO'] = co

    input_data.loc[0, 'SO2'] = so2

    input_data.loc[0, 'O3'] = o3

    input_data.loc[0, 'Benzene'] = benzene

    input_data.loc[0, 'Toluene'] = toluene

    input_data.loc[0, 'Xylene'] = xylene

    # -------------------------
    # Fill Date Features
    # -------------------------

    input_data.loc[0, 'Year'] = year

    input_data.loc[0, 'Month'] = month

    input_data.loc[0, 'Day'] = day

    # -------------------------
    # Fill Season Dummy
    # -------------------------

    season_col = f"Season_{season}"

    if season_col in input_data.columns:

        input_data.loc[
            0,
            season_col
        ] = 1

    # -------------------------
    # Fill City Dummy
    # -------------------------

    city_col = f"City_{city}"

    if city_col in input_data.columns:

        input_data.loc[
            0,
            city_col
        ] = 1

    # -------------------------
    # Prediction
    # -------------------------

    predicted_aqi = model.predict(
        input_data
    )[0]


    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=predicted_aqi,
    title={'text':"Predicted AQI"},
    gauge={
        'axis':{'range':[0,500]}
    }
    ))

    st.plotly_chart(
    fig,
    use_container_width=True
    )

    # -------------------------
    # AQI Category
    # -------------------------

    if predicted_aqi <= 50:

        category = "Good 🟢"

    elif predicted_aqi <= 100:

        category = "Satisfactory 🟡"

    elif predicted_aqi <= 200:

        category = "Moderate 🟠"

    elif predicted_aqi <= 300:

        category = "Poor 🔴"

    elif predicted_aqi <= 400:

        category = "Very Poor 🟣"

    else:

        category = "Severe ⚫"

    # -------------------------
    # Display Results
    # -------------------------

    st.success(
        f"Predicted AQI : {predicted_aqi:.2f}"
    )

    st.info(
        f"AQI Category : {category}"
    )
