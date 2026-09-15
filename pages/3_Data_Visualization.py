import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "Dataset/Air_Quality_Feature_Engineered.csv"
)

st.title("📊 Data Visualization Dashboard")

st.title("AQI DISTRIBUTION:")

fig,ax=plt.subplots()

ax.hist(df['AQI'],bins=20)

ax.set_title("AQI Distribution")

st.pyplot(fig)

st.title("AQI DISTRIBUTION PIE CHART")

def category(aqi):

    if aqi<=50:
        return "Good"

    elif aqi<=100:
        return "Satisfactory"

    elif aqi<=200:
        return "Moderate"

    elif aqi<=300:
        return "Poor"

    elif aqi<=400:
        return "Very Poor"

    else:
        return "Severe"

df["AQI_Category"]=df["AQI"].apply(category)

aqi_counts=df["AQI_Category"].value_counts()

fig,ax=plt.subplots()

ax.pie(
    aqi_counts,
    labels=aqi_counts.index,
    autopct="%1.1f%%"
)

st.pyplot(fig)

st.title("AQI FEATURE IMPORTANCE:")


feature_importance = {
    "CO":70.57,
    "NO":12.15,
    "PM2.5":5.40,
    "SO2":1.75,
    "O3":1.58,
    "NO2":1.33,
    "NOx":1.21,
    "PM10":1.14,
    "Toluene":1.08,
    "Xylene":1.08
}

st.bar_chart(feature_importance)



