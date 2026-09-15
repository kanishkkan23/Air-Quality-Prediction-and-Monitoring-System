import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Air_Quality_Feature_Engineered.csv")

# AQI Distribution Pie-Chart

'''def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

df['AQI_Category'] = df['AQI'].apply(aqi_category)


aqi_counts = df['AQI_Category'].value_counts()


plt.figure(figsize=(8,8))

plt.pie(
    aqi_counts,
    labels=aqi_counts.index,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Air Quality Index (AQI) Category Distribution")

plt.legend(
    title="AQI Categories",
    bbox_to_anchor=(1, 1)
)

plt.tight_layout()
plt.show()'''


# AQI Correlation HeatMap

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()


# AQI Trends Over the Year


'''df['Date'] = pd.to_datetime(df['Date'])

daily_aqi = df.groupby('Date')['AQI'].mean()

plt.figure(figsize=(14,6))

plt.plot(daily_aqi.index, daily_aqi.values)

plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average AQI")

plt.show()'''

# Top 10 Cities of Average AQI

'''city_aqi = df.groupby('City')['AQI'].mean()

city_aqi = city_aqi.sort_values(
    ascending=False
).head(10)

plt.figure(figsize=(10,6))

city_aqi.plot(kind='bar')

plt.title("Top 10 Cities by Average AQI")
plt.xlabel("City")
plt.ylabel("Average AQI")'''

plt.show()
