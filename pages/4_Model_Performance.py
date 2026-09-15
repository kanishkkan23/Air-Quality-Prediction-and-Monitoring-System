import streamlit as st

st.title("📈 Model Performance Dashboard")

st.metric("R² Score", "0.8407")
st.metric("MAE", "54.15")
st.metric("RMSE", "107.73")
st.metric("MSE", "11604.98")

st.markdown("---")

st.success(
"""
Random Forest achieved 84.07% variance explanation.
"""
)
