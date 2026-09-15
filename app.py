import streamlit as st

st.set_page_config(
    page_title="Air Quality Monitoring",
    page_icon="🌍",
    layout="wide"
)

# Hide sidebar

st.markdown("""
<style>
section[data-testid="stSidebar"]{
display:none;
}
</style>
""", unsafe_allow_html=True)

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

import base64

def add_bg():

    with open("images/background.jpg", "rb") as image_file:
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


# Hero Section

st.markdown("""

<div class="hero">

<h1>🌍 Air Quality Monitoring and Prediction System</h1>

<h3>Machine Learning Based AQI Analysis Dashboard</h3>

<p style="font-size:20px;color:white;">
Predict Air Quality Index using Random Forest Machine Learning
</p>

</div>

""", unsafe_allow_html=True)

st.write("")

# Navigation Buttons

col1,col2=st.columns(2)

with col1:

    if st.button(
        "🔮 AQI Prediction",
        use_container_width=True
    ):
        st.switch_page(
            "pages/2_AQI_Prediction.py"
        )

    if st.button(
        "📈 Model Performance",
        use_container_width=True
    ):
        st.switch_page(
            "pages/4_Model_Performance.py"
        )

with col2:

    if st.button(
        "📊 Data Visualization",
        use_container_width=True
    ):
        st.switch_page(
            "pages/3_Data_Visualization.py"
        )

    if st.button(
        "ℹ️ Project Information",
        use_container_width=True
    ):
        st.switch_page(
            "pages/5_Project_Information.py"
        )

st.write("")

st.markdown("---")

st.subheader("🌈 AQI Category Guide")

st.markdown("""
<div style="
background-color:#d4edda;
padding:12px;
border-radius:10px;
margin-bottom:10px;">
🟢 <b>Good</b> : 0 - 50
</div>

<div style="
background-color:#d6e4f0;
padding:12px;
border-radius:10px;
margin-bottom:10px;">
🟡 <b>Satisfactory</b> : 51 - 100
</div>

<div style="
background-color:#fff3cd;
padding:12px;
border-radius:10px;
margin-bottom:10px;">
🟠 <b>Moderate</b> : 101 - 200
</div>

<div style="
background-color:#f8d7da;
padding:12px;
border-radius:10px;
margin-bottom:10px;">
🔴 <b>Poor</b> : 201 - 300
</div>

<div style="
background-color:#e2d6f5;
padding:12px;
border-radius:10px;
margin-bottom:10px;">
🟣 <b>Very Poor</b> : 301 - 400
</div>

<div style="
background-color:#d6d8db;
padding:12px;
border-radius:10px;
margin-bottom:10px;">
⚫ <b>Severe</b> : 401 - 500
</div>
""", unsafe_allow_html=True)
