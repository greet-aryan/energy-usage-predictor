import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the saved model
model = joblib.load("energy_predictor.pkl")

# Streamlit UI
st.title("⚡ Smart Home Energy Usage Predictor")

# Input fields
date = st.date_input("Date")
time = st.time_input("Time")
temperature = st.number_input("Outdoor Temperature (°C)", value=25.0)
household_size = st.number_input("Household Size", value=3, min_value=1)
appliance_type = st.selectbox("Appliance Type", ["Refrigerator", "TV", "Heater", "Washing Machine"])
season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])

if st.button("Predict Energy Usage (kWh)"):
    # Prepare input
    dt = pd.to_datetime(f"{date} {time}")
    hour = dt.hour
    day_of_week = dt.dayofweek
    month = dt.month
    is_weekend = int(day_of_week in [5, 6])
    
    input_data = pd.DataFrame([{
        "Outdoor Temperature (°C)": temperature,
        "Household Size": household_size,
        "Appliance Type": appliance_type,
        "Season": season,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * (month-1) / 12),
        "month_cos": np.cos(2 * np.pi * (month-1) / 12)
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Estimated Energy Consumption: **{prediction:.2f} kWh**")
