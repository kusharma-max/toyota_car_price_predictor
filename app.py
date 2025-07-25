# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Toyota Car Price Predictor", layout="wide")

st.title("🚗 Toyota Car Price Predictor")
scaler = joblib.load("scaler.pkl")


# Load model and feature columns
model = joblib.load("toyota_model.pkl")
model_columns = joblib.load("model_columns.pkl")


# ---------- SLIDERS IN 2 COLUMNS ----------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("📅 Age of Car (in months)", min_value=1, max_value=80, value=60, step=1)

with col2:
    km = st.slider("🛣️ Kilometers Driven", min_value=1, max_value=243000, value=50000, step=1000)

st.markdown("---")

# ---------- FIRST ROW OF INPUTS ----------
r1c1, r1c2, r1c3 = st.columns(3)

with r1c1:
    fuel_type = st.selectbox("⛽ Fuel Type", ["Diesel", "Petrol", "CNG"])

with r1c2:
    hp = st.number_input("🏇 Horsepower (HP)", min_value=69, max_value=192, value=90)

with r1c3:
    cc = st.number_input("🔧 Engine CC", min_value=1300, max_value=16000, value=1400)

# ---------- SECOND ROW OF INPUTS ----------
r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    automatic = st.selectbox("⚙️ Transmission Type", ["Manual", "Automatic"])

with r2c2:
    doors = st.selectbox("🚪 Number of Doors", [2, 3, 4, 5])

with r2c3:
    gears = st.selectbox("🔁 Number of Gears", [3, 4, 5, 6])

# Fixed inputs
cylinders = 4
weight = st.number_input("⚖️ Weight (kg)", min_value=1000, max_value=1615, value=1200)
auto_flag = 1 if automatic == "Automatic" else 0

st.markdown("---")

# ---------- PREPARE INPUT FOR MODEL ----------
input_dict = {
    'age_months': age,
    'km': km,
    'hp': hp,
    'automatic': auto_flag,
    'cc': cc,
    'doors': doors,
    'cylinders': cylinders,
    'gears': gears,
    'weight': weight,
    'fuel_type_Diesel': 1 if fuel_type == "Diesel" else 0,
    'fuel_type_Petrol': 1 if fuel_type == "Petrol" else 0
    # 'fuel_type_CNG': 1 if fuel_type == "CNG" else 0,
}

# Ensure all expected columns are present
for col in model_columns:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])

# 4) Scale numeric features
num_cols = ['age_months', 'km', 'hp', 'automatic', 'cc', 'doors','cylinders', 'gears', 'weight']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# ---------- PREDICTION ----------
center = st.columns([3, 2, 3])[1]
with center:
    if st.button("🎯 Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")
