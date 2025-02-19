import streamlit as st
import joblib
import numpy as np

# Load model dan label encoder
model = joblib.load("random_forest_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

st.title("Atlet Classification App")
st.write("Masukkan nilai untuk mendapatkan klasifikasi atlet: Beginner, Intermediate, atau Advance.")

# Input user
leg_power = st.number_input("Leg Power", min_value=0.0, format="%.2f")
hand_power = st.number_input("Hand Power", min_value=0.0, format="%.2f")
endurance = st.number_input("Endurance", min_value=0.0, format="%.2f")
speed = st.number_input("Speed", min_value=0.0, format="%.2f")

if st.button("Klasifikasikan"):
    # Menggabungkan input dalam bentuk array
    input_data = np.array([[leg_power, hand_power, endurance, speed]])
    
    # Melakukan prediksi
    prediction = model.predict(input_data)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"Kategori Atlet: {predicted_class}")
