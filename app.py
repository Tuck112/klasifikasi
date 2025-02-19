import streamlit as st
import joblib
import numpy as np

# Load model yang telah diekspor
model_path = "random_forest_model.pkl"
rf_model = joblib.load(model_path)

# Judul aplikasi
st.title("Klasifikasi Atlet Berdasarkan Performa")
st.write("Masukkan nilai leg power, hand power endurance, dan speed untuk mendapatkan klasifikasi atlet.")

# Input dari pengguna
leg_power = st.number_input("Leg Power (cm)", min_value=0.0, format="%.2f")
hand_power_endurance = st.number_input("Hand Power Endurance", min_value=0.0, format="%.2f")
speed = st.number_input("Speed (detik)", min_value=0.0, format="%.2f")

# Tombol Prediksi
if st.button("Prediksi Kategori"):
    input_data = np.array([[leg_power, hand_power_endurance, speed]])
    prediction = rf_model.predict(input_data)
    st.success(f"Kategori Atlet: {prediction[0]}")
