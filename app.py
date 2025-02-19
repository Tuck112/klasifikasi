import streamlit as st
import joblib
import numpy as np

# Load model dan encoder
model_data = joblib.load("random_forest_model_compatible.pkl")
model = model_data['model']
label_encoder = model_data['label_encoder']

# Judul aplikasi
st.title("Klasifikasi Atlet Berdasarkan Performa")
st.write("Masukkan nilai untuk menentukan kategori atlet")

# Input fitur
leg_power = st.number_input("Leg Power", min_value=0.0, step=0.1)
hand_power = st.number_input("Hand Power", min_value=0.0, step=0.1)
endurance = st.number_input("Endurance (Vo2 max)", min_value=0.0, step=0.1)
speed = st.number_input("Speed", min_value=0.0, step=0.01)

# Tombol prediksi
if st.button("Prediksi Kategori"):
    # Format input data
    input_data = np.array([[leg_power, hand_power, endurance, speed]])
    
    # Prediksi menggunakan model
    prediction = model.predict(input_data)
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    
    # Tampilkan hasil prediksi
    st.success(f"Kategori Atlet: {predicted_category}")

# Jalankan aplikasi dengan perintah: streamlit run app.py
