import streamlit as st
import pickle
import numpy as np

# Load model Random Forest yang sudah dilatih
model_filename = "random_forest.pkl"  # Pastikan file ini tersedia
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Judul Aplikasi
st.title("Klasifikasi Atlet: Beginner, Intermediate, Advance")

# Input dari pengguna
leg_power = st.number_input("Leg Power", min_value=0.0, max_value=100.0, step=0.1)
hand_power = st.number_input("Hand Power", min_value=0.0, max_value=100.0, step=0.1)
endurance = st.number_input("Endurance", min_value=0.0, max_value=100.0, step=0.1)
speed = st.number_input("Speed", min_value=0.0, max_value=100.0, step=0.1)

# Tombol untuk melakukan prediksi
if st.button("Prediksi Klasifikasi"):
    # Menggabungkan input menjadi array
    input_data = np.array([[leg_power, hand_power, endurance, speed]])
    
    # Melakukan prediksi
    prediction = model.predict(input_data)
    
    # Menampilkan hasil prediksi
    class_mapping = {0: "Beginner", 1: "Intermediate", 2: "Advance"}  # Pastikan sesuai dengan model Anda
    st.write(f"**Hasil Klasifikasi:** {class_mapping[prediction[0]]}")

# Jalankan aplikasi dengan command:
# streamlit run nama_file.py
