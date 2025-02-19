import pickle
import streamlit as st
import numpy as np

# Load model Random Forest
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Klasifikasi Atlet: Beginner, Intermediate, Advance")

# Input dari pengguna
leg_power = st.number_input("Leg Power", min_value=0.0, max_value=100.0, step=0.1)
hand_power = st.number_input("Hand Power", min_value=0.0, max_value=100.0, step=0.1)
speed = st.number_input("Speed", min_value=0.0, max_value=100.0, step=0.1)
vo2_max = st.number_input("Vo2 Max", min_value=0.0, max_value=100.0, step=0.1)

# Prediksi
if st.button("Prediksi Klasifikasi"):
    input_data = np.array([[leg_power, hand_power, speed, vo2_max]])
    prediction = model.predict(input_data)

    class_mapping = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}
    st.write(f"**Hasil Klasifikasi:** {class_mapping[prediction[0]]}")
