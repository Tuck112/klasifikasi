import pickle
import streamlit as st
import numpy as np

# Load model
with open("random_forest.pkl", "rb") as file:
    model = pickle.load(file)

# Pastikan model memiliki metode predict()
if not hasattr(model, "predict"):
    st.error("❌ Model tidak memiliki metode predict(). Periksa kembali file .pkl Anda!")
    st.stop()

st.title("Klasifikasi Atlet: Beginner, Intermediate, Advance")

# Input dari pengguna
leg_power = st.number_input("Leg Power", min_value=0.0, max_value=100.0, step=0.1)
hand_power = st.number_input("Hand Power", min_value=0.0, max_value=100.0, step=0.1)
endurance = st.number_input("Endurance", min_value=0.0, max_value=100.0, step=0.1)
speed = st.number_input("Speed", min_value=0.0, max_value=100.0, step=0.1)

# Prediksi
if st.button("Prediksi Klasifikasi"):
    input_data = np.array([[leg_power, hand_power, endurance, speed]])
    
    try:
        prediction = model.predict(input_data)
        class_mapping = {0: "Beginner", 1: "Intermediate", 2: "Advance"}
        st.write(f"**Hasil Klasifikasi:** {class_mapping[prediction[0]]}")
    except Exception as e:
        st.error(f"❌ Error saat melakukan prediksi: {e}")
