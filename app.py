import streamlit as st
import joblib
import numpy as np
import sklearn

# Pastikan kompatibilitas model
print("Scikit-learn version:", sklearn.__version__)

# Load model yang sudah disimpan
try:
    model_path = "random_forest_model_fixed.pkl"
    rf_model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    rf_model = None

# Judul aplikasi
st.title("Atlet Performance Classifier")
st.write("Masukkan nilai fitur untuk memprediksi kategori atlet (Beginner, Intermediate, Advance).")

# Input fitur
leg_power = st.number_input("Leg Power", min_value=0.0, format="%.2f")
hand_power = st.number_input("Hand Power", min_value=0.0, format="%.2f")
endurance = st.number_input("Endurance (Vo2 max)", min_value=0.0, format="%.2f")
speed = st.number_input("Speed", min_value=0.0, format="%.2f")

# Tombol prediksi
if st.button("Prediksi Kategori") and rf_model is not None:
    try:
        input_features = np.array([[leg_power, hand_power, endurance, speed]])
        prediction = rf_model.predict(input_features)[0]
        st.success(f"Kategori Atlet: {prediction}")
    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")
