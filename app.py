import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Fungsi untuk memuat dan membersihkan data
def load_data():
    file_path = "Hasil_Tes_Atlet_Porda.csv"
    data = pd.read_csv(file_path, delimiter=";", encoding="utf-8")
    
    # Konversi tipe data numerik
    columns_to_convert = ['Berat Badan', 'Power Otot Tungkai', 'Hand Grip kanan', 'Hand Grip Kiri', 'Kecepatan', 'Vo2 max']
    for col in columns_to_convert:
        data[col] = data[col].astype(str).str.replace(',', '.', regex=True)
        data[col] = data[col].str.replace(r'[^0-9.]', '', regex=True)
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    # Hitung fitur
    data['Leg Power'] = 2.21 * data['Berat Badan'] * (data['Power Otot Tungkai'] / 100)
    data['Hand Power'] = data['Hand Grip kanan'] / data['Hand Grip Kiri']
    data['Speed'] = 20 / data['Kecepatan']
    
    return data

# Fungsi untuk melatih model
def train_model(data):
    features = data[['Leg Power', 'Hand Power', 'Speed', 'Vo2 max']]
    target = data['Overall Category']
    
    # Encoding target
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)
    
    # Normalisasi fitur
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Model Random Forest
    model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(features_scaled, target_encoded)
    
    return model, scaler, label_encoder

# Load data dan latih model
data = load_data()
model, scaler, label_encoder = train_model(data)

# Streamlit UI
st.title("Klasifikasi Atlet berdasarkan Tes Fisik")
gender = st.selectbox("Pilih Gender", ["Pria", "Wanita"])
leg_power = st.number_input("Masukkan Leg Power", min_value=0.0, format="%.2f")
hand_power = st.number_input("Masukkan Hand Power", min_value=0.0, format="%.2f")
speed = st.number_input("Masukkan Speed", min_value=0.0, format="%.2f")
endurance = st.number_input("Masukkan Endurance (Vo2 max)", min_value=0.0, format="%.2f")

if st.button("Klasifikasikan"):
    input_data = np.array([[leg_power, hand_power, speed, endurance]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Hasil Klasifikasi: {predicted_category}")
