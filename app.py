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
    
    # Klasifikasi Endurance
    def classify_vo2max(row):
        if row['Gender'] == 'Pria':
            return 'Beginner' if row['Vo2 max'] <= 24 else 'Intermediate' if row['Vo2 max'] <= 36 else 'Advanced'
        else:
            return 'Beginner' if row['Vo2 max'] <= 22 else 'Intermediate' if row['Vo2 max'] <= 33 else 'Advanced'
    data['Endurance Category'] = data.apply(classify_vo2max, axis=1)
    
    # Klasifikasi Speed
    def classify_speed(row):
        if row['Gender'] == 'Pria':
            return 'Beginner' if row['Speed'] >= 3.70 else 'Intermediate' if row['Speed'] >= 3.31 else 'Advanced'
        else:
            return 'Beginner' if row['Speed'] >= 3.90 else 'Intermediate' if row['Speed'] >= 3.51 else 'Advanced'
    data['Speed Category'] = data.apply(classify_speed, axis=1)
    
    # Klasifikasi Leg Power
    def classify_leg_power(row):
        leg_power = row['Leg Power']
        if row['Gender'] == 'Pria':
            return 'Advanced' if leg_power >= 79 else 'Intermediate' if leg_power >= 65 else 'Beginner'
        else:
            return 'Advanced' if leg_power >= 59 else 'Intermediate' if leg_power >= 49 else 'Beginner'
    data['Leg Power Category'] = data.apply(classify_leg_power, axis=1)
    
    # Klasifikasi Hand Power
    def classify_handpower(row):
        if row['Gender'] == 'Pria':
            return 'Beginner' if row['Hand Power'] >= 1.30 or row['Hand Power'] < 0.85 else 'Intermediate' if row['Hand Power'] >= 0.85 else 'Advanced'
        else:
            return 'Beginner' if row['Hand Power'] >= 1.25 or row['Hand Power'] < 0.80 else 'Intermediate' if row['Hand Power'] >= 0.80 else 'Advanced'
    data['Hand Power Category'] = data.apply(classify_handpower, axis=1)
    
    # Tentukan kategori keseluruhan
    data['Overall Category'] = data[['Endurance Category', 'Speed Category', 'Leg Power Category', 'Hand Power Category']].mode(axis=1)[0]
    
    return data

# Fungsi untuk melatih model
def train_model(data):
    features = data[['Leg Power', 'Hand Power', 'Speed']]
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

if st.button("Klasifikasikan"):
    input_data = np.array([[leg_power, hand_power, speed]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Hasil Klasifikasi: {predicted_category}")
