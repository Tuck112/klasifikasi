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
        if row['Vo2 max'] <= 24:
            return 'Beginner'
        elif 25 <= row['Vo2 max'] <= 36:
            return 'Intermediate'
        else:
            return 'Advanced'
    data['Endurance Category'] = data.apply(classify_vo2max, axis=1)
    
    # Klasifikasi Speed
    def classify_speed(row):
        if row['Speed'] >= 3.70:
            return 'Beginner'
        elif 3.31 <= row['Speed'] < 3.50:
            return 'Intermediate'
        else:
            return 'Advanced'
    data['Speed Category'] = data.apply(classify_speed, axis=1)
    
    # Klasifikasi Leg Power
    def classify_leg_power(row):
        if row['Leg Power'] >= 79:
            return 'Advanced'
        elif 65 <= row['Leg Power'] < 79:
            return 'Intermediate'
        else:
            return 'Beginner'
    data['Leg Power Category'] = data.apply(classify_leg_power, axis=1)
    
    # Klasifikasi Hand Power
    def classify_handpower(row):
        if row['Hand Power'] >= 1.30 or row['Hand Power'] < 0.85:
            return 'Beginner'
        elif 0.85 <= row['Hand Power'] < 1.15:
            return 'Intermediate'
        else:
            return 'Advanced'
    data['Hand Power Category'] = data.apply(classify_handpower, axis=1)
    
    # Pastikan kategori keseluruhan dihitung sebelum dikembalikan
    data['Overall Category'] = data[['Endurance Category', 'Speed Category', 'Leg Power Category', 'Hand Power Category']].mode(axis=1)[0]
    
    return data

# Fungsi untuk melatih model
def train_model(data):
    if 'Overall Category' not in data.columns:
        raise KeyError("Kolom 'Overall Category' tidak ditemukan dalam data. Pastikan preprocessing sudah benar.")
    
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
st.set_page_config(page_title="Klasifikasi Atlet", page_icon="🏅", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1571019614242-c5c5dee9ae8a");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .title-container {
            text-align: center;
            color: white;
            padding: 10px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
        }
        h1 {color: #F4D03F;}
        .stButton>button {
            background-color: #F4D03F;
            color: black;
            font-size: 18px;
            padding: 12px;
            border-radius: 10px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #D4AC0D;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title-container'><h1>🏋️ Klasifikasi Atlet berdasarkan Tes Fisik</h1></div>", unsafe_allow_html=True)
st.subheader("Masukkan data atlet untuk mendapatkan hasil klasifikasi")

with st.form("classification_form"):
    gender = st.selectbox("Pilih Gender", ["Pria", "Wanita"], index=None, placeholder="Pilih jenis kelamin")
    leg_power = st.number_input("Masukkan Leg Power", min_value=0.0, format="%.2f", placeholder="Masukkan nilai Leg Power")
    hand_power = st.number_input("Masukkan Hand Power", min_value=0.0, format="%.2f", placeholder="Masukkan nilai Hand Power")
    speed = st.number_input("Masukkan Speed", min_value=0.0, format="%.2f", placeholder="Masukkan nilai Speed")
    endurance = st.number_input("Masukkan Endurance (Vo2 max)", min_value=0.0, format="%.2f", placeholder="Masukkan nilai Endurance")
    submit_button = st.form_submit_button("🔍 Klasifikasikan")

if submit_button:
    input_data = np.array([[leg_power, hand_power, speed, endurance]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    
    st.markdown(f"<h3 style='text-align: center; color: white; background: rgba(0, 0, 0, 0.6); padding: 10px; border-radius: 10px;'>Hasil Klasifikasi: {predicted_category}</h3>", unsafe_allow_html=True)
    
    st.success("Klasifikasi berhasil dilakukan! 🚀")

    
