import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
def load_data():
    df = pd.read_csv("Hasil_Tes_Atlet_Porda.csv", delimiter=";", encoding="utf-8")
    
    # Data Cleaning
    columns_to_convert = ['Kelentukan', 'Kelincahan', 'Kecepatan', 'Hand Grip kanan',
                          'Hand Grip Kiri', 'Vo2 max', 'Berat Badan', 'Power Otot Tungkai']
    for col in columns_to_convert:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=True)
        df[col] = df[col].str.replace(r'[^0-9.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0).astype(float)
    
    df['Usia'] = df['Usia'].round().astype(int)
    
    # Feature Engineering
    df['Leg Power'] = (2.21 * df['Berat Badan'] * (df['Power Otot Tungkai'] / 100))
    df['Hand Power'] = df['Hand Grip kanan'] / df['Hand Grip Kiri']
    df['Speed'] = 20 / df['Kecepatan']
    
    # Pastikan kategori klasifikasi dibuat sebelum encoding
    if {'Endurance Category', 'Speed Category', 'Leg Power Category', 'Hand Power Category'}.issubset(df.columns):
        df['Overall Category'] = df[['Endurance Category', 'Speed Category', 'Leg Power Category', 'Hand Power Category']].mode(axis=1)[0]
    else:
        st.error("Kolom kategori belum lengkap, pastikan semua kategori tersedia sebelum dikombinasikan.")
    
    return df

# Encode Target Label
def encode_labels(df):
    if 'Overall Category' in df.columns and not df['Overall Category'].isnull().all():
        label_encoder = LabelEncoder()
        df['Overall Category Encoded'] = label_encoder.fit_transform(df['Overall Category'])
        return df, label_encoder
    else:
        st.error("Kolom 'Overall Category' kosong atau tidak ditemukan dalam dataset. Pastikan semua kategori telah dihitung dengan benar.")
        return df, None

# Train Model with KFold
def train_model(df):
    features = df[['Leg Power', 'Hand Power', 'Speed', 'Vo2 max']]
    target = df['Overall Category Encoded']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    N = len(features_scaled)
    k = min(10, max(2, int(np.sqrt(N))))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_split=30,
                                      min_samples_leaf=15, max_features="sqrt", bootstrap=True,
                                      class_weight='balanced', random_state=42)
    
    for train_index, test_index in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        rf_model.fit(X_train, y_train)
    
    return rf_model, scaler

# Load and preprocess data
df = load_data()
df, label_encoder = encode_labels(df)

if label_encoder is not None:
    model, scaler = train_model(df)

    # Streamlit UI
    st.title("Klasifikasi Tingkat Atlet")
    st.write("Masukkan nilai untuk menentukan tingkat atlet: Beginner, Intermediate, atau Advanced")

    leg_power = st.number_input("Leg Power", min_value=0.0, max_value=500.0, step=0.1)
    hand_power = st.number_input("Hand Power", min_value=0.0, max_value=5.0, step=0.01)
    endurance = st.number_input("VO2 Max", min_value=0.0, max_value=100.0, step=0.1)
    speed = st.number_input("Speed", min_value=0.0, max_value=10.0, step=0.01)

    if st.button("Klasifikasikan"):
        input_data = np.array([[leg_power, hand_power, speed, endurance]])
        input_data_scaled = scaler.transform(input_data)
        prediction_encoded = model.predict(input_data_scaled)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        st.write(f"Hasil Klasifikasi: **{prediction}**")
