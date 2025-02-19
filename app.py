import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fungsi untuk memuat dan membersihkan data
def load_data():
    df = pd.read_csv('Hasil_Tes_Atlet_Porda.csv', delimiter=';', encoding='utf-8')
    df = df[['Power Otot Tungkai', 'Hand Grip kanan', 'Hand Grip Kiri', 'Kecepatan', 'Berat Badan', 'Vo2 max', 'Gender']].copy()
    
    # Membersihkan data dari karakter non-numerik
    for col in ['Power Otot Tungkai', 'Hand Grip kanan', 'Hand Grip Kiri', 'Kecepatan', 'Berat Badan', 'Vo2 max']:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=True)  # Ganti koma ke titik
        df[col] = df[col].str.replace(r'[^0-9.]', '', regex=True)  # Hapus karakter selain angka dan titik
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Konversi ke numerik, NaN jika gagal
    
    # Mengisi nilai kosong dengan median
    df.fillna(df.median(), inplace=True)
    
    # Perhitungan fitur tambahan
    df['Leg Power'] = (2.21 * df['Berat Badan'] * (df['Power Otot Tungkai'] / 100))
    df['Hand Power'] = df['Hand Grip kanan'] / df['Hand Grip Kiri']
    df['Speed'] = 20 / df['Kecepatan']
    
    # Klasifikasi berdasarkan kategori
    def classify_vo2max(row):
        if row['Gender'] == 'Pria':
            if row['Vo2 max'] <= 24:
                return 'Beginner'
            elif 25 <= row['Vo2 max'] <= 36:
                return 'Intermediate'
            elif 37 <= row['Vo2 max'] < 50:
                return 'Advanced'
        elif row['Gender'] == 'Wanita':
            if row['Vo2 max'] <= 22:
                return 'Beginner'
            elif 23 <= row['Vo2 max'] <= 33:
                return 'Intermediate'
            elif 34 <= row['Vo2 max'] < 46:
                return 'Advanced'
        return 'Uncategorized'
    df['Endurance Category'] = df.apply(classify_vo2max, axis=1)
    
    def classify_speed(row):
        if row['Gender'] == 'Pria':
            if row['Speed'] >= 3.70:
                return 'Beginner'
            elif 3.31 <= row['Speed'] < 3.50:
                return 'Intermediate'
            elif row['Speed'] < 3.11:
                return 'Advanced'
        elif row['Gender'] == 'Wanita':
            if row['Speed'] >= 3.90:
                return 'Beginner'
            elif 3.51 <= row['Speed'] < 3.70:
                return 'Intermediate'
            elif row['Speed'] < 3.50:
                return 'Advanced'
        return 'Uncategorized'
    df['Speed Category'] = df.apply(classify_speed, axis=1)
    
    def classify_leg_power(row):
        leg_power = row['Leg Power']
        if row['Gender'] == 'Pria':
            if leg_power >= 79:
                return 'Advanced'
            elif 65 <= leg_power < 79:
                return 'Intermediate'
            else:
                return 'Beginner'
        elif row['Gender'] == 'Wanita':
            if leg_power >= 59:
                return 'Advanced'
            elif 49 <= leg_power < 59:
                return 'Intermediate'
            else:
                return 'Beginner'
    df['Leg Power Category'] = df.apply(classify_leg_power, axis=1)
    
    df['Overall Category'] = df[['Endurance Category', 'Speed Category', 'Leg Power Category']].mode(axis=1)[0]
    label_encoder = LabelEncoder()
    df['Overall Category Encoded'] = label_encoder.fit_transform(df['Overall Category'])
    return df

# Fungsi untuk melatih model
def train_model(df):
    features = df[['Leg Power', 'Hand Power', 'Speed', 'Vo2 max']]
    target = df['Overall Category Encoded']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    N = len(features_scaled)
    k = min(10, max(2, int(np.sqrt(N))))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for train_index, test_index in kf.split(features_scaled):
        X_train_fold, X_test_fold = features_scaled[train_index], features_scaled[test_index]
        y_train_fold, y_test_fold = target.iloc[train_index], target.iloc[test_index]
        
        model = RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_split=30, min_samples_leaf=15, max_features="sqrt", bootstrap=True, class_weight='balanced', random_state=42)
        model.fit(X_train_fold, y_train_fold)
    
    return model

# Fungsi untuk prediksi klasifikasi
def predict_category(model):
    st.write("### Prediksi Kategori Atlet")
    leg_power = st.number_input("Leg Power", min_value=0.0, format="%.2f")
    hand_power = st.number_input("Hand Power", min_value=0.0, format="%.2f")
    speed = st.number_input("Speed", min_value=0.0, format="%.2f")
    vo2_max = st.number_input("Vo2 Max", min_value=0.0, format="%.2f")
    
    if st.button("Prediksi Kategori"):
        input_data = np.array([[leg_power, hand_power, speed, vo2_max]])
        prediction = model.predict(input_data)
        st.success(f"Kategori Atlet: {prediction[0]}")

# Main function untuk aplikasi Streamlit
def main():
    st.title("Klasifikasi Atlet Berdasarkan Performa")
    df = load_data()
    model = train_model(df)
    predict_category(model)

if __name__ == "__main__":
    main()
