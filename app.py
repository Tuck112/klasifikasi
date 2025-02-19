import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Fungsi untuk memuat dan membersihkan data
def load_data():
    df = pd.read_csv('Hasil_Tes_Atlet_Porda.csv', delimiter=';', encoding='utf-8')
    df = df[['Power Otot Tungkai', 'Hand Grip kanan', 'Hand Grip Kiri', 'Kecepatan']].copy()
    
    # Membersihkan data
    df['Power Otot Tungkai'] = df['Power Otot Tungkai'].astype(str).str.replace(',', '.').astype(float)
    df['Hand Grip kanan'] = df['Hand Grip kanan'].astype(str).str.replace(',', '.').astype(float)
    df['Hand Grip Kiri'] = df['Hand Grip Kiri'].astype(str).str.replace(',', '.').astype(float)
    df['Kecepatan'] = df['Kecepatan'].astype(str).str.replace(',', '.').astype(float)
    
    # Menambahkan fitur "Hand Power Endurance"
    df['Hand Power Endurance'] = (df['Hand Grip kanan'] + df['Hand Grip Kiri']) / 2
    df = df[['Power Otot Tungkai', 'Hand Power Endurance', 'Kecepatan']]
    
    # Menentukan kategori berdasarkan kuartil
    df['Score'] = df.mean(axis=1)
    q1, q3 = df['Score'].quantile([0.25, 0.75])
    df['Category'] = df['Score'].apply(lambda x: 'Beginner' if x <= q1 else ('Intermediate' if x <= q3 else 'Advance'))
    df.drop(columns=['Score'], inplace=True)
    return df

# Fungsi untuk menampilkan statistik deskriptif
def display_data_statistics(df):
    st.write("### Data Head")
    st.write(df.head())
    st.write("### Data Shape")
    st.write(df.shape)
    st.write("### Data Description")
    st.write(df.describe())

# Fungsi untuk menampilkan plot
def display_plots(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['Power Otot Tungkai'], ax=ax[0], kde=True)
    ax[1].boxplot(df['Power Otot Tungkai'])
    st.pyplot(fig)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['Hand Power Endurance'], ax=ax[0], kde=True)
    ax[1].boxplot(df['Hand Power Endurance'])
    st.pyplot(fig)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['Kecepatan'], ax=ax[0], kde=True)
    ax[1].boxplot(df['Kecepatan'])
    st.pyplot(fig)

# Fungsi untuk melatih model
def train_model(df):
    X = df[['Power Otot Tungkai', 'Hand Power Endurance', 'Kecepatan']]
    y = df['Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Fungsi untuk prediksi klasifikasi
def predict_category(model):
    st.write("### Prediksi Kategori Atlet")
    leg_power = st.number_input("Leg Power (cm)", min_value=0.0, format="%.2f")
    hand_power_endurance = st.number_input("Hand Power Endurance", min_value=0.0, format="%.2f")
    speed = st.number_input("Speed (detik)", min_value=0.0, format="%.2f")
    
    if st.button("Prediksi Kategori"):
        input_data = np.array([[leg_power, hand_power_endurance, speed]])
        prediction = model.predict(input_data)
        st.success(f"Kategori Atlet: {prediction[0]}")

# Main function untuk aplikasi Streamlit
def main():
    st.title("Klasifikasi Atlet Berdasarkan Performa")
    df = load_data()
    display_data_statistics(df)
    display_plots(df)
    model = train_model(df)
    predict_category(model)

if __name__ == "__main__":
    main()
