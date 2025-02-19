import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
def load_data():
    df = pd.read_csv("Hasil_Tes_Atlet_Porda.csv")
    df['Leg Power'] = (2.21 * df['Berat Badan'] * (df['Power Otot Tungkai'] / 100))
    df['Hand Power'] = df['Hand Grip kanan'] / df['Hand Grip Kiri']
    df['Speed'] = 20 / df['Kecepatan']
    
    # Endurance Category
    df['Endurance Category'] = df.apply(classify_vo2max, axis=1)
    df['Speed Category'] = df.apply(classify_speed, axis=1)
    df['Leg Power Category'] = df.apply(classify_leg_power, axis=1)
    df['Hand Power Category'] = df.apply(classify_handpower, axis=1)
    
    df['Overall Category'] = df[['Endurance Category', 'Speed Category', 'Leg Power Category', 'Hand Power Category']].mode(axis=1)[0]
    df['Overall Category Encoded'] = LabelEncoder().fit_transform(df['Overall Category'])
    return df

# Classification functions
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

def classify_handpower(row):
    if row['Gender'] == 'Pria':
        if row['Hand Power'] >= 1.30 or row['Hand Power'] < 0.85:
            return 'Beginner'
        elif 0.85 <= row['Hand Power'] < 1.15:
            return 'Intermediate'
        elif 1.15 <= row['Hand Power'] < 1.30:
            return 'Advanced'
    elif row['Gender'] == 'Wanita':
        if row['Hand Power'] >= 1.25 or row['Hand Power'] < 0.80:
            return 'Beginner'
        elif 0.80 <= row['Hand Power'] < 1.10:
            return 'Intermediate'
        elif 1.10 <= row['Hand Power'] < 1.25:
            return 'Advanced'
    return 'Uncategorized'

# Train model
def train_model(df):
    features = df[['Leg Power', 'Hand Power', 'Speed', 'Vo2 max']]
    target = df['Overall Category Encoded']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    N = len(features_scaled)
    k = min(10, max(2, int(np.sqrt(N))))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for train_index, test_index in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        model = RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_split=30,
                                       min_samples_leaf=15, max_features="sqrt",
                                       bootstrap=True, class_weight='balanced',
                                       random_state=42)
        model.fit(X_train, y_train)
    return model

# Streamlit UI
st.title("Klasifikasi Tingkat Atlet")
df = load_data()
model = train_model(df)

leg_power = st.number_input("Leg Power", min_value=0.0, max_value=100.0, step=0.1)
hand_power = st.number_input("Hand Power", min_value=0.0, max_value=100.0, step=0.1)
endurance = st.number_input("Endurance (Vo2 Max)", min_value=0.0, max_value=100.0, step=0.1)
speed = st.number_input("Speed", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Klasifikasikan"):
    input_data = [[leg_power, hand_power, speed, endurance]]
    prediction = model.predict(input_data)[0]
    label_map = dict(enumerate(df['Overall Category'].unique()))
    st.write(f"Hasil Klasifikasi: **{label_map[prediction]}**")
