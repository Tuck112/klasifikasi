import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Streamlit UI
st.title("Atlet Performance Classification")
st.write("Masukkan nilai untuk leg power, hand power endurance, dan speed:")

# File uploader
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is not None:
    # Load dataset
data = pd.read_csv(uploaded_file)
    
    # Select features and target
    features = ['leg power', 'hand power endurance', 'speed']
    target = 'classification'  # Assuming this is the column with labels
    
    # Encode target labels
    le = LabelEncoder()
    data[target] = le.fit_transform(data[target])
    
    # Split data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = '/mnt/data/random_forest_model.pkl'
    joblib.dump(model, model_path)
    
    # User input
    leg_power = st.number_input("Leg Power", min_value=0.0, max_value=100.0, step=0.1)
    hand_power = st.number_input("Hand Power Endurance", min_value=0.0, max_value=100.0, step=0.1)
    speed = st.number_input("Speed", min_value=0.0, max_value=100.0, step=0.1)
    
    # Prediction
    if st.button("Predict"):
        model = joblib.load(model_path)
        input_data = np.array([[leg_power, hand_power, speed]])
        prediction = model.predict(input_data)
        predicted_class = le.inverse_transform(prediction)[0]
        st.success(f"Classification Result: {predicted_class}")
