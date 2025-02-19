import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("random_forest_model.joblib", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Athlete Classification using Random Forest")
st.write("Enter the required values to classify the athlete's performance.")

# Input fields
leg_power = st.number_input("Leg Power", min_value=0.0, format="%.2f")
hand_power = st.number_input("Hand Power", min_value=0.0, format="%.2f")
endurance = st.number_input("Endurance (VO2 Max)", min_value=0.0, format="%.2f")
speed = st.number_input("Speed", min_value=0.0, format="%.2f")

# Predict button
if st.button("Classify Athlete"):
    # Convert inputs to NumPy array
    input_data = np.array([[leg_power, hand_power, endurance, speed]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"The athlete's classification is: **{prediction[0]}**")
