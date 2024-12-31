import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load("telecom_churn_model.pkl")

# Streamlit App
st.title("Telecom Service Churn Prediction")
st.write("Predict if a user will change their telecom service.")

# Input sliders for user data
user_data = {
    "Age": st.slider("Age", 18, 65, 30),
    "Monthly_Bill_Amount": st.slider("Monthly Bill Amount", 100, 2000, 500),
    "Network_Quality_Rating": st.slider("Network Quality Rating", 1, 5, 3),
    "Customer_Service_Rating": st.slider("Customer Service Rating", 1, 5, 3),
    "Internet_Speed_Rating": st.slider("Internet Speed Rating", 1, 5, 3),
    "Loyalty_Duration_Years": st.slider("Loyalty Duration (Years)", 0, 10, 5),
    "Is_Satisfied": st.selectbox("Is Satisfied", [0, 1]),
}

# Convert user input to a DataFrame
user_df = pd.DataFrame([user_data])

# Predict churn
if st.button("Predict"):
    prediction = model.predict(user_df)
    result = "Yes" if prediction[0] == 1 else "No"
    st.success(f"Will Change Service: {result}")
