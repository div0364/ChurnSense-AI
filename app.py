import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Generate synthetic dataset for training and testing
def generate_data():
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 65, 5000),
        "Monthly_Bill_Amount": np.random.randint(100, 2000, 5000),
        "Network_Quality_Rating": np.random.randint(1, 6, 5000),
        "Customer_Service_Rating": np.random.randint(1, 6, 5000),
        "Internet_Speed_Rating": np.random.randint(1, 6, 5000),
        "Loyalty_Duration_Years": np.random.randint(0, 10, 5000),
        "Is_Satisfied": np.random.choice([0, 1], 5000),
    }
    df = pd.DataFrame(data)

    # Target variable based on logical rules
    df["Will_Change_Service"] = (
        (df["Network_Quality_Rating"] <= 2) |
        (df["Customer_Service_Rating"] <= 2) |
        (df["Internet_Speed_Rating"] <= 2) |
        (df["Monthly_Bill_Amount"] > 1500) |
        (df["Is_Satisfied"] == 0)
    ).astype(int)

    return df

def train_model():
    # Load and preprocess data
    df = generate_data()
    X = df.drop(columns=["Will_Change_Service"])
    y = df["Will_Change_Service"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with increased estimators and tuned parameters
    model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy, scaler, df.drop(columns=["Will_Change_Service"]).columns

# Streamlit App
def main():
    st.set_page_config(page_title="Telecom Churn Prediction", layout="wide", initial_sidebar_state="expanded")
    st.title("Telecom Service Churn Prediction")

    model, accuracy, scaler, features = train_model()

    st.subheader("Model Accuracy")
    st.success(f"Accuracy: {accuracy:.2%}")

    st.subheader("Make Predictions")
    st.write("Adjust the sliders to input user details and predict if they will change their service.")

    user_data = {}
    for feature in features:
        if feature == "Is_Satisfied":
            user_data[feature] = st.selectbox(feature, [0, 1])
        else:
            min_value = 100 if feature == "Monthly_Bill_Amount" else 0
            max_value = 2000 if feature == "Monthly_Bill_Amount" else 10 if feature == "Loyalty_Duration_Years" else 5 if "Rating" in feature else 65
            user_data[feature] = st.slider(feature, min_value=min_value, max_value=max_value, value=(min_value + max_value) // 2)

    if st.button("Predict"):
        user_df = pd.DataFrame([user_data])
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)
        result = "Yes" if prediction[0] == 1 else "No"
        st.subheader("Prediction Result")
        st.success(f"Will Change Service: {result}")

if __name__ == "__main__":
    main()
