import streamlit as st
import joblib  
import numpy as np

regression_model = joblib.load('regression_model.pkl')
classification_model = joblib.load('classification_model.pkl')

st.title("Copper Industry Prediction App")
st.sidebar.header("Input Parameters")

selling_price = st.sidebar.number_input('Selling Price (Enter 0 if unknown)', min_value=0.0, format="%.2f")
feature_1 = st.sidebar.number_input('Feature 1 (e.g., Quantity)', min_value=0.0, format="%.2f")
feature_2 = st.sidebar.number_input('Feature 2 (e.g., Quality Index)', min_value=0.0, format="%.2f")
feature_3 = st.sidebar.number_input('Feature 3 (e.g., Market Demand Index)', min_value=0.0, format="%.2f")

if st.sidebar.button("Predict Selling Price"):
    features = np.array([[feature_1, feature_2, feature_3]])  # Adjust to match your model's input structure
    predicted_price = regression_model.predict(features)[0]
    st.write(f"Predicted Selling Price: ${predicted_price:.2f}")

if st.sidebar.button("Predict Lead Status (WON/LOST)"):
    features = np.array([[feature_1, feature_2, feature_3]])  # Adjust to match your model's input structure
    predicted_status = classification_model.predict(features)[0]
    status_label = "WON" if predicted_status == 1 else "LOST"
    st.write(f"Predicted Lead Status: {status_label}")

st.markdown("### About")
st.markdown("This app predicts the selling price and lead status (WON/LOST) based on input parameters using machine learning.")
