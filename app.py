import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load preprocessed dataset and pipeline
pipeline = joblib.load("pipe.pkl")  # Ensure you save the trained pipeline
house_data = pd.read_pickle("df.pkl")  # Ensure dataset is saved

# Streamlit UI
st.title("House Price Prediction App")
st.write("Enter the house details to predict the price.")

# User inputs
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, step=1)
mainroad = st.selectbox("Is the house on the main road?", ["yes", "no"])
guestroom = st.selectbox("Does the house have a guestroom?", ["yes", "no"])
basement = st.selectbox("Does the house have a basement?", ["yes", "no"])
hotwaterheating = st.selectbox("Does the house have hot water heating?", ["yes", "no"])
airconditioning = st.selectbox("Does the house have air conditioning?", ["yes", "no"])
parking = st.number_input("Number of Parking Spaces", min_value=0, max_value=5, step=1)
prefarea = st.selectbox("Is the house in a preferred area?", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
square_footage = st.number_input("Square Footage", min_value=500, max_value=10000, step=10)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[bathrooms, mainroad, guestroom, basement, hotwaterheating,
                                airconditioning, parking, prefarea, furnishingstatus, square_footage]],
                              columns=["bathrooms", "mainroad", "guestroom", "basement", "hotwaterheating",
                                       "airconditioning", "parking", "prefarea", "furnishingstatus", "square_footage"])
    
    predicted_price = pipeline.predict(input_data)[0]
    predicted_price = np.expm1(predicted_price)  # Reverse log transformation if applied
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")
