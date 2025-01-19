import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
@st.cache_resource  # Caches the model to speed up loading
def load_model():
    return tf.keras.models.load_model("nns_regressor.keras")

model = load_model()

# App Title
st.title("Forest Fire Area Prediction")
st.write("Enter temp and RH feature values to predict the fire-affected area.")

# Input fields for features
temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
RH = st.number_input("Relative Humidity (%)", min_value=0.0, step=0.1)

# Predict Button
if st.button("Predict"):
    # Prepare input data
    features = np.array([[temp, RH]])
    prediction = model.predict(features)[0][0]
    # Display the prediction
    st.subheader(f"Predicted Fire Area: {prediction:.2f} ha")
    




