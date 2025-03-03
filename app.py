import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
import datetime
import catboost
import sklearn

# Load trained model, PCA, and StandardScaler
with open("model_pickup.pkl", "rb") as f:
    model = pickle.load(f)

with open("pca_pickup.pkl", "rb") as f:
    pca = pickle.load(f)

with open("scaler_pickup.pkl", "rb") as f:
    scaler = pickle.load(f)

# Features to scale
features_to_scale = ['pickup_distance', 'log_pickup_distance', 'accept_hour_of_day',
                     'accept_day_of_week', 'pickup_hour', 'pickup_day_of_week']

# Streamlit App
st.title("Pickup ETA Prediction")

# User Inputs
region_id = st.number_input("Region ID", value=1)
lng = st.number_input("Longitude", value=0.0)
lat = st.number_input("Latitude", value=0.0)
aoi_id = st.number_input("AOI ID", value=1)
aoi_type = st.number_input("AOI Type", value=1)
accept_gps_lng = st.number_input("Accept GPS Longitude", value=0.0)
accept_gps_lat = st.number_input("Accept GPS Latitude", value=0.0)
pickup_gps_lng = st.number_input("Pickup GPS Longitude", value=0.0)
pickup_gps_lat = st.number_input("Pickup GPS Latitude", value=0.0)

# Accept & Delivery Times
accept_time = st.time_input("Accept Time", datetime.time(12, 0))
pickup_time = st.time_input("Pickup Time", datetime.time(12, 30))

# Feature Engineering
accept_dt = datetime.datetime.combine(datetime.date.today(), accept_time)
pickup_dt = datetime.datetime.combine(datetime.date.today(), pickup_time)

# Compute Distance and Log Distance
pickup_distance = math.sqrt((pickup_gps_lng - accept_gps_lng) ** 2 + (pickup_gps_lat - accept_gps_lat) ** 2)
log_pickup_distance = np.log1p(pickup_distance)

# Extract Date Features
accept_hour_of_day = accept_dt.hour
accept_day_of_week = accept_dt.weekday()
pickup_hour = pickup_dt.hour
pickup_day_of_week = pickup_dt.weekday()

# Create DataFrame
input_data = pd.DataFrame({
    "region_id": [region_id],
    "lng": [lng],
    "lat": [lat],
    "aoi_id": [aoi_id],
    "aoi_type": [aoi_type],
    "pickup_gps_lng": [pickup_gps_lng],
    "pickup_gps_lat": [pickup_gps_lat],
    "accept_gps_lng": [accept_gps_lng],
    "accept_gps_lat": [accept_gps_lat],
    "accept_hour_of_day": [accept_hour_of_day],
    "accept_day_of_week": [accept_day_of_week],
    "pickup_hour": [pickup_hour],
    "pickup_day_of_week": [pickup_day_of_week],
    "pickup_distance": [pickup_distance],
    "log_pickup_distance": [log_pickup_distance]
})

# Scale selected features
input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])

# Apply PCA
pca_transformed = pca.transform(input_data)

# Combine original & PCA-transformed features
final_input = np.hstack((input_data, pca_transformed))

# Predict Button
if st.button("Predict ETA"):
    prediction = model.predict(final_input)
    st.success(f"ETA: {prediction[0]:.2f} minutes")
