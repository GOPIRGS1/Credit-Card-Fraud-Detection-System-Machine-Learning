import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details:")

# --- Input fields ---
amt = st.number_input("Transaction Amount", value=50.0)
merchant = st.text_input("Merchant", value="Merchant_A")
category = st.text_input("Category", value="food_dining")
gender = st.selectbox("Gender", ["M", "F"])
cc_num = st.text_input("Credit Card Number", value="1234567890123456")
lat = st.number_input("Latitude", value=0.0)
long = st.number_input("Longitude", value=0.0)
city_pop = st.number_input("City Population", value=1000)
job = st.text_input("Job", value="Engineer")
dob = st.date_input("Date of Birth")
merch_lat = st.number_input("Merchant Latitude", value=0.0)
merch_long = st.number_input("Merchant Longitude", value=0.0)
first = st.text_input("First Name", value="John")
last = st.text_input("Last Name", value="Doe")
street = st.text_input("Street", value="123 Main St")

if st.button("Predict"):

    # --- Create DataFrame ---
    input_df = pd.DataFrame({
        'amt':[amt],
        'cc_num':[hash(str(cc_num)) % (10**4)],
        'merchant':[merchant],
        'category':[category],
        'gender':[gender],
        'lat':[lat],
        'long':[long],
        'city_pop':[city_pop],
        'job':[job],
        'merch_lat':[merch_lat],
        'merch_long':[merch_long],
        'first':[first],
        'last':[last],
        'street':[street],
        'age':[ (pd.to_datetime('today') - pd.to_datetime(dob)).days // 365 ]
    })

    # --- Encode categorical columns ---
    for col in ['merchant','category','gender','job','first','last','street']:
        le = encoder[col]
        input_df[col] = input_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # --- Optional features ---
    input_df['amt_log'] = np.log1p(input_df['amt'])
    input_df['distance_km'] = np.sqrt((input_df['lat'] - input_df['merch_lat'])**2 + (input_df['long'] - input_df['merch_long'])**2)

    # --- Align with model features ---
    input_df = input_df[model.feature_name_]

    # --- Predict ---
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Prediction: Fraudulent (Probability: {proba:.2f})")
    else:
        st.success(f"Prediction: Legitimate (Probability: {proba:.2f})")
