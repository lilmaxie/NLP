import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#load model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
    
with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)
    

# streamlit app
st.title("Customer Churn Prediction")

# user input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# one-hot encode geography
geo_encoded = one_hot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# combine one-hot encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale
input_data_scaled = scaler.transform(input_data)

# predict
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]
st.write(f"Prediction Probability: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("Prediction: Customer will churn")
else:
    st.write("Prediction: Customer will not churn")