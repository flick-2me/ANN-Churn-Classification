import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import tensorflow as tf 

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the preprocessors
with open('labelencoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('OHE_Geography.pkl', 'rb') as f:
    OHE_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Customer Churn Prediction')

# User Input (Removed trailing commas that were converting inputs into tuples)
Geography = st.selectbox('Geography', OHE_encoder.categories_[0])
CreditScore = st.number_input('Credit Score', value=600)
Gender = st.selectbox('Gender', label_encoder.classes_)
Age = st.slider('Age', 18, 92, 40)
Tenure = st.slider('Tenure', 0, 10, 5)
Balance = st.number_input('Balance', value=0.0)
NumOfProducts = st.slider('NumOfProducts', 1, 4, 1)
HasCrCard = st.selectbox('HasCreditCard', [0, 1])
IsActiveMember = st.selectbox('IsActiveMember', [0, 1])
EstimatedSalary = st.number_input('EstimatedSalary', value=50000.0)

# Build Initial DataFrame (Fixed typo: Geography was mapped to CreditScore)
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# One-Hot Encode Geography
geo_encoded = OHE_encoder.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=OHE_encoder.get_feature_names_out(['Geography'])
)

# Label Encode Gender
input_data['Gender'] = label_encoder.transform(input_data['Gender'])

# Combine encoded Geography with input_data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale features
input_scaled = scaler.transform(input_data)

# Predict
pred = model.predict(input_scaled)
prediction_proba = pred[0][0]

st.write(f"**Churn Probability:** {prediction_proba:.2%}")

if prediction_proba > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')