import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load model
@st.cache
def load_model():
    model = RandomForestClassifier()  # Replace with your best model
    model.fit(X_train, y_train)
    return model

# Streamlit app
st.title("Heart Disease Prediction")
st.write("Input the patient data to predict the likelihood of heart disease.")

# User input fields
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", [0, 1])  # Replace with dataset values
cholesterol = st.number_input("Cholesterol Level", 100, 400, 200)

# Prediction
model = load_model()
input_data = pd.DataFrame([[age, sex, cholesterol]], columns=["age", "sex", "cholesterol"])
input_data = scaler.transform(input_data)  # Scale input
prediction = model.predict(input_data)

st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
