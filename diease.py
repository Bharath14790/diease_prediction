import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


# load saved models (use relative paths)
diabetes = pickle.load(open('diabetes_model.sav', 'rb'))
heart = pickle.load(open('heart_model.sav', 'rb'))


# Sidebar menu
with st.sidebar:
    select = option_menu("Disease Prediction",
                         ["Diabetes Prediction",
                          "Heart Disease Prediction"],
                         icons=['activity','heart'],
                         default_index=0)

# -----------------------------------
# Diabetes Prediction
# -----------------------------------
if select == 'Diabetes Prediction':
    st.title("Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("No of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Level")
    with col1:
        SkinThickness = st.text_input("Skin Thickness Level")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    with col2:
        Age = st.text_input("Age of Person")

    diagnosis = ''

    if st.button("Get Diabetes Result"):
        try:
            # Convert all inputs to float
            input_data = np.array([
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]).reshape(1, -1)

            prediction = diabetes.predict(input_data)
            if prediction[0] == 1:
                diagnosis = 'Diabetes is Positive'
            else:
                diagnosis = 'Diabetes is Negative'

        except ValueError:
            st.error("Please fill in all fields with valid numbers!")

    st.success(diagnosis)

# -----------------------------------
# Heart Disease Prediction
# -----------------------------------
if select == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input("Age of Person")
    with col2:
        Sex = st.text_input("Gender (1=Male, 0=Female)")
    with col3:
        CP = st.text_input("Chest Pain Type")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        Chol = st.text_input("Cholesterol")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar")
    with col1:
        restecg = st.text_input("Resting ECG Result")
    with col2:
        thalach = st.text_input("Maximum Heart Rate")
    with col3:
        exang = st.text_input("Exercise-Induced Angina")
    with col1:
        oldpeak = st.text_input("Oldpeak (ST Depression)")
    with col2:
        slope = st.text_input("Slope")
    with col3:
        ca = st.text_input("Coronary Artery Disease")
    with col1:
        thal = st.text_input("Thalassemia")

    diagnosis = ''

    if st.button("Get Heart Disease Result"):
        try:
            # Convert all inputs to float
            input_data = np.array([
                float(Age), float(Sex), float(CP), float(trestbps),
                float(Chol), float(fbs), float(restecg), float(thalach),
                float(exang), float(oldpeak), float(slope), float(ca), float(thal)
            ]).reshape(1, -1)

            prediction = heart.predict(input_data)
            if prediction[0] == 1:
                diagnosis = 'Heart Disease is Positive'
            else:
                diagnosis = 'Heart Disease is Negative'

        except ValueError:
            st.error("Please fill in all fields with valid numbers!")

    st.success(diagnosis)
