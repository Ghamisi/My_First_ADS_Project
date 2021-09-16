import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier

st.write("""
# Diabetes Prediction App

This app predicts the **Diabetes Status** of an Individual
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file. Open the repository and choose "diabetes.csv".](https://github.com/Ghamisi/My_First_ADS_Project/)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Glucose = st.sidebar.slider('Glucose level', 0,200,120)
        Age = st.sidebar.slider('Age', 21, 88, 33)
        BMI = st.sidebar.slider('BMI', 0, 67, 20)
        Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
        BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
        SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20 )
        Insulin = st.sidebar.slider('Insulin', 0, 846, 79 )
        DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
        data = {'Glucose': Glucose,
                'Age': Age,
                'BMI': BMI,
                'Pregnancies': Pregnancies,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction}

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
diabetes_raw = pd.read_csv('diabetes.csv')
diabetes = diabetes_raw.drop(columns=['Outcome'])
df = pd.concat([input_df, diabetes],axis=0)

# Selects only the first row (the user input data)
df = df[:1]

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

load_dtc = pickle.load(open('diabetes_dtc.pkl', 'rb'))

# Apply model to make predictions
prediction = load_dtc.predict(df)
prediction_proba = load_dtc.predict_proba(df)

st.subheader('Prediction')
Diabetes_Status = np.array(['0','1'])
st.write(Diabetes_Status[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)