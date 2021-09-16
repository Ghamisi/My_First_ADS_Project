import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import  classification_report, accuracy_score, precision_score, recall_score,f1_score

df = pd.read_csv('diabetes.csv')

st.title('Simple Diabetes Prediction App')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

st.subheader('Training Data')
st.write(df.head())
st.write(df.describe())

st.subheader('Visualisation')
st.bar_chart(df)

X = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

def user_report():
    Glucose = st.sidebar.slider('Glucose level', 0,200,120)
    Age = st.sidebar.slider('Age', 21, 88, 33)
    BMI = st.sidebar.slider('BMI', 0, 67, 20)
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20 )
    Insulin = st.sidebar.slider('Insulin', 0, 846, 79 )
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)

    user_report = {
        'Glucose': Glucose,
        'Age': Age,
        'BMI': BMI,
        'Pregnancies': Pregnancies,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    }
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report()

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, dtc.predict(X_test))*100)+'%')

user_result = dtc.predict(user_data)
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
    output = 'You are not diabetic'
else:
    output = 'You are diabetic'

st.write(output)