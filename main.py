import numpy as np
import pickle 
import streamlit as st

model = pickle.load(open("svm_trained_model.sav", "rb"))
st.title('ML Project')

with st.expander('Diabetes Prediction Web App'):
     st.write("Building a web app to predict Diabetes")

def diabetes_prediction(data):
     data_np = np.asarray(data)
     data_reshaped  = data_np.reshape(1,-1)
     prediction = model.predict(data_reshaped)
     print(prediction)

     if (prediction[0] == 0):
          return "The person is not diabetic"
     else:
          return "The person is diabetic"

def main():
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diagnosis = ""

    if st.button('Diabetes Test Result'):
         diagnosis = diabetes_prediction([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]) 

    st.write(diagnosis)   

if __name__ == '__main__':
     main()