
import streamlit as st
import joblib

model = joblib.load("iris_model.pkl")

st.title("Iris Flower Prediction")
st.write("Enter the flower measurements to predict its species.")

sl = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sw = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
pl = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
pw = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted flower: {flowers[pred[0]]}")
