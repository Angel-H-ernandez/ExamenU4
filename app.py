import uvicorn
from fastapi import FastAPI
from Notes import Notes
import numpy as np
import pickle
import pandas as np

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classfier = pickle.load(pickle_in)

@app.get("/")
def index():
    return{"mensaje":"Hola, bienvenido al modelo"}

@app.get("/Bienvenido")
def fun_nombre(name:str):
    return{"hola bienvenido ": f'{name}'}

@app.post("/predict")
def predict_banknote(data:Notes):
    data = data.dict()
    Pregnancies = data["Pregnancies"]
    Glucose = data["Glucose"]
    BloodPressure = data["BloodPressure"]
    SkinThickness = data["SkinThickness"]
    Insulin = data["Insulin"]
    BMI = data["BMI"]
    DiabetesPedigreeFunction = data["DiabetesPedigreeFunction"]
    Age = data["Age"]

    prediction = classfier.predict([[Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction, Age]])

    if(prediction[0]==0):
        prediction = "No tiene diabetes"
    else:
        prediction = "Tiene diabetes"
    return{"prediction":prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)