from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import joblib
from pydantic import BaseModel

model = tf.keras.models.load_model("diabetes_model.h5")
scaler = joblib.load("scaler,pkl")

app = FastAPI()

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    Blood_Pressure: float
    Skin_Thickness: float
    Insulin: float
    BMI: float
    Diabetes_Pedigree_Function: float
    Age: float

@app.get("/")
def home():
    return {"Message": "Diabetes Prediction"}

@app.post("/predict")
def predict_diabetes(data: PatientData):
    input_data = np.array([[data.Pregnancies, data.Glucose,
                            data.Blood_Pressure, data.Skin_Thickness,
                            data.Insulin, data.BMI,
                            data.Diabetes_Pedigree_Function,
                            data.Age]])

    input_data - scaler.transform(input_data)

    prediction = model.predict(input_data)[0][0]
    result = "Diabetic" if prediction > 0.5 else "Not Diabetic"

    return {"prediction": result, "confidence": f"{prediction:0.2f}"}
