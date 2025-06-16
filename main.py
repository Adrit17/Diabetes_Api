from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

#Loading model & scaler
model = tf.keras.models.load_model("diabetes_model.keras")
scaler = joblib.load("scaler.pkl")

THRESHOLD = 0.487  #0.487 maximizes the F1 score

app = FastAPI()

class PatientData(BaseModel):
    Age: float
    Gender: int
    BMI: float
    SBP: float
    DBP: float
    FPG: float
    Chol: float
    Tri: float
    HDL: float
    LDL: float
    ALT: float
    BUN: float
    CCR: float
    FFPG: float
    smoking: int
    drinking: int
    family_history: int

def convert_mgdl_to_mmol(feature_dict):
    glucose_factor = 18
    cholesterol_factor = 38.67
    triglycerides_factor = 88.57

    #Converting GlucosE(FPG, FFPG)
    feature_dict["FPG"] /= glucose_factor
    feature_dict["FFPG"] /= glucose_factor

    #Converting CholesteroL(Chol, HDL, LDL)
    feature_dict["Chol"] /= cholesterol_factor
    feature_dict["HDL"] /= cholesterol_factor
    feature_dict["LDL"] /= cholesterol_factor

    #Converting Triglycerides(Tri)
    feature_dict["Tri"] /= triglycerides_factor

    return feature_dict

@app.get("/")
def read_root():
    return {"message": "Welcome to Diabetes Prediction API"}

@app.post("/predict")
def predict_diabetes(data: PatientData):
    try:
        data_dict = data.dict()
        #Automatically converting mg/dL units to mmol/L 
        data_dict = convert_mgdl_to_mmol(data_dict)

        features = np.array([[data_dict["Age"], data_dict["Gender"], data_dict["BMI"], data_dict["SBP"], data_dict["DBP"],
                              data_dict["FPG"], data_dict["Chol"], data_dict["Tri"], data_dict["HDL"], data_dict["LDL"],
                              data_dict["ALT"], data_dict["BUN"], data_dict["CCR"], data_dict["FFPG"],
                              data_dict["smoking"], data_dict["drinking"], data_dict["family_history"]]])

        scaled = scaler.transform(features)
        prob = model.predict(scaled)[0][0]
        prediction = "Diabetic" if prob >= THRESHOLD else "Not Diabetic"

        return {
            "prediction": prediction,
            "confidence": f"{prob:.2f}",
            "threshold": THRESHOLD,
            "raw_probability": f"{prob:.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
