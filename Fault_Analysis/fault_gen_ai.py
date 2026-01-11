from fault_Copy import FeatureEngineer

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
# Loading the saved model
pipeline = joblib.load("detection_pipeline.pkl")


#Initializing the application
app = FastAPI()

#creating the pydantic model
class FaultFeatures(BaseModel):
    Va: float
    Vb: float
    Vc: float
    Ia: float
    Ib: float
    Ic: float

# creating endpoints
@app.get("/")
def welcome():
    return{
        "message": "Welcome to Transmission Line Fault Predictor"
    }


@app.post("/predict")
def predict(line: FaultFeatures):

    features = pd.DataFrame([{
        "Va": line.Va,
        "Vb": line.Vb,
        "Vc": line.Vc,
        "Ia": line.Ia,
        "Ib": line.Ib,
        "Ic": line.Ic
    }])

    prediction = pipeline.predict(features)[0]
    proba = pipeline.predict_proba(features).max()

    if prediction == "No fault":
        return {
            "status": "no_fault",
            "fault": prediction,
            "confidence": round(float(proba), 3)
        }

    return {
        "status": "fault",
        "fault": prediction,
        "confidence": round(float(proba), 3)
    }
