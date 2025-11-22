from fault_Copy import FeatureEngineer

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd

load_dotenv()

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

#prediction endppoint
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

    #scaling input features using the loaded scaler
    prediction = pipeline.predict(features)

    return {"Predicted fault": str(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("host"), port=int(os.getenv("port"))) #converted port to integer because all .env entries are strings

