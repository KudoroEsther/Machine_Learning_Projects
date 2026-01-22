"""
Added a post endpoint for the llm, it predicts the fault and sends it to the builder agent
where solutions are provided based on the rag document
"""
from fault_Copy import FeatureEngineer

from fault_rag_using_utils import(
    api_key,
    llm,
    chunks,
    embeddings,
    vectorstore,
    builder_agent
)

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn



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
            "fault_lable": prediction,
            "confidence": round(float(proba), 3)
        }

    return {
        "status": "fault",
        "fault_label": prediction,
        "confidence": round(float(proba), 3)
    }

#GENAI Integration
FAULT_EXPLANATIONS = {
    "LLLG fault": {
        "name": "Three-Phase-to-Ground Fault",
        "description": "All three phases are shorted to ground."
    },
    "LLG fault": {
        "name": "Double Line-to-Ground Fault",
        "description": "Two phases are shorted together and to ground."
    },
    "LG fault": {
        "name": "Single Line-to-Ground Fault",
        "description": "One phase is shorted to ground."
    }
}

@app.post("/diagnose")
def diagnose(line: FaultFeatures):
    features = pd.DataFrame([line.model_dump()])
    prediction = pipeline.predict(features)[0]
    proba = pipeline.predict_proba(features).max()
    
    answer = builder_agent.invoke({
        "fault_label": prediction,
        "confidence": proba,
        "retrieved_docs": "",
        "final_answer": ""
    })

    return answer

    # if prediction == "No fault":
    #     return {
    #         "fault_label": "no_fault",
    #         "confidence": round(float(proba), 3),
    #         "message": "System operating normally. No fault detected."
    #     }