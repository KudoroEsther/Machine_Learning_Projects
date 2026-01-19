# THIS IS WORKING JUST AS A PLAIN LLM NOT RAG

from fault_Copy import FeatureEngineer

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from utils_openai import (
    setup_openai_api,
    create_llm,
    create_embeddings,
    create_llm,
    create_vectorstore,
    system_prompt_def,
    load_and_chunk_documents
)


load_dotenv()
api_key = setup_openai_api()

# Initialize RAG components
embeddings = create_embeddings(api_key)
llm = create_llm(api_key, temperature=0)

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

def build_prompt(fault_label: str, confidence: float):

    fault_info = FAULT_EXPLANATIONS[fault_label]

    return f"""
You are an electrical fault diagnosis assistant.

The ML system has detected the following fault:

Fault label: {fault_label}
Fault name: {fault_info['name']}
Confidence: {confidence * 100:.1f}%

Explain:
1. What this fault means in a power transmission system
2. Common causes
3. Step-by-step practical troubleshooting actions
4. Safety precautions for technicians

Do not guess the fault.
Do not ask for measurements.
"""


@app.post("/diagnose")
def diagnose(line: FaultFeatures):

    # Step 1: ML prediction
    features = pd.DataFrame([line.dict()])
    prediction = pipeline.predict(features)[0]
    proba = pipeline.predict_proba(features).max()

    if prediction == "No fault":
        return {
            "status": "no_fault",
            "confidence": round(float(proba), 3),
            "message": "System operating normally. No fault detected."
        }

    # Step 2: Build GenAI prompt
    prompt = build_prompt(prediction, proba)

    # Step 3: Call LLM (pseudo-code)
    genai_response = llm.invoke(prompt)

    return {
        "status": "fault",
        "fault": prediction,
        "confidence": round(float(proba), 3),
        "diagnosis": genai_response
    }

## ADD LLM AND INVOKE PROMPT