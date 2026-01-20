#THIS IS MEANT TO BE THE AGENTIC RAG (WE CAN USE EXTENDED MESSAGESSTATE INSTEAD OF TYPEDDICT)

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Literal, TypedDict, Optional
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd

print("All imports successful")


from utils_openai import (
    setup_openai_api,
    create_embeddings,
    create_llm,
    create_vectorstore,
    system_prompt_def,
    load_and_chunk_documents,
    load_existing_vectorstore
)

"""
WORKFLOW

Load API Key
initialize llm
load documents
chunk documents (plan to use recursive splitter then semantic splitter)
embed
persist to chromadb
define tools (retriever(plan to use mmr), retrieve documents)
define system prompt
bind tools to llm
nodes: start, build_query_ should_continue, generate_answer, end
define assitant and conditional nodes
build stategraph
define query agent
test agent

"""

""" As of 2026/1/20 Adding the FastAPI part and merging RAG with ML model
"""

# Load API key
api_key = setup_openai_api()
print("API key loaded successfully!")

## Initialize LLM
llm = create_llm(api_key, temperature=0)
print(f"LLM initialized: {llm.model_name}")


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
            "fault_label": "no_fault",
            "fault": prediction,
            "confidence": round(float(proba), 3)
        }

    return {
        "fault_label": "fault",
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

# Document Collection
# the sizes are tentative
data_path = r"C:\Users\ncc333\Desktop\ML_projects\Fault_Analysis\Fault_docs"

chunks = load_and_chunk_documents(
    data_path= data_path,
    chunk_size=1000,
    chunk_overlap=100)
print("Documents loaded and chunked")


# Vector store and embedding
embeddings = create_embeddings(api_key)
print("Embeddings model initialized")
path_dir = ("./chroma_db_fault_rag")

if os.path.exists(path_dir):
    vectorstore = load_existing_vectorstore(embeddings=embeddings)

else:
    vectorstore = create_vectorstore(
    embeddings=embeddings
    )

#Retriever toool
@tool
def retrieve_documents(query: str) -> str:
    """
    Search for relevant documents in the knowledge base.
    
    Use this tool when you need information from the document collection
    to answer the user's question. Do NOT use this for:
    - General knowledge questions
    - Greetings or small talk
    - Simple calculations
    
    Args:
        query: The search query describing what information is needed
        
    Returns:
        Relevant document excerpts that can help answer the question
    """
    # Using MMR for diverse results
    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":5, "fetch_k":10}
    )

    results = retriever.invoke(query)
    if not results:
        return "No relevant documents found"
    
    formatted = "\n\n---\n\n".join(
        f"Document {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(results)
    )
    return formatted

print("Retrieval tool created")

#SYSTEM PROMPT
system_prompt = system_prompt_def()
print(" System prompt configured")

# Defining Stategraph and using TypedDict instead of MessagesState
class FaultAgentState(MessagesState):
    fault_label: str
    confidence: float
    retrieved_docs: Optional[str]
    final_answer: Optional[str]


# Bind tool to LLM
tools = [retrieve_documents]
llm_with_tools = llm.bind_tools(tools)

def assistant(state: FaultAgentState) -> dict:
    """
    Assistant node - decides whether to retrieve or answer directly.
    """
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Defining nodes
# Decide retrieval query
def build_query(state: FaultAgentState):
    fault = state["fault_label"]
    return {
        "query": f"IEEE standard explanation causes mitigation of {fault} in transmission lines"
    }

#Final diagnosis
def finalizer(state: FaultAgentState):
    prompt = f"""
Fault detected by ML system:
Fault: {state['fault_label']}
Confidence: {state['confidence'] * 100:.1f}%

IEEE / Protection Manual Context:
{state['retrieved_docs']}

Provide:
1. Explanation of the fault
2. Common causes
3. Step-by-step resolution
4. Safety precautions
"""

    response = llm.invoke(prompt)
    return {"final_answer": response}

def should_continue(state: FaultAgentState) -> Literal["tools", "finalizer"]:
    """
    Decide whether to call tools or finish.
    """
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    return "__end__"
print("Agent nodes defined")

builder = StateGraph(FaultAgentState)

builder.add_node("build_query", build_query)
builder.add_node("finalizer", finalizer)
builder.add_node("assistant", assistant)


builder.add_edge(START, "build_query")
builder.add_edge("build_query", "assistant")
builder.add_conditional_edges(
    "assistant", 
    should_continue,
    {"tools": "tools", "finalizer": "finalizer"})
builder.add_edge("finalizer", END)

builder_agent = builder.compile()

answer = builder_agent.invoke({
    "fault_label": "",
    "confidence": "",
    "retrieved_docs": "",
    "final_answer": ""
})

# # Bind tool to LLM
# tools = [retrieve_documents]
# llm_with_tools = llm.bind_tools(tools)

# def assistant(state: MessagesState) -> dict:
#     """
#     Assistant node - decides whether to retrieve or answer directly.
#     """
#     messages = [system_prompt] + state["messages"]
#     response = llm_with_tools.invoke(messages)
#     return {"messages": [response]}

# def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
#     """
#     Decide whether to call tools or finish.
#     """
#     last_message = state["messages"][-1]

#     if last_message.tool_calls:
#         return "tools"
#     return "__end__"
# print("Agent nodes defined")

