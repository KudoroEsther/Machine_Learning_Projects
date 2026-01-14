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
from typing import Literal
import os

print("All imports successful")


# Load API key
load_dotenv()
api_key = os.getenv("paid_api")

if not api_key:
    raise ValueError("API_Key not found. Please set it in your .env file")
print("API key loaded")

## Initialize LLM
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature=0.5,
    api_key = api_key
)
print(f"LLM initialized: {llm.model_name}")

# Document Collection
file_path = r""

loader = PyPDFDirectoryLoader(file_path)
pages = []

# async for page in loader.alazy_load():
#     pages.append(page)

pages = loader.load()
    
print("Documents loaded.")