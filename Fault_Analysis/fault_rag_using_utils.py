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

print("All imports successful")


from utils_openai import (
    setup_openai_api,
    create_embeddings,
    create_llm,
    create_vectorstore,
    system_prompt_def,
    load_and_chunk_documents
)

# Load API key
api_key = setup_openai_api()
print("API key loaded successfully!")

## Initialize LLM
llm = create_llm(api_key, temperature=0)
print(f"LLM initialized: {llm.model_name}")


# Document Collection
file_path = r""

loader = PyPDFDirectoryLoader(file_path)
pages = []

# async for page in loader.alazy_load():
#     pages.append(page)

pages = loader.load()
    
print("Documents loaded.")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

doc_splits = text_splitter.split_documents(pages)

print(f"Sample chunk: \n{doc_splits[0].page_content[:200]}...")
print("Documents chunked")

chunks = load_and_chunk_documents()

# Vector store and embedding

embeddings = create_embeddings(api_key)
print("Embeddings model initialized")

vectorstore = create_vectorstore(
    embeddings=embeddings
)

chroma_path = "./chroma_db_fault_rag"
vectorstore = Chroma(
    collection_name="agentic_fault_docs",
    persist_directory=chroma_path,
    embedding_function=embeddings
)

#Add documents
vectorstore.add_documents(documents=doc_splits)
print(f"Vector store created with {len(doc_splits)} chunks")
print(f"Persisted to: {chroma_path}")

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
class FaultAgentState(TypedDict):
    fault_label: str
    confidence: float
    retrieved_docs: Optional[str]
    final_answer: Optional[str]


# Defining nodes
# Decide retrieval query
def build_query(state: FaultAgentState):
    fault = state["fault_label"]
    return {
        "query": f"IEEE standard explanation causes mitigation of {fault} in transmission lines"
    }

#Final diagnosis
def generate_answer(state: FaultAgentState):

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

builder = StateGraph(FaultAgentState)

builder.add_node("build_query", build_query)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "build_query")
builder.add_edge("")

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

