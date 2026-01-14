import os
from dotenv import load_dotenv
from typing import Tuple, List, Dict
from pathlib import Path
from typing import List
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# API SETUP
# ============================================

def setup_openai_api() -> str:
    """
    Load OpenAI API key from environment
    """
    load_dotenv()
    api_key = os.getenv("paid_api")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Please set it in your .env file or environment variables."
        )

    return api_key

# ===========================================
# DATA LOADING
# ===========================================

def load_and_chunk_documents(
    data_path: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[Document]:
    """Load documents from directory, chunk them, and prepare for ChromaDB."""
    
    data_dir = Path(data_path)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Invalid directory: {data_path}")
    
    # Define loaders for each file type
    loaders = {
        '.txt': lambda p: TextLoader(str(p), encoding='utf-8'),
        '.pdf': lambda p: PyPDFLoader(str(p))
    }
    
    # Load all documents
    documents = []
    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue
            
        loader_fn = loaders.get(file_path.suffix.lower())
        if not loader_fn:
            continue
        
        try:
            documents.extend(loader_fn(file_path).load())
            logger.info(f"Loaded: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
    
    if not documents:
        raise ValueError(f"No documents loaded from {data_path}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add ChromaDB-friendly metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_id': f"chunk_{idx}",
            'chunk_index': idx,
            'source_file': Path(chunk.metadata.get('source', 'unknown')).name
        })
        # Ensure all metadata values are strings, ints, floats, or bools
        chunk.metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                         for k, v in chunk.metadata.items()}
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


# ==============================================
# MODEL INITIALIZATION
# ==============================================

def create_embeddings(
    api_key: str,
    model: str = "text-embedding-3-small"
) -> OpenAIEmbeddings:
    """
    Initialize OpenAI embeddings model
    """
    embeddings = OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key
    )
    print(f"[OK] Initialized embeddings: {model}")
    return embeddings


def create_llm(
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> ChatOpenAI:
    """
    Initialize OpenAI chat model
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key
    )
    print(f"[OK] Initialized LLM: {model} (temp={temperature})")
    return llm

# ==============================================
# VECTOR STORE
# ==============================================
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

def create_vectorstore(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    collection_name: str = "tax_collection",
    persist_directory: str = "./chroma_db_fault_rag"
):
    """
    Create and populate vector store. 
    """
    # Force SQLite backend
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(allow_reset=True)
    )

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    
    # Add documents
    # Extract texts, metadatas, and ids from Document chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.metadata['chunk_id'] for chunk in chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"[OK] Created Chroma vector store: {collection_name}")
    return vectorstore

def load_existing_vectorstore(
    embeddings: OpenAIEmbeddings,
    collection_name: str = "agentic_fault_docs",
    persist_directory: str = "./chroma_db_fault_rag"
):
    """
    Load existing vector store.
    """
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(allow_reset=True)
    )
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    print(f"[OK] Loaded existing Chroma vector store: {collection_name}")
    return vectorstore


# Example usage pipeline
if __name__ == "__main__":
    # Load and chunk documents
    chunks = load_and_chunk_documents(r"C:\RAG\Project\documents")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store
    vectorstore = create_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        collection_name="msme"
    )
    
    # Or load existing
    # vectorstore = load_existing_vectorstore(embeddings=embeddings)
# =================================================
# PROMPTS
# =================================================

def system_prompt_def():
    system_prompt = SystemMessage(content="""You are PowerBot, a helpfulassistant with access to a document retrieval tool.

RETRIEVAL DECISION RULES:

DO NOT retrieve for:
- Greetings: "Hello", "Hi", "How are you"
- Questions about your capabilities: "What can you help with?", "What do you do?"
- Simple math or general knowledge: "What is 2+2?"
- Casual conversation: "Thank you", "Goodbye"

DO retrieve for:
- Questions asking for specific information that would be in documents
- Requests for facts, definitions, or explanations about specialized topics
- Any question where citing sources would improve the answer

Rule of thumb: If the user is asking for information (not just chatting), retrieve first.

When you retrieve documents, cite them in your answer. If documents don't contain the answer, say so.
""")

    return system_prompt

    


# =============================================================================
# UTILITIES
# =============================================================================

def print_retrieval_results(docs: List, max_docs: int = 3, max_chars: int = 200):
    """
    Pretty print retrieved documents
    """
    print(f"\n{'='*80}")
    print(f"Retrieved {len(docs)} documents:")
    print(f"{'='*80}\n")

    for i, doc in enumerate(docs[:max_docs], 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}

        print(f"Document {i}:")
        print(f"Title: {metadata.get('doc_title', 'N/A')}")
        print(f"Content: {content[:max_chars]}...")
        print(f"{'-'*80}\n")


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (rough estimate)
    """
    # Rough approximation: 1 token ≈ 4 characters
    # For more accuracy, use tiktoken:
    # import tiktoken
    # encoding = tiktoken.encoding_for_model("gpt-4")
    # return len(encoding.encode(text))
    return len(text) // 4


def calculate_token_reduction(before: int, after: int) -> float:
    """
    Calculate percentage token reduction
    """
    if before == 0:
        return 0
    return ((before - after) / before) * 100


def format_docs(docs: List) -> str:
    """
    Format retrieved documents into a single string for context
    """
    return "\n\n".join([
        doc.page_content if hasattr(doc, 'page_content') else str(doc)
        for doc in docs
    ])


# ==============================================
#TOOLS
# ==============================================
