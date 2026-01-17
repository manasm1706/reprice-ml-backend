import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_chroma import Chroma

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "phones.csv")
DB_DIR = os.path.join(BASE_DIR, "vectorDB")

# Embeddings (CPU-safe)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = None

def build_vector_db():
    """Build vector database from CSV"""
    print("⚠️ Building new Vector Database...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"📊 Processing {len(df)} phones...")

    documents = []
    for _, row in df.iterrows():
        text = (
            f"Model: {row.get('brand','')} "
            f"{row.get('model','')} "
            f"{row.get('variant','')} | "
            f"Price: {row.get('price','')}"
        ).strip()

        doc = Document(
            page_content=text,
            metadata={
                "brand": str(row.get("brand", "")),
                "model": str(row.get("model", "")),
                "variant": str(row.get("variant", "")),
                "price": str(row.get("price", "")),
                "image": str(row.get("link", "") or row.get("image", "")),
            },
        )
        documents.append(doc)

    # Create vector store with explicit settings
    os.makedirs(DB_DIR, exist_ok=True)
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="phone_collection"
    )

    print("✅ Vector Database created successfully")
    return vector_store

def init_vector_store():
    """Initialize or load vector store"""
    global vector_store

    if vector_store is not None:
        print("✅ Vector store already loaded")
        return

    try:
        # Try loading existing DB
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            print("🔄 Loading existing Vector DB...")
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=DB_DIR,
                collection_name="phone_collection"
            )
            print("✅ Existing Vector DB loaded")
        else:
            # Build new DB
            vector_store = build_vector_db()
            
    except Exception as e:
        print(f"❌ Vector DB initialization error: {e}")
        # Try rebuilding if loading failed
        if os.path.exists(DB_DIR):
            print("🔄 Attempting to rebuild Vector DB...")
            vector_store = build_vector_db()
        else:
            raise

@tool
def retriever(query: str) -> str:
    """
    Retrieve closest matching phone models from the vector DB.
    """
    if vector_store is None:
        return "Error: Vector database not initialized"
    
    try:
        results = vector_store.similarity_search(query, k=5)

        if not results:
            return "No matching Phone Found"

        output = "Found the following similar models:\n"
        for i, doc in enumerate(results):
            output += (
                f"{i+1}. Model: {doc.metadata.get('brand')} "
                f"{doc.metadata.get('model')} "
                f"{doc.metadata.get('variant')} | "
                f"Price: {doc.metadata.get('price')}\n"
            )

        return output
        
    except Exception as e:
        return f"Error during retrieval: {str(e)}"