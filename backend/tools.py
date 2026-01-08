import os
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_chroma import Chroma

# --------------------------------------------------
# Base paths (ABSOLUTE, Render-safe)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "phones.csv")
DB_DIR = os.path.join(BASE_DIR, "vectorDB")

# --------------------------------------------------
# Embeddings (CPU-safe)
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Vector DB Builder
# --------------------------------------------------
def build_vector_db():
    print("⚠️ Vector DB not found. Creating new Vector Database...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

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
                "brand": row.get("brand"),
                "model": row.get("model"),
                "variant": row.get("variant"),
                "price": row.get("price"),
                "image": row.get("link") or row.get("image") or "",
            },
        )
        documents.append(doc)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print("✅ Vector Database created successfully")
    return vector_store

# --------------------------------------------------
# Load or Create Vector DB
# --------------------------------------------------
if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    print("✅ Loading existing Vector Database")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
else:
    vector_store = build_vector_db()

# --------------------------------------------------
# Retriever Tool (used by LangGraph)
# --------------------------------------------------
@tool
def retriever(query: str) -> str:
    """
    Retrieve closest matching phone models from the vector DB.
    """
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
