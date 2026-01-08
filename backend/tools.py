import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_chroma import Chroma
import os

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_Dir = "./vectorDB"

def build_vector_DB():
    print("Creating new Vector Database")
    # Prefer all_phones_2 which includes image links
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_Dir = os.path.join(BASE_DIR, "vectorDB")



    

    df = pd.read_csv(csv_path)

    documents = []

    for _, row in df.iterrows():
        text = (
            f"Model: {row.get('brand','')} "
            f"{row.get('model','')} "
            f"{row.get('variant','')} | "
            f"Price: {row.get('price', '')} "
        ).strip()

        doc = Document(
            page_content=text,
            metadata={
                "brand" : row.get('brand'),
                "model": row.get('model'),
                "variant": row.get('variant'),
                "price": row.get('price'),
                # include image/link so retriever can show images if needed
                "image": row.get('link') or row.get('image') or ''
            }  
        )

        documents.append(doc)

    print(documents[0])


    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_Dir
    )
    print("✅ Vector Database Created Successfully.")
    
    return vector_store
    

if os.path.exists(DB_Dir) and os.listdir(DB_Dir):
    print("✅ Loading Existing Vector Database")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_Dir
    )
else:
    print("⚠️ Database not found. Creating new Vector Database from CSV...")
    vector_store = build_vector_DB()
    

@tool
def retriever(query: str) -> str:
    """
    Retrieves the closest matching phone from the vector database based on the query.
    NOTE: The result is the closest match, which may not be exactly relevant.
    The LLM should decide if the returned phone matches the user intent.
    """
    results = vector_store.similarity_search(query, k=5)

    if not results:
        return "No matching Phone Found"
    
    output_text = "Found the following similar models:\n"
    for i, doc in enumerate(results):
        mobile_brand = doc.metadata.get("brand", "Unknown Brand")
        mobile_model = doc.metadata.get("model", "Unknown Model")
        mobile_variant = doc.metadata.get("variant", "Unknown Variant")
        mobile_price = doc.metadata.get("price", "0")
        output_text += f"{i+1}. Model: {mobile_brand} {mobile_model} {mobile_variant} | Price: {mobile_price}\n"


    print(output_text)
    return output_text

if __name__ == "__main__":
    print(retriever.invoke("Galaxy F15"))