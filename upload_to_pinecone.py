import os
import pandas as pd
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "reprice-index")
PINECONE_EMBED_MODEL = "llama-text-embed-v2"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

df = pd.read_csv("data/phones.csv")

texts = (
    df["brand"] + " " +
    df["model"] + " " +
    df["variant"]
).tolist()

metadatas = df.to_dict(orient="records")

BATCH_SIZE = 96

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_meta = metadatas[i:i+BATCH_SIZE]

    embeddings = pc.inference.embed(
        model=PINECONE_EMBED_MODEL,
        inputs=batch_texts,
        parameters={"input_type": "passage"}   # 🔥 THIS IS THE KEY
    )

    vectors = [
        {
            "id": str(i + j),
            "values": emb["values"],
            "metadata": batch_meta[j]
        }
        for j, emb in enumerate(embeddings)
    ]

    index.upsert(vectors)
    print(f"Uploaded {i + len(batch_texts)} / {len(texts)}")

print("✅ Uploaded to Pinecone successfully")
