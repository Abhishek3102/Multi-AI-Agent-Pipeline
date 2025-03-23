import pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX
from langchain.embeddings import HuggingFaceEmbeddings

pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index = pinecone.Index(PINECONE_INDEX)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store_analysis(text, category):
    vector = embeddings.embed_query(text)  # Convert text to vector
    index.upsert(vectors=[{"id": category, "values": vector}])
    print(f"Stored analysis under category '{category}'.")
