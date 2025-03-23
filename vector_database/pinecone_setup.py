import pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")

if PINECONE_INDEX not in pinecone.list_indexes():
    pinecone.create_index(name=PINECONE_INDEX, dimension=384, metric="cosine")

print(f"Pinecone index '{PINECONE_INDEX}' is set up.")
