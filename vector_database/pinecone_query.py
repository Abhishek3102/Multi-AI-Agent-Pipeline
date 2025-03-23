import pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX
from langchain.embeddings import HuggingFaceEmbeddings

pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index = pinecone.Index(PINECONE_INDEX)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_analysis(category):
    query_vector = embeddings.embed_query(category)
    response = index.query(vector=query_vector, top_k=1, include_values=True)
    
    if response["matches"]:
        return response["matches"][0]["values"]
    else:
        return "No relevant past analysis found."
