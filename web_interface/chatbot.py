import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from config import PINECONE_API_KEY, PINECONE_INDEX

llm = ChatGroq(model_name="mixtral-8x7b-32768")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)

memory = ConversationBufferMemory(llm=llm)

chatbot_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever(), memory=memory)

st.title("💬 AI Chat Assistant")

user_query = st.text_input("Ask me anything about your past analyses:")
if user_query:
    response = chatbot_chain.run(user_query)
    st.write("🤖 AI:", response)
