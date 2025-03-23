import streamlit as st
from ai_agents.multi_agent_pipeline import run_agents
from file_processing.pdf_extractor import extract_text_from_pdf

st.title("AI-Powered Risk Assessment")

uploaded_file = st.file_uploader("Upload a document", type=["pdf"])
if uploaded_file:
    document_text = extract_text_from_pdf(uploaded_file)
    result = run_agents(document_text)
    st.write(result)
