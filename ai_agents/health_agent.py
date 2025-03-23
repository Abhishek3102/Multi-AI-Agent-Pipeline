from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

model = ChatGroq(model_name="mixtral-8x7b-32768")

health_prompt = PromptTemplate(
    template="Analyze this medical report and provide risk assessment:\n{document_text}",
    input_variables=["document_text"]
)

def analyze_health(document_text):
    response = model.invoke(health_prompt.format(document_text=document_text))
    return response["content"]
