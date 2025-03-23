from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

model = ChatGroq(model_name="mixtral-8x7b-32768")

finance_prompt = PromptTemplate(
    template="Analyze this financial report and provide investment risks:\n{document_text}",
    input_variables=["document_text"]
)

def analyze_finance(document_text):
    response = model.invoke(finance_prompt.format(document_text=document_text))
    return response["content"]
