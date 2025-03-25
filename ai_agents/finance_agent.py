from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

model = ChatGroq(model_name="mixtral-8x7b-32768")

finance_prompt = PromptTemplate(
    template="Analyze this financial report and provide investment risks:\n{document_text}",
    input_variables=["document_text"]
)


def analyze_finance(document_text):
    try:
        response = model.invoke(finance_prompt.format(document_text=document_text))
        if not response or "content" not in response:
            raise ValueError("Invalid response format from LLM")
        return response["content"]
    
    except Exception as e:
        print(f"❌ Error in analyze_finance: {e}")
        return "⚠️ Error: Unable to analyze the financial report at this time. Please try again later."