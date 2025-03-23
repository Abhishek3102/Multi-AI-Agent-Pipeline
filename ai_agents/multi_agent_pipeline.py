from langchain.schema.runnable import RunnableParallel
from ai_agents.health_agent import analyze_health
from ai_agents.finance_agent import analyze_finance

def run_agents(document_text):
    agents = RunnableParallel({
        "health_analysis": analyze_health,
        "finance_analysis": analyze_finance
    })
    return agents.invoke({"document_text": document_text})
