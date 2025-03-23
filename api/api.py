from fastapi import FastAPI
from ai_agents.multi_agent_pipeline import run_agents

app = FastAPI()

@app.post("/analyze/")
async def analyze_text(document_text: str):
    return run_agents(document_text)
