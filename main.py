# main.py
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Job Matcher - Render Free")

# Rotas principais expostas para serem chamadas por um cron/servidor externo
@app.get("/health")
def health():
    return {"status": "ok", "message": "Job Matcher IA rodando (Render Free) 🚀"}

@app.get("/scrap-vagas")
def scrap_vagas(max_pages: int = 3):
    """
    Dispara o scraping de vagas (chamado por servidor externo via HTTP).
    Query param max_pages controla quantas páginas cada site deve tentar.
    """
    try:
        from webscrapping_vagas_multi import scrap_todos
        vagas = scrap_todos(max_pages=max_pages)
        return {"status": "ok", "total_vagas_processadas": len(vagas)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no scraping: {e}")

@app.get("/processar-curriculos")
def processar_curriculos():
    """
    Dispara o processamento de um currículo pendente (um por vez).
    Chamado por servidor externo via HTTP.
    """
    try:
        from jobs import worker_run_once
        result = worker_run_once()
        return {"status": "ok", "detalhe": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar currículos: {e}")
