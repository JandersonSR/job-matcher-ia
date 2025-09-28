import os
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

# Conexão com MongoDB
MONGO_URL = os.getenv("MONGO_URL")
mongo_client = MongoClient(MONGO_URL)
db = mongo_client["jobmatcher"]

# Importa a função de jobs (ajuste o nome se diferente)
from jobs import worker_loop

# Cria app FastAPI
app = FastAPI()

# Inicia agendador
scheduler = BackgroundScheduler()
scheduler.add_job(worker_loop, "interval", minutes=5)  # roda a cada 5 minutos
scheduler.start()

# Rota de teste/saúde
@app.get("/health")
def health():
    return {"status": "ok", "message": "Job Matcher IA rodando no Render 🚀"}
