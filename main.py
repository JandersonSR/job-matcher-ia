# main.py
import os
import asyncio
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from jobs_runner import process_pending_once

load_dotenv()

CRON_SCHEDULE = os.getenv("CRON_SCHEDULE", "")  # ex: "0 */6 * * *" (cada 6h)
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="Job Matcher Worker")

# agendador global
scheduler = AsyncIOScheduler()
job_running = False  # flag simples para evitar reentrância

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/run")
async def run_now():
    global job_running
    if job_running:
        raise HTTPException(status_code=409, detail="Job em execução")
    job_running = True
    try:
        result = process_pending_once()
        return result
    finally:
        job_running = False

def cron_job_wrapper():
    # Função chamada pelo scheduler (it is sync, APScheduler trata)
    global job_running
    if job_running:
        print("Cron disparado, mas job já está em execução. Pulando.")
        return
    job_running = True
    try:
        r = process_pending_once()
        print("Cron job result:", r)
    except Exception as e:
        print("Erro no cron job:", e)
    finally:
        job_running = False

@app.on_event("startup")
async def startup_event():
    # se CRON_SCHEDULE definido, registra job
    if CRON_SCHEDULE:
        try:
            trigger = CronTrigger.from_crontab(CRON_SCHEDULE)
            scheduler.add_job(cron_job_wrapper, trigger)
            scheduler.start()
            print(f"Scheduler iniciado com CRON: {CRON_SCHEDULE}")
        except Exception as e:
            print("Erro ao registrar CRON:", e)
    else:
        print("CRON_SCHEDULE não definido; agendamento interno desativado.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
