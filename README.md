# job-matcher-ia

### Para iniciar o projeto

> python -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt

### Rodar o projeto
> uvicorn main:app --reload

### Teste de endpoints
#### health
curl http://localhost:8000/health

#### disparo manual
curl -X POST http://localhost:8000/run
