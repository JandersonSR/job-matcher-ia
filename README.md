# job-matcher-ia

### Para iniciar o projeto

> python -m venv .venv
> source .venv/bin/activate ou source .venv/Scripts/activate
> pip install -r requirements.txt

### Rodar o projeto
> uvicorn main:app --reload

### Se der erro rode no bash:
python -m venv .venv
source .venv/bin/activate   # Linux/Mac // Ou source .venv/Scripts/activate

### ou no Windows PowerShell:
> .venv\Scripts\Activate.ps1

E rode novamente o comando:
> uvicorn main:app --reload
ou
> python -m uvicorn main:app --reload

### Teste de endpoints
#### health
curl http://localhost:8000/health

#### disparo manual
curl -X POST http://localhost:8000/run
