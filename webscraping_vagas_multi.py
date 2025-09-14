import requests
from bs4 import BeautifulSoup
import pymongo
import os
from dotenv import load_dotenv
import time

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
vagas_col = db["vagas"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ------------------------------
# Scraping para sites diferentes
# ------------------------------

def scrap_vagas_infojobs(max_pages=3):
    """
    Scraping simplificado do InfoJobs
    """
    base_url = "https://www.infojobs.com.br/vagas-de-emprego"
    vagas = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?Page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Erro ao acessar {url}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        job_cards = soup.find_all("div", class_="vaga")  # exemplo, ajuste conforme site real

        for job in job_cards:
            titulo_tag = job.find("h2")
            empresa_tag = job.find("span", class_="empresa")
            descricao_tag = job.find("p", class_="descricao")

            if not titulo_tag or not empresa_tag:
                continue

            vaga = {
                "titulo": titulo_tag.get_text(strip=True),
                "empresa": empresa_tag.get_text(strip=True),
                "descricao": descricao_tag.get_text(strip=True) if descricao_tag else "",
                "url": url,
                "site": "InfoJobs"
            }
            vagas.append(vaga)

            # Salvar no MongoDB
            vagas_col.update_one(
                {"titulo": vaga["titulo"], "empresa": vaga["empresa"]},
                {"$set": vaga},
                upsert=True
            )

        print(f"[InfoJobs] Página {page} processada, {len(job_cards)} vagas")

        time.sleep(1)  # evita bloqueio por requisições rápidas

    return vagas

def scrap_vagas_catho(max_pages=3):
    """
    Scraping simplificado do Catho
    """
    base_url = "https://www.catho.com.br/vagas"
    vagas = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?pagina={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Erro ao acessar {url}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        job_cards = soup.find_all("div", class_="vaga-card")  # ajuste conforme site real

        for job in job_cards:
            titulo_tag = job.find("h2")
            empresa_tag = job.find("div", class_="empresa")
            descricao_tag = job.find("p")

            if not titulo_tag or not empresa_tag:
                continue

            vaga = {
                "titulo": titulo_tag.get_text(strip=True),
                "empresa": empresa_tag.get_text(strip=True),
                "descricao": descricao_tag.get_text(strip=True) if descricao_tag else "",
                "url": url,
                "site": "Catho"
            }
            vagas.append(vaga)

            # Salvar no MongoDB
            vagas_col.update_one(
                {"titulo": vaga["titulo"], "empresa": vaga["empresa"]},
                {"$set": vaga},
                upsert=True
            )

        print(f"[Catho] Página {page} processada, {len(job_cards)} vagas")
        time.sleep(1)

    return vagas

# ------------------------------
# Função principal
# ------------------------------

def scrap_todos(max_pages=3):
    print("Iniciando scraping de todos os sites...")
    vagas_infojobs = scrap_vagas_infojobs(max_pages)
    vagas_catho = scrap_vagas_catho(max_pages)
    total = len(vagas_infojobs) + len(vagas_catho)
    print(f"Scraping concluído. Total de vagas processadas: {total}")
    return vagas_infojobs + vagas_catho

# ------------------------------
# Executar
# ------------------------------
if __name__ == "__main__":
    scrap_todos(max_pages=3)
