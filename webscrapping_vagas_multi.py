import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pymongo

load_dotenv()

# ------------------------------
# MongoDB
# ------------------------------
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")

client = pymongo.MongoClient(MONGO_URL)
db = client[DB_NAME]
vagas_col = db["vagas"]

HEADERS = {
    "User-Agent": os.getenv("USER_AGENT",
                             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100 Safari/537.36")
}

# ------------------------------
# Funções auxiliares
# ------------------------------
def make_uid(titulo: str, empresa: str, site: str, url: str) -> str:
    key = f"{titulo}|{empresa}|{site}|{url}"
    return hashlib.sha1(key.encode()).hexdigest()

def _safe_get_text(tag):
    return tag.get_text(strip=True) if tag else ""


def scrap_vagascom(term="desenvolvedor", max_pages=3):
    print("[VAGAS.COM] Iniciando scraping...")

    vagas = []
    base_url = "https://www.vagas.com.br/vagas-de"

    for page in range(1, max_pages + 1):
        url = f"{base_url}-{term}?pagina={page}"
        print(f"[VAGAS.COM] Página {page}: {url}")

        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"[VAGAS.COM] Status {resp.status_code} na página {page}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

        except Exception as e:
            print(f"[VAGAS.COM] Erro ao carregar página {page}:", e)
            continue

        cards = soup.select("article.job-card")

        print(f"[VAGAS.COM] Encontrados {len(cards)} cards na página {page}")

        for card in cards:
            # ---- Link ----
            link_tag = card.find("a", class_="job-card__title-link")
            if not link_tag:
                continue

            link = "https://www.vagas.com.br" + link_tag.get("href")

            # ---- Título ----
            titulo = link_tag.text.strip()

            # ---- Empresa ----
            empresa_tag = card.find("span", class_="job-card__company")
            empresa = empresa_tag.text.strip() if empresa_tag else ""

            # ---- Descrição (puxar da página da vaga) ----
            descricao = scrap_vagas_com_detalhes(link)

            doc = {
                "_uid": make_uid(titulo, empresa, "Vagas.com", link),
                "titulo": titulo,
                "empresa": empresa,
                "descricao": descricao,
                "url": link,
                "site": "Vagas.com"
            }

            vagas_col.update_one({"_uid": doc["_uid"]}, {"$set": doc}, upsert=True)
            vagas.append(doc)

        time.sleep(1)

    print(f"[VAGAS.COM] Total coletado: {len(vagas)}")
    return vagas


# ------------------------------
# EXECUTAR TODOS
# ------------------------------
def scrap_todos(max_pages=1):
    print("\n[SCRAP_TODOS] Iniciando...")

    # vagascom = scrap_vagascom()

    # todas = vagascom

    # print(f"[SCRAP_TODOS] Total final coletado: {len(todas)}")

    return


if __name__ == "__main__":
    scrap_todos()