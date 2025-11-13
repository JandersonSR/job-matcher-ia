# webscrapping.py
import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pymongo

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")

client = pymongo.MongoClient(MONGO_URL)
db = client[DB_NAME]
vagas_col = db["vagas"]

HEADERS = {
    "User-Agent": os.getenv("USER_AGENT",
                             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100 Safari/537.36")
}

def _make_id(titulo: str, empresa: str, site: str, url: str) -> str:
    key = f"{titulo}|{empresa}|{site}|{url}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def _safe_get_text(tag):
    return tag.get_text(strip=True) if tag else ""

def scrap_vagas_infojobs(max_pages=3):
    """
    Scraping simplificado do InfoJobs.
    Observação: sites podem mudar estrutura — mantenha seletores atualizados.
    """
    print("[scrap_vagas_infojobs] iniciando...")
    base_url = "https://www.infojobs.com.br/vagas-de-emprego"
    vagas = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?Page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[InfoJobs] status {resp.status_code} em {url}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Tenta buscar por cartões de vaga (seletor genérico). Ajuste conforme a real estrutura do site.
            job_cards = soup.select("a[href*='/vaga/'], div.vaga, li.resultado")  # tentativa multi-seletor
            if not job_cards:
                # fallback: encontrar links que pareçam vagas
                job_cards = soup.find_all("a")
                job_cards = [a for a in job_cards if "vaga" in (a.get("href") or "")][:50]

            for card in job_cards:
                # Tentar extrair título e empresa de diferentes formas
                titulo = _safe_get_text(card.find("h2")) or _safe_get_text(card.find("h3")) or _safe_get_text(card)
                empresa = _safe_get_text(card.find("span", class_="company")) or _safe_get_text(card.find("span", class_="empresa"))
                descricao = _safe_get_text(card.find("p")) or ""
                link = card.get("href") or url

                if not titulo or not empresa:
                    # se card não tem info suficiente, pulamos
                    continue

                doc = {
                    "titulo": titulo,
                    "empresa": empresa,
                    "descricao": descricao,
                    "url": link if link.startswith("http") else f"https://www.infojobs.com.br{link}",
                    "site": "InfoJobs"
                }
                doc["_uid"] = _make_id(doc["titulo"], doc["empresa"], doc["site"], doc["url"])

                # upsert por _uid
                vagas_col.update_one({"_uid": doc["_uid"]}, {"$set": doc}, upsert=True)
                vagas.append(doc)

            print(f"[InfoJobs] página {page} processada, vagas coletadas nesta página: {len(job_cards)}")
            time.sleep(1.0)
        except Exception as e:
            print(f"[InfoJobs] erro ao processar {url}: {e}")

    return vagas

def scrap_vagas_catho(max_pages=3):
    """
    Scraping simplificado do Catho. Ajuste seletores conforme necessidade.
    """
    print("[scrap_vagas_catho] iniciando...")
    base_url = "https://www.catho.com.br/vagas"
    vagas = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?pagina={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[Catho] status {resp.status_code} em {url}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            job_cards = soup.select("a[href*='/vagas/'], div.vaga-card, li.vaga-list-item")
            if not job_cards:
                job_cards = soup.find_all("a")
                job_cards = [a for a in job_cards if "vaga" in (a.get("href") or "")][:50]

            for card in job_cards:
                titulo = _safe_get_text(card.find("h2")) or _safe_get_text(card.find("h3")) or _safe_get_text(card)
                empresa = _safe_get_text(card.find("div", class_="empresa")) or _safe_get_text(card.find("span", class_="company"))
                descricao = _safe_get_text(card.find("p")) or ""
                link = card.get("href") or url

                if not titulo or not empresa:
                    continue

                doc = {
                    "titulo": titulo,
                    "empresa": empresa,
                    "descricao": descricao,
                    "url": link if link.startswith("http") else f"https://www.catho.com.br{link}",
                    "site": "Catho"
                }
                doc["_uid"] = _make_id(doc["titulo"], doc["empresa"], doc["site"], doc["url"])

                vagas_col.update_one({"_uid": doc["_uid"]}, {"$set": doc}, upsert=True)
                vagas.append(doc)

            print(f"[Catho] página {page} processada, vagas coletadas nesta página: {len(job_cards)}")
            time.sleep(1.0)
        except Exception as e:
            print(f"[Catho] erro ao processar {url}: {e}")

    return vagas

def scrap_todos(max_pages=3):
    """
    Executa todos os scrapers e retorna a lista de vagas encontradas nesta execução.
    """
    print("[scrap_todos] iniciando scraping de todos os sites...")
    vagas_infojobs = scrap_vagas_infojobs(max_pages=max_pages)
    vagas_catho = scrap_vagas_catho(max_pages=max_pages)

    total = len(vagas_infojobs) + len(vagas_catho)
    print(f"[scrap_todos] concluído. Total de vagas processadas (nesta execução): {total}")
    return vagas_infojobs + vagas_catho

if __name__ == "__main__":
    scrap_todos(max_pages=1)
