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

# ------------------------------
# GUPY API (FUNCIONA 100%)
# ------------------------------
def scrap_gupy(term="desenvolvedor"):
    print("[GUPY] Coletando vagas via API...")
    base_url = f"https://portal.gupy.io/api/jobs?term={term}"

    try:
        resp = requests.get(base_url, timeout=15)
        data = resp.json()
    except Exception as e:
        print("[GUPY] Erro:", e)
        return []

    vagas = []
    for item in data.get("data", []):
        titulo = item.get("name", "")
        empresa = item.get("careerPageName", "Gupy")
        descricao = item.get("description", "")
        link = f"https://portal.gupy.io/jobs/{item.get('id')}"

        uid = make_uid(titulo, empresa, "Gupy", link)

        doc = {
            "_uid": uid,
            "titulo": titulo,
            "empresa": empresa,
            "descricao": descricao,
            "url": link,
            "site": "Gupy"
        }

        vagas_col.update_one({"_uid": uid}, {"$set": doc}, upsert=True)
        vagas.append(doc)

    print(f"[GUPY] Total coletado: {len(vagas)}")
    return vagas

# ------------------------------
# ADZUNA API (FUNCIONA 100%)
# ------------------------------
def scrap_adzuna(term="developer", country="br", max_pages=1):
    print("[ADZUNA] Coletando vagas...")

    APP_ID = os.getenv("ADZUNA_ID")
    APP_KEY = os.getenv("ADZUNA_KEY")

    if not APP_ID or not APP_KEY:
        print("[ADZUNA] Chaves não encontradas no .env")
        return []

    vagas = []
    base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{{}}?app_id={APP_ID}&app_key={APP_KEY}&results_per_page=50&what={term}"

    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(base_url.format(page), timeout=15)
            data = resp.json()
        except Exception as e:
            print("[ADZUNA] Erro:", e)
            continue

        for item in data.get("results", []):
            titulo = item.get("title", "")
            empresa = item.get("company", {}).get("display_name", "")
            descricao = item.get("description", "")
            link = item.get("redirect_url", "")

            uid = make_uid(titulo, empresa, "Adzuna", link)

            doc = {
                "_uid": uid,
                "titulo": titulo,
                "empresa": empresa,
                "descricao": descricao,
                "url": link,
                "site": "Adzuna"
            }

            vagas_col.update_one({"_uid": uid}, {"$set": doc}, upsert=True)
            vagas.append(doc)

        print(f"[ADZUNA] Página {page}: {len(vagas)} acumuladas")
        time.sleep(1)

    return vagas

# ------------------------------
# JOOLE API (FUNCIONA 100%)
# ------------------------------
def scrap_jooble(keyword="developer"):
    print("[JOOBLE] Coletando vagas...")

    JOOBLE_KEY = os.getenv("JOOBLE_KEY")
    if not JOOBLE_KEY:
        print("[JOOBLE] Chave não encontrada no .env")
        return []

    url = f"https://jooble.org/api/{JOOBLE_KEY}"
    body = {"keywords": keyword}

    try:
        resp = requests.post(url, json=body, timeout=15)
        data = resp.json()
    except Exception as e:
        print("[JOOBLE] Erro:", e)
        return []

    vagas = []
    for item in data.get("jobs", []):
        titulo = item.get("title", "")
        empresa = item.get("company", "")
        descricao = item.get("snippet", "")
        link = item.get("link", "")

        uid = make_uid(titulo, empresa, "Jooble", link)

        doc = {
            "_uid": uid,
            "titulo": titulo,
            "empresa": empresa,
            "descricao": descricao,
            "url": link,
            "site": "Jooble"
        }

        vagas_col.update_one({"_uid": uid}, {"$set": doc}, upsert=True)
        vagas.append(doc)

    print(f"[JOOBLE] Total: {len(vagas)}")
    return vagas

# ------------------------------
# TRABALHA BRASIL (API PÚBLICA)
# ------------------------------
def scrap_trabalhabrasil(keyword="desenvolvedor"):
    print("[TRABALHABRASIL] Coletando vagas...")

    url = f"https://api.trabalhabrasil.com.br/v1/jobs?search={keyword}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except Exception as e:
        print("[TRABALHABRASIL] Erro:", e)
        return []

    vagas = []
    for item in data.get("items", []):
        titulo = item.get("title", "")
        empresa = item.get("company", "")
        descricao = item.get("description", "")
        link = item.get("url", "")

        uid = make_uid(titulo, empresa, "TrabalhaBrasil", link)

        doc = {
            "_uid": uid,
            "titulo": titulo,
            "empresa": empresa,
            "descricao": descricao,
            "url": link,
            "site": "TrabalhaBrasil"
        }

        vagas_col.update_one({"_uid": uid}, {"$set": doc}, upsert=True)
        vagas.append(doc)

    print(f"[TRABALHABRASIL] Total: {len(vagas)}")
    return vagas

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

                # if not titulo or not empresa:
                #     # se card não tem info suficiente, pulamos
                #     continue

                doc = {
                    "titulo": titulo,
                    "empresa": empresa,
                    "descricao": descricao,
                    "url": link if link.startswith("http") else f"https://www.infojobs.com.br{link}",
                    "site": "InfoJobs"
                }
                doc["_uid"] = make_uid(doc["titulo"], doc["empresa"], doc["site"], doc["url"])

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
                doc["_uid"] = make_uid(doc["titulo"], doc["empresa"], doc["site"], doc["url"])

                vagas_col.update_one({"_uid": doc["_uid"]}, {"$set": doc}, upsert=True)
                vagas.append(doc)

            print(f"[Catho] página {page} processada, vagas coletadas nesta página: {len(job_cards)}")
            time.sleep(1.0)
        except Exception as e:
            print(f"[Catho] erro ao processar {url}: {e}")

    return vagas

def scrap_indeed(term="desenvolvedor", location="Brasil"):
    print("[INDEED] Coletando vagas...")

    query = term.replace(" ", "+")
    url = f"https://br.indeed.com/jobs?q={query}&l={location}"

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0"
        })
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print("[INDEED] Erro:", e)
        return []

    cards = soup.select("div.job_seen_beacon")
    vagas = []

    for card in cards:
        titulo = card.select_one("h2 span")
        empresa = card.select_one(".companyName")
        link = card.select_one("a")

        if not titulo or not empresa or not link:
            continue

        titulo = titulo.text.strip()
        empresa = empresa.text.strip()
        url_vaga = "https://br.indeed.com" + link.get("href")

        uid = hashlib.sha1(f"{titulo}|{empresa}|Indeed|{url_vaga}".encode()).hexdigest()

        doc = {
            "_uid": uid,
            "titulo": titulo,
            "empresa": empresa,
            "url": url_vaga,
            "descricao": "",
            "site": "Indeed"
        }

        vagas_col.update_one({"_uid": doc['uid']}, {"$set": doc}, upsert=True)
        vagas.append(doc)

    print(f"[INDEED] Total coletado: {len(vagas)}")
    return vagas

def scrap_vagascom(term="desenvolvedor"):
    print("[VAGAS.COM] Coletando vagas...")

    query = term.replace(" ", "%20")
    url = f"https://www.vagas.com.br/vagas-de-{query}"

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0"
        })
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print("[VAGAS.COM] Erro:", e)
        return []

    vagas = []
    cards = soup.select("li.vaga")

    for card in cards:
        titulo = card.select_one("h2")
        empresa = card.select_one(".empresanome")
        link = card.select_one("a")

        if not titulo or not link:
            continue

        titulo = titulo.text.strip()
        empresa = empresa.text.strip() if empresa else "Vagas.com"
        url_vaga = "https://www.vagas.com.br" + link.get("href")

        uid = hashlib.sha1(f"{titulo}|{empresa}|Vagas.com|{url_vaga}".encode()).hexdigest()

        doc = {
            "_uid": uid,
            "titulo": titulo,
            "empresa": empresa,
            "descricao": "",
            "url": url_vaga,
            "site": "Vagas.com"
        }

        vagas_col.update_one({"_uid": doc['_uid']}, {"$set": doc}, upsert=True)
        vagas.append(doc)

    print(f"[VAGAS.COM] Total coletado: {len(vagas)}")
    return vagas

# ------------------------------
# EXECUTAR TODOS
# ------------------------------
def scrap_todos(max_pages=1):
    print("\n[SCRAP_TODOS] Iniciando...")

    gupy = scrap_gupy()
    adzuna = scrap_adzuna()
    jooble = scrap_jooble()
    tb = scrap_trabalhabrasil()

    vagas_infojobs = scrap_vagas_infojobs(max_pages=max_pages)
    vagas_catho = scrap_vagas_catho(max_pages=max_pages)

    indeed = scrap_indeed()
    vagascom = scrap_vagascom()

    todas = vagas_infojobs + vagas_catho + gupy + adzuna + jooble + tb + indeed + vagascom

    print(f"[SCRAP_TODOS] Total final coletado: {len(todas)}")

    return todas


if __name__ == "__main__":
    scrap_todos()
