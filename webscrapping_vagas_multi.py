import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pymongo
import re

load_dotenv()

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options

# chrome_options = Options()
# chrome_options.add_argument("--headless")  # roda sem abrir janela
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")

# service = Service("/caminho/para/chromedriver")  # coloque o caminho correto
# driver = webdriver.Chrome(service=service, options=chrome_options)


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


# --------------------------------------------------
# SCRAPING DO LAYOUT NOVO (cards <article.job-card>)
# --------------------------------------------------
def scrap_vagascom_novo_layout(soup):
    cards = soup.select("article.job-card")
    vagas = []

    for card in cards:
        link_tag = card.find("a", class_="job-card__title-link")
        if not link_tag:
            continue

        link = "https://www.vagas.com.br" + link_tag.get("href")
        titulo = link_tag.text.strip()

        empresa_tag = card.find("span", class_="job-card__company")
        empresa = empresa_tag.text.strip() if empresa_tag else ""

        descricao = scrap_vagas_com_detalhes(link)

        vagas.append({
            "_uid": make_uid(titulo, empresa, "Vagas.com", link),
            "titulo": titulo,
            "empresa": empresa,
            "descricao": descricao,
            "url": link,
            "site": "Vagas.com"
        })
    return vagas


# --------------------------------------------------
# SCRAPING DO LAYOUT ANTIGO (<li class="vaga">)
# --------------------------------------------------
def scrap_vagascom_layout_antigo(soup):
    itens = soup.select("li.vaga")
    vagas = []

    for item in itens:
        link_tag = item.select_one("a.link-detalhes-vaga")
        if not link_tag:
            continue

        link = "https://www.vagas.com.br" + link_tag.get("href")
        titulo = link_tag.get("title") or link_tag.text.strip()

        empresa_tag = item.select_one(".emprVaga")
        empresa = empresa_tag.text.strip() if empresa_tag else ""

        descricao_tag = item.select_one(".detalhes p")
        descricao_resumida = descricao_tag.text.strip() if descricao_tag else ""

        descricao = scrap_vagas_com_detalhes(link)

        vagas.append({
            "_uid": make_uid(titulo, empresa, "Vagas.com", link),
            "titulo": titulo,
            "empresa": empresa,
            "descricao": descricao or descricao_resumida,
            "url": link,
            "site": "Vagas.com"
        })

    return vagas


# --------------------------------------------------
# SCRAPING PRINCIPAL + FALLBACK
# --------------------------------------------------
def scrap_vagascom(term="desenvolvedor", max_pages=1):
    print("[VAGAS.COM] Iniciando scraping...")

    vagas_final = []
    base_url = "https://www.vagas.com.br/vagas-de"

    for page in range(0, max_pages + 1):
        url = (
            f"{base_url}-{term}"
            # f"{base_url}-{term}" if page == 0
            # else f"{base_url}-{term}?pagina={page}"
        )

        if page > 0:
            continue

        print(f"[VAGAS.COM] Página {page}: {url}")

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"[VAGAS.COM] Status {resp.status_code} na página {page}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

        except Exception as e:
            print(f"[VAGAS.COM] Erro ao carregar página {page}:", e)
            continue

        # ------------------------------
        # TENTAR LAYOUT NOVO
        # ------------------------------
        vagas = [] # scrap_vagascom_novo_layout(soup)
        # print(f"[VAGAS.COM] Novo layout retornou: {len(vagas)} vagas")

        if len(vagas) == 0:
            # ------------------------------
            # FALLBACK PARA O LAYOUT ANTIGO
            # ------------------------------
            print("[VAGAS.COM] Tentando layout antigo...")
            vagas = scrap_vagascom_layout_antigo(soup)
            print(f"[VAGAS.COM] Layout antigo retornou: {len(vagas)} vagas")

        # Armazena no DB e adiciona à lista final
        for vaga in vagas:
            vagas_col.update_one({"_uid": vaga["_uid"]}, {"$set": vaga}, upsert=True)
            vagas_final.append(vaga)

        time.sleep(1)

    print(f"[VAGAS.COM] Total coletado: {len(vagas_final)}")

    return vagas_final

# ------------------------------
# EXECUTAR TODOS
# ------------------------------
def scrap_todos(max_pages=1):
    print("\n[SCRAP_TODOS] Iniciando...")

    # vagascom = scrap_vagascom()

    # todas = vagascom

    # print(f"[SCRAP_TODOS] Total final coletado: {len(todas)}")

    return []

def _clean_text(txt: str) -> str:
    """Limpeza simples do texto: normaliza espaços e remove prefixos comuns."""
    if not txt:
        return ""
    # remover "Descrição:" ou "Descrição da vaga:" no começo
    txt = re.sub(r'^\s*descri[cç][aã]o[:\-]?\s*', '', txt, flags=re.IGNORECASE)
    # normalizar espaços e quebras de linha
    txt = re.sub(r'\r', ' ', txt)
    txt = re.sub(r'\n{2,}', '\n', txt)
    txt = re.sub(r'[ \t]{2,}', ' ', txt)
    txt = txt.strip()
    return txt

def scrap_vagas_com_detalhes(url, driver):
    try:
        driver.get(url)
        time.sleep(2)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # ===== 1) Remover scripts, anúncios e elementos invisíveis =====
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # ===== 2) Possíveis seletores de descrição (NOVO + ANTIGO) =====
        seletores = [
            "div.job-description",                # layout novo
            "div#descricao",                      # layout antigo
            "div.descricao",                      # outra variante antiga
            "div.container p",                    # fallback interno
            "div.text-content",                   # usado em algumas vagas antigas
        ]

        descricao = None

        for sel in seletores:
            bloco = soup.select_one(sel)
            if bloco:
                descricao = bloco.get_text(separator=" ", strip=True)
                break

        # ===== 3) Se ainda não achou, tentar dentro do main =====
        if not descricao:
            main = soup.find("main")
            if main:
                descricao = main.get_text(separator=" ", strip=True)

        # ===== 4) última barreira — remover lixo de anúncios =====
        blacklist = [
            "googletag", "pbjs", "adUnits", "PREBID",
            "refresh()", "gpt", "adserverRequest", "FAILSAFE"
        ]

        if descricao:
            for termo in blacklist:
                if termo in descricao:
                    descricao = None
                    break

        # ===== 5) fallback final: não retorna JavaScript =====
        if not descricao or len(descricao) < 30:
            descricao = "Descrição não disponível."

        return descricao

    except Exception as e:
        print("Erro ao extrair descrição:", e)
        return "Descrição não disponível."


# def scrap_vagas_com_detalhes(url: str, timeout: int = 10, retry: int = 2, sleep_between_retries: float = 0.6) -> str:
#     def clean_text(txt: str) -> str:
#         if not txt:
#             return ""

#         # remove javascript, prebid, googletag, tracking, CSS, etc
#         blacklist = [
#             r"googletag", r"pbjs", r"function\s*\(", r"var ", r"adserver", r"pubads",
#             r"GPTAsync", r"setTimeout", r"PREBID", r"FAILSAFE"
#         ]
#         for b in blacklist:
#             if re.search(b, txt, flags=re.IGNORECASE):
#                 return ""

#         txt = re.sub(r"\s+", " ", txt).strip()
#         return txt

#     headers = {
#         "User-Agent": "Mozilla/5.0"
#     }

#     for attempt in range(retry):
#         try:
#             resp = requests.get(url, headers=headers, timeout=timeout)
#             if resp.status_code != 200:
#                 time.sleep(sleep_between_retries)
#                 continue

#             soup = BeautifulSoup(resp.text, "html.parser")

#             # --- REMOVER TUDO QUE NÃO QUEREMOS ---
#             for tag in soup(["script", "style", "noscript"]):
#                 tag.decompose()
#             for advert in soup.select(".publicidade, .ad, .ads, .banner"):
#                 advert.decompose()

#             # --- SELETORES DO LAYOUT ANTIGO DA VAGAS.COM ---
#             selectors_prioritarios = [
#                 "div.detalhes > p",             # layout antigo clássico
#                 "div#detalhes > p",
#                 "div.boxVaga > p",
#                 "section#detalhes-vaga p",
#                 "div.informacoes > p",
#             ]

#             textos = []

#             # 1) tenta seletores antigos
#             for sel in selectors_prioritarios:
#                 nodes = soup.select(sel)
#                 for p in nodes:
#                     t = clean_text(p.get_text(" ", strip=True))
#                     if t and len(t) > 20:
#                         textos.append(t)

#             if textos:
#                 return "\n".join(textos)

#             # 2) fallback: pegar <p> significativos, ignorando lixo
#             all_p = soup.find_all("p")
#             textos_fallback = []
#             for p in all_p:
#                 t = clean_text(p.get_text(" ", strip=True))
#                 # rejeitar textos muito curtos ou suspeitos
#                 if t and len(t) > 30:
#                     textos_fallback.append(t)

#             if textos_fallback:
#                 # retorna só os maiores textos (descrição real)
#                 textos_fallback = sorted(textos_fallback, key=len, reverse=True)
#                 return "\n".join(textos_fallback[:3])

#             return ""

#         except Exception:
#             time.sleep(sleep_between_retries)
#             continue

#     return ""


if __name__ == "__main__":
    scrap_todos()