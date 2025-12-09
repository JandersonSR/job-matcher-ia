import os
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pymongo
import re
import unicodedata
from typing import List, Dict

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
profissoes_regex_col = db["profissoes_regex"]

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

        descricao = descricao_resumida + " " + scrap_vagas_com_detalhes(link)

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
        vagas = []
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
            vaga = upsert_vaga_with_search_terms(vaga)

            tem_vaga = vagas_col.find_one({
                "$or": [
                    {"_uid": vaga["_uid"]},
                    {"titulo": vaga["titulo"], "empresa": vaga["empresa"], "site": vaga["site"]}
                ]
            })
            if tem_vaga:
                vagas_col.update_one({"_uid": tem_vaga["_uid"]}, {"$set": vaga})
                continue

            vagas_col.update_one({"_uid": vaga["_uid"]}, {"$set": vaga}, upsert=True)
            vagas_final.append(vaga)

        time.sleep(1)

    print(f"[VAGAS.COM] Total coletado: {len(vagas_final)}")

    return vagas_final

# -------------------------
# Normalização / utilitários
# -------------------------
STOP_WORDS = {
    "de","da","do","dos","das","e","ou","a","o","que","para","por","com","sem",
    "um","uma","como","em","no","na","nos","nas","sua","seu","se","os","as",
    "profissional","vaga","vagas"
}

def _normalize_token(tok: str) -> str:
    """
    Normaliza um token: tira acentos, lower, mantém letras/números/-
    Retorna token vazio se inválido.
    """
    if not tok:
        return ""
    tok = tok.lower().strip()
    tok = unicodedata.normalize("NFD", tok)
    tok = tok.encode("ascii", "ignore").decode("utf-8")  # remove acentos
    tok = re.sub(r"[^a-z0-9\-]", " ", tok)
    tok = re.sub(r"\s+", " ", tok).strip()
    if not tok or len(tok) < 3:
        return ""
    if tok in STOP_WORDS:
        return ""
    return tok

# -------------------------
# Gerar search_terms compactos
# -------------------------
def gerar_search_terms(doc: Dict, max_terms: int = 12) -> List[str]:
    """
    Gera um conjunto pequeno e relevante de termos para busca (search_terms).
    Prioriza palavras do título e bigramas do título, depois pega palavras relevantes da descrição.
    Limita ao 'max_terms' para evitar arrays gigantes.
    """
    titulo = (doc.get("titulo") or "").strip()
    descricao = (doc.get("descricao") or "").strip()

    tokens = []
    # 1) tokens do título (maior prioridade)
    for part in re.split(r"[\/\-\|,]", titulo):
        for w in re.split(r"\s+", part):
            t = _normalize_token(w)
            if t:
                tokens.append(t)

    # 2) bigramas do título (ex: "chefe cozinha" -> "chefe-cozinha")
    titulo_words = [w for w in (_normalize_token(w) for w in re.split(r"\s+", titulo)) if w]
    for i in range(len(titulo_words) - 1):
        big = f"{titulo_words[i]}-{titulo_words[i+1]}"
        if len(big) <= 30:
            tokens.append(big)

    # 3) pegar palavras significativas do início da descrição (até N palavras)
    if not tokens or len(tokens) < (max_terms // 2):
        # extrair primeiros parágrafos ou primeiras 200 chars
        head = descricao.split("\n")[0][:400]
        for w in re.split(r"[^\w\-]+", head):
            t = _normalize_token(w)
            if t:
                tokens.append(t)

    # 4) dedupe preservando ordem e limitar
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms:
            break

    return out

# -------------------------
# Upsert otimizado de vaga (usar no scraper)
# -------------------------
def upsert_vaga_with_search_terms(vaga: Dict, max_terms: int = 12) -> Dict:
    """
    Gera 'search_terms' e salva a vaga no Mongo (update_one upsert).
    - vaga: dict contendo pelo menos '_uid','titulo','descricao','url'
    """
    # gera termos compactos
    termos = gerar_search_terms(vaga, max_terms=max_terms)
    vaga["search_terms"] = termos

    # tentar mapear profissões existentes (se coleção fornecida)
    if profissoes_regex_col is not None and termos:
        # procurar profissões que batem com qualquer termo (campo 'term' deve conter termos normalizados)
        encontrados = list(profissoes_regex_col.find({"term": {"$in": termos}}, {"term": 1}))
        profs = [e["term"] for e in encontrados]
        if profs:
            vaga["profissoes_detectadas"] = profs

    # salvar (upsert por _uid)
    vagas_col.update_one({"_uid": vaga["_uid"]}, {"$set": vaga}, upsert=True)
    return vaga

# -------------------------
# Busca rápida usando search_terms
# -------------------------
def buscar_vagas_por_term_simples(term: str, limit: int = 50) -> List[Dict]:
    """
    Busca rápida e indexável: normaliza o termo e pesquisa equality em search_terms.
    """
    t = _normalize_token(term)
    if not t:
        return []
    docs = list(vagas_col.find({"search_terms": t}, {"titulo":1,"descricao":1,"url":1,"empresa":1,"site":1,"embedding":1}).limit(limit))
    return docs

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

def scrap_vagas_com_detalhes(url: str, timeout: int = 10, retry: int = 2, sleep_between_retries: float = 0.6) -> str:
    """
    Extrai a descrição da vaga do HTML, suportando layouts antigos e novos da Vagas.com.
    """

    def clean_text(txt: str) -> str:
        if not txt:
            return ""
        # remove javascript, tracking, anúncios, CSS, etc
        blacklist = [
            r"googletag", r"pbjs", r"function\s*\(", r"var ", r"adserver", r"pubads",
            r"GPTAsync", r"setTimeout", r"PREBID", r"FAILSAFE"
        ]
        for b in blacklist:
            if re.search(b, txt, flags=re.IGNORECASE):
                return ""
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for attempt in range(retry):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                time.sleep(sleep_between_retries)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # --- remover scripts, estilos, anúncios ---
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            for advert in soup.select(".publicidade, .ad, .ads, .banner"):
                advert.decompose()

            textos = []

            # --- 1) Layout antigo Vagas.com ---
            selectors_antigos = [
                "div.detalhes > p",
                "div#detalhes > p",
                "div.boxVaga > p",
                "section#detalhes-vaga p",
                "div.informacoes > p",
            ]
            for sel in selectors_antigos:
                nodes = soup.select(sel)
                for p in nodes:
                    t = clean_text(p.get_text(" ", strip=True))
                    if t and len(t) > 20:
                        textos.append(t)

            # --- 2) Layout novo: job-tab-content ---
            if not textos:
                nodes = soup.select("div.job-tab-content.job-description__text p, div.job-tab-content.job-description__text")
                for p in nodes:
                    t = clean_text(p.get_text(" ", strip=True))
                    if t and len(t) > 20:
                        textos.append(t)

            # --- 3) fallback: pegar os <p> mais longos da página ---
            if not textos:
                all_p = soup.find_all("p")
                fallback = []
                for p in all_p:
                    t = clean_text(p.get_text(" ", strip=True))
                    if t and len(t) > 30:
                        fallback.append(t)
                if fallback:
                    fallback = sorted(fallback, key=len, reverse=True)
                    textos.extend(fallback[:3])

            # --- 4) retorna resultado ou mensagem padrão ---
            if textos:
                return "\n".join(textos)
            else:
                return "Descrição não disponível."

        except Exception:
            time.sleep(sleep_between_retries)
            continue

    return ""

if __name__ == "__main__":
    scrap_todos()