import os
import json
import time
import requests
import hashlib
import logging
from typing import List, Tuple, Optional

import torch
import numpy as np
import pymongo
import re
import unicodedata
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# se usar Ollama local via OpenAI compat wrapper (opcional)
from openai import OpenAI

# webscraper que você já tem
from webscrapping_vagas_multi import scrap_vagascom

load_dotenv()
logging.basicConfig(level=logging.INFO)

# -----------------------
# CONFIG / DB
# -----------------------
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL não encontrado no .env")

client = pymongo.MongoClient(MONGO_URL)
db = client[DB_NAME]

curriculos_col = db["curriculos"]
vagas_col = db["vagas"]
scrap_cache_col = db["scrap_cache"]  # cache de scrapping

# recomenda criar índices (executar uma vez)
try:
    vagas_col.create_index([("titulo", "text"), ("descricao", "text")])
    scrap_cache_col.create_index("term", unique=True)
    curriculos_col.create_index("doc_hash", unique=False)
except Exception:
    pass

# -----------------------
# MODELOS
# -----------------------
# Embedding model (carregar uma vez)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM local (opcional)
RUN_LOCAL_LLM = os.getenv("RUN_LOCAL_LLM", "false").lower() == "true"
llm_client = None
if RUN_LOCAL_LLM:
    llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    logging.warning("⚠️ LLM local habilitado (Ollama)")

# OpenAI API (para _call_llm)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.info("OPENAI_API_KEY não configurado — funções LLM vão falhar se usadas")

# -----------------------
# CONSTANTES / HYPERPARAMS
# -----------------------
SIM_THRESHOLD = 0.45       # limiar para considerar requisito atendido
TOP_K_EMBEDDINGS = 10      # número de vagas consideradas via embeddings
TOP_N_LLM = 3              # número de vagas que serão avaliadas pelo LLM
SCRAP_MAX_PAGES = 2        # páginas a scrapear quando necessário

# -----------------------
# UTIL: hash do currículo (para cache)
# -----------------------
def hash_curriculo(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

# -----------------------
# UTIL: sanitizadores
# -----------------------
def reduzir_profissao(profissao: str) -> str:
    if not profissao:
        return ""
    STOP_WORDS = [
        "responsavel", "responsável", "pleno", "junior", "júnior", "senior", "sênior",
        "assistente", "lider", "líder", "especialista",
        "técnico", "tecnico"
    ]
    p = profissao.lower()
    p = unicodedata.normalize("NFD", p).encode("ascii", "ignore").decode("utf-8")
    palavras = p.split()
    palavras_filtradas = [w for w in palavras if w not in STOP_WORDS]
    if not palavras_filtradas:
        return palavras[0] if palavras else ""
    return palavras_filtradas[0]

def sanitizer_vagas_term(term: str) -> str:
    term = term.strip().lower()
    term = term.replace(" ", "-")
    term = re.sub(r"[^a-z0-9\-]+", "", term)
    return term

# -----------------------
# FUNÇÕES DE EMBEDDINGS OTIMIZADAS
# - salvar embedding de vaga no DB ao inserção / scrap
# - recuperar embedding já salvo
# -----------------------
def vaga_embedding_from_db(vaga_doc: dict) -> Optional[torch.Tensor]:
    emb = vaga_doc.get("embedding")
    if emb is None:
        return None
    # emb salvo como lista de floats
    arr = np.array(emb, dtype=np.float32)
    return torch.from_numpy(arr)

def ensure_vaga_embedding(vaga_doc: dict) -> torch.Tensor:
    emb = vaga_embedding_from_db(vaga_doc)
    if emb is not None:
        return emb
    # se não existir, calcule e salve
    descricao = (vaga_doc.get("titulo", "") + " " + vaga_doc.get("descricao", "")).strip()
    emb_vec = embedding_model.encode(descricao, convert_to_tensor=False)
    # salvar como lista (compatível com Mongo)
    vagas_col.update_one({"_id": vaga_doc["_id"]}, {"$set": {"embedding": emb_vec.tolist()}})
    return torch.from_numpy(np.array(emb_vec, dtype=np.float32))

# -----------------------
# EXTRACAO DE REQUISITOS
# -----------------------
def extrair_requisitos(texto_descricao: str) -> List[str]:
    linhas = texto_descricao.split("\n")
    requisitos = []
    for linha in linhas:
        linha = linha.strip("-•* \t").strip()
        if len(linha) > 3:
            sub_reqs = [r.strip() for r in re.split(r",|;|\.", linha) if len(r.strip()) > 2]
            requisitos.extend(sub_reqs)
    # opcional: deduplicate preserving order
    seen = set()
    out = []
    for r in requisitos:
        if r.lower() not in seen:
            seen.add(r.lower())
            out.append(r)
    return out

# -----------------------
# COMPARAR REQUISITOS (usa embeddings de habilidades do currículo)
# -----------------------
def comparar_requisitos(requisitos_pairs: List[Tuple[str, np.ndarray]], habilidades_emb: torch.Tensor) -> Tuple[List[str], List[str]]:
    requisitos_atendidos = []
    requisitos_nao_atendidos = []
    if habilidades_emb is None or habilidades_emb.shape[0] == 0:
        # Nenhuma habilidade detectada -> tudo não atendido
        for req_text, _ in requisitos_pairs:
            requisitos_nao_atendidos.append(req_text)
        return requisitos_atendidos, requisitos_nao_atendidos

    # preparar habilidades_emb como tensor [N, dim]
    if isinstance(habilidades_emb, np.ndarray):
        habilidades_emb = torch.from_numpy(habilidades_emb)
    for req_text, req_emb in requisitos_pairs:
        if req_emb is None:
            requisitos_nao_atendidos.append(req_text)
            continue
        if isinstance(req_emb, np.ndarray):
            req_emb = torch.from_numpy(np.array(req_emb, dtype=np.float32))
        if req_emb.ndim == 1:
            req_emb = req_emb.unsqueeze(0)
        sims = util.cos_sim(req_emb, habilidades_emb)
        max_sim = sims.max().item()
        sub_reqs = [r.strip() for r in re.split(r",|;|\.", req_text) if len(r.strip()) > 2]
        if max_sim >= SIM_THRESHOLD:
            requisitos_atendidos.extend(sub_reqs)
        else:
            requisitos_nao_atendidos.extend(sub_reqs)
    return requisitos_atendidos, requisitos_nao_atendidos

def gerar_melhorias(requisitos_nao_atendidos: List[str]) -> List[str]:
    melhorias = []
    for req in requisitos_nao_atendidos:
        req_norm = req.lower()
        melhorias.append(f"Adicionar experiência com {req_norm}.")
    return melhorias

# -----------------------
# LLM helper (mantive similar ao seu _call_llm)
# -----------------------
def _call_llm(prompt: str, timeout: int = 30, retries: int = 1) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurado")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(retries + 1):
        try:
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.2
            }
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logging.warning(f"[GPT] Tentativa {attempt + 1} falhou: {e}")
            time.sleep(1 + attempt * 2)
    raise RuntimeError("OpenAI API falhou após múltiplas tentativas")

def _extract_text_from_hf_response(resp_json) -> str:
    if isinstance(resp_json, list) and resp_json:
        first = resp_json[0]
        if isinstance(first, dict):
            return first.get("generated_text") or json.dumps(first)
        return str(first)
    return str(resp_json)

# -----------------------
# Extrair profissões (com cache no documento do currículo)
# -----------------------
def extrair_profissao_principal(texto_curriculo: str, max_profissoes: int = 5) -> List[str]:
    """
    Usa LLM para extrair profissões. Deve ser chamado apenas quando não houver profissões salvas.
    """
    prompt = f"""
Extraia no minimo 3 e maximo {max_profissoes} possíveis profissões do currículo abaixo.
Retorne APENAS as palavras/termos representando as profissões, separadas por vírgula,
sem qualquer explicação ou texto adicional.
Currículo:
{texto_curriculo}

Retorne no minimo 3 e no maximo {max_profissoes} profissões, separadas por vírgula:
"""
    resp = _call_llm(prompt)
    profissoes = [p.strip().lower() for p in resp.split(",") if p.strip()]
    return profissoes[:max_profissoes]

# -----------------------
# PROCESSAMENTO VIA EMBEDDINGS (rapido)
# - assume que cada vaga tem embedding salvo no DB (campo embedding)
# - compara currículo com vagas via cosine, sem recalcular embeddings das vagas
# -----------------------
def processar_com_embeddings(texto: str, vagas: List[dict], top_k: int = TOP_K_EMBEDDINGS) -> List[dict]:
    if not vagas:
        return []

    # 1) embedding do currículo (1 vez)
    embedding_curriculo = embedding_model.encode(texto, convert_to_tensor=True)

    # 2) quebra currículo em habilidades curtas e gera emb (batch)
    habilidades = [s.strip() for s in texto.split("\n") if len(s.strip()) > 3]
    habilidades_emb = None
    if habilidades:
        habilidades_vecs = embedding_model.encode(habilidades, convert_to_tensor=False)
        habilidades_emb = torch.from_numpy(np.array(habilidades_vecs, dtype=np.float32))

    # 3) para cada vaga, obter embedding (do DB ou calcular e salvar) e calcular similaridade
    resultados = []
    for vaga in vagas:
        descricao = (vaga.get("titulo", "") + " " + vaga.get("descricao", "")).strip()
        if not descricao:
            continue
        # obter embedding salvo OU criar e persistir
        emb_vaga = vaga_embedding_from_db(vaga)
        if emb_vaga is None:
            # calculo e salvamento em background (sincrono aqui)
            emb_vec = embedding_model.encode(descricao, convert_to_tensor=False)
            vagas_col.update_one({"_id": vaga["_id"]}, {"$set": {"embedding": emb_vec.tolist()}})
            emb_vaga = torch.from_numpy(np.array(emb_vec, dtype=np.float32))

        # compatibilidade geral
        score = util.cos_sim(embedding_curriculo, emb_vaga).item()

        # requisitos: extrai textos e gera embeddings em batch (se houver)
        req_text_list = extrair_requisitos(descricao)
        req_embeddings = []
        if req_text_list:
            req_vecs = embedding_model.encode(req_text_list, convert_to_tensor=False)
            req_embeddings = [np.array(v, dtype=np.float32) for v in req_vecs]
        requisitos_pairs = list(zip(req_text_list, req_embeddings))

        requisitos_atendidos, requisitos_nao_atendidos = comparar_requisitos(
            requisitos_pairs,
            habilidades_emb
        )
        melhorias = gerar_melhorias(requisitos_nao_atendidos)

        resultados.append({
            "vaga_id": vaga.get("_id") or vaga.get("_uid") or str(vaga.get("_id")),
            "titulo": vaga.get("titulo", "Sem título"),
            "empresa": vaga.get("empresa", "Desconhecida"),
            "descricao": descricao,
            "url": vaga.get("url"),
            "site": vaga.get("site", "Desconhecido"),
            "compatibilidade": round(float(score), 4),
            "requisitos_atendidos": requisitos_atendidos,
            "requisitos_nao_atendidos": requisitos_nao_atendidos,
            "melhorias_sugeridas": melhorias,
            "_raw_doc": vaga  # para referência se precisar
        })

    # ordena e retorna top_k
    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)
    return resultados[:top_k]

# -----------------------
# PROCESSAR COM LLM (apenas top-N vagas)
# - reduz chamadas ao LLM: recebe apenas as melhores vagas de embeddings
# -----------------------
def processar_com_llm(texto: str, vagas: List[dict], max_vagas_llm: int = TOP_N_LLM) -> List[dict]:
    resultados = []
    to_call = vagas[:max_vagas_llm]
    for vaga in to_call:
        descricao = vaga.get("descricao") or vaga.get("titulo") or ""
        prompt = f"""
Você é um especialista em Recrutamento e Seleção.
Analise cuidadosamente a VAGA e extraia uma lista de REQUISITOS a partir dela
(somente itens realmente mencionados no texto).
Depois, compare cada requisito com o CURRÍCULO.
REGRAS:
- Use apenas o texto da vaga e do currículo.
- Retorne APENAS um JSON válido com os campos:
  compatibilidade (numero entre 0 e 1), requisitos_atendidos (lista), requisitos_nao_atendidos (lista), melhorias_sugeridas (lista)
Currículo:
{texto}

Vaga:
{descricao}

Exemplo de saída:
{{"compatibilidade": 0.73, "requisitos_atendidos":["Python"], "requisitos_nao_atendidos":["Docker"], "melhorias_sugeridas":["Curso Docker básico"]}}
"""
        vaga_id = vaga.get("_id") or vaga.get("vaga_id") or vaga.get("_uid") or str(vaga.get("_id"))
        try:
            resp_text = _call_llm(prompt)
            raw = _extract_text_from_hf_response(resp_text)
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                json_text = raw[start:end+1]
            else:
                json_text = raw
            try:
                parsed = json.loads(json_text)
            except:
                parsed = {
                    "compatibilidade": 0.0,
                    "requisitos_atendidos": [],
                    "requisitos_nao_atendidos": [],
                    "melhorias_sugeridas": []
                }
            resultados.append({
                "vaga_id": vaga_id,
                "titulo": vaga.get("titulo"),
                "empresa": vaga.get("empresa"),
                "url": vaga.get("url"),
                "site": vaga.get("site"),
                "compatibilidade": round(float(parsed.get("compatibilidade", 0)), 2),
                "requisitos_atendidos": parsed.get("requisitos_atendidos", []),
                "requisitos_nao_atendidos": parsed.get("requisitos_nao_atendidos", []),
                "melhorias_sugeridas": parsed.get("melhorias_sugeridas", [])
            })
        except Exception as e:
            logging.warning(f"[LLM] erro ao processar vaga {vaga.get('titulo')}: {e}")
            resultados.append({
                "vaga_id": vaga_id,
                "titulo": vaga.get("titulo"),
                "empresa": vaga.get("empresa"),
                "compatibilidade": 0.0,
                "requisitos_atendidos": [],
                "requisitos_nao_atendidos": [],
                "melhorias_sugeridas": []
            })
    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)
    return resultados

# -----------------------
# GARANTIR VAGAS PARA PROFISSÃO
# - tenta no banco (filtro eficiente), se não houver -> usar scrap_cache -> scrap e inserir no banco
# -----------------------
def garantir_vagas_para_profissao(texto_curriculo: str, min_por_profissao: int = 5) -> List[dict]:
    profissoes = extrair_profissao_principal_cached(texto_curriculo)
    todas_vagas = []
    for profissao in profissoes:
        nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(nucleo)

        # busca no cache de scrapping primeiro (usando term normalizado)
        cache = scrap_cache_col.find_one({"term": term})
        if cache and cache.get("vagas"):
            logging.info(f"[CACHE] Usando vagas cacheadas para '{term}' ({len(cache['vagas'])})")
            # cache armazena os docs completos (ou pelo menos titulo/descricao/url)
            vagas_docs = cache["vagas"]
            # garantir que vagas retornadas sejam objetos como do DB (não têm _id) ->
            # para consistência, tentamos casar pelo url no DB e inserir se necessário
            normalized = []
            for v in vagas_docs:
                # tenta achar no DB por url
                if v.get("url"):
                    found = vagas_col.find_one({"url": v["url"]})
                    if found:
                        normalized.append(found)
                    else:
                        # inserir nova vaga com embedding e retornar
                        new_doc = dict(v)
                        new_doc["_id"] = vagas_col.insert_one(new_doc).inserted_id
                        emb_vec = embedding_model.encode((new_doc.get("titulo","") + " " + new_doc.get("descricao","")).strip(), convert_to_tensor=False)
                        vagas_col.update_one({"_id": new_doc["_id"]}, {"$set": {"embedding": emb_vec.tolist()}})
                        new_doc["embedding"] = emb_vec.tolist()
                        normalized.append(new_doc)
                else:
                    # sem url, só insere
                    new_doc = dict(v)
                    new_doc["_id"] = vagas_col.insert_one(new_doc).inserted_id
                    emb_vec = embedding_model.encode((new_doc.get("titulo","") + " " + new_doc.get("descricao","")).strip(), convert_to_tensor=False)
                    vagas_col.update_one({"_id": new_doc["_id"]}, {"$set": {"embedding": emb_vec.tolist()}})
                    new_doc["embedding"] = emb_vec.tolist()
                    normalized.append(new_doc)
            vagas_existentes = normalized
        else:
            # busca no DB com projeção para evitar transferir campos pesados
            regex = re.compile(re.escape(term), re.IGNORECASE)
            vagas_existentes = list(vagas_col.find(
                {"$or": [{"titulo": regex}, {"descricao": regex}]},
                {"titulo": 1, "descricao": 1, "url": 1, "empresa": 1, "site": 1, "embedding": 1}
            ).limit(50))  # limitar scans

        if len(vagas_existentes) >= min_por_profissao:
            logging.info(f"[OK] Encontradas {len(vagas_existentes)} vagas para '{term}' no DB")
            todas_vagas.extend(vagas_existentes)
            continue

        # se não tem vagas suficientes -> verificar scrap_cache para o termo (se não houve, scrap)
        if cache is None:
            logging.info(f"[SCRAP] Necessário scrap para termo '{term}'")
            # scrap_vagascom espera term (string) - seu scrap aceita slug ou regex? adaptado para usar term
            novas = scrap_vagascom(term=term, max_pages=SCRAP_MAX_PAGES)
            if not novas:
                logging.info(f"[SCRAP] Nenhuma vaga encontrada pelo scrap para '{term}'")
                continue

            # salvar resultado no cache para próxima vez
            try:
                scrap_cache_col.insert_one({"term": term, "vagas": novas, "ts": time.time()})
            except pymongo.errors.DuplicateKeyError:
                scrap_cache_col.update_one({"term": term}, {"$set": {"vagas": novas, "ts": time.time()}})

            # inserir as novas vagas no DB (evitar duplicatas - por url)
            inserted_docs = []
            for v in novas:
                url = v.get("url")
                # tenta evitar duplicata por url
                if url and vagas_col.find_one({"url": url}):
                    doc = vagas_col.find_one({"url": url})
                    inserted_docs.append(doc)
                    continue
                # insere e calcula embedding
                doc = dict(v)
                doc["_id"] = vagas_col.insert_one(doc).inserted_id
                emb_vec = embedding_model.encode((doc.get("titulo","") + " " + doc.get("descricao","")).strip(), convert_to_tensor=False)
                vagas_col.update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb_vec.tolist()}})
                doc["embedding"] = emb_vec.tolist()
                inserted_docs.append(doc)
            vagas_existentes = inserted_docs

        # se ainda tiver alguma vaga, adiciona
        if vagas_existentes:
            todas_vagas.extend(vagas_existentes)

    # devolver lista (pode conter duplicatas entre profissões — opcionalmente deduplicar por url)
    # deduplicar por url
    seen_urls = set()
    unique = []
    for v in todas_vagas:
        u = v.get("url") or str(v.get("_id"))
        if u in seen_urls:
            continue
        seen_urls.add(u)
        unique.append(v)
    return unique

# -----------------------
# Helper: extrair profissões com cache no curriculo
# -----------------------
def extrair_profissao_principal_cached(texto_curriculo: str, max_profissoes: int = 5) -> List[str]:
    # tenta achar documento de currículo pelo hash
    doc_hash = hash_curriculo(texto_curriculo)
    curr = curriculos_col.find_one({"doc_hash": doc_hash}, {"profissoes_detectadas": 1})
    if curr and curr.get("profissoes_detectadas"):
        logging.info("[CACHE] Profissões encontradas no currículo (cache)")
        return curr["profissoes_detectadas"]

    # se não há, extrai via LLM
    profs = extrair_profissao_principal(texto_curriculo, max_profissoes=max_profissoes)
    # salva no documento do currículo (upsert baseado no hash)
    curriculos_col.update_one(
        {"doc_hash": doc_hash},
        {"$set": {"profissoes_detectadas": profs, "doc_hash": doc_hash, "last_profession_extract_ts": time.time()}},
        upsert=True
    )
    logging.info("[LLM] Profissões extraídas e salvas no currículo")
    return profs

# -----------------------
# MODOS DE COMPARAÇÃO
# -----------------------
def comparar_por_embeddings(texto: str, top_k: int = TOP_K_EMBEDDINGS):
    # buscar uma amostra de vagas relevantes (por texto) - aqui usamos text search simples com as profissões
    profs = extrair_profissao_principal_cached(texto)
    candidate_vagas = []
    for profissao in profs:
        nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(nucleo)
        regex = re.compile(re.escape(term), re.IGNORECASE)
        # projeção leve (embedding pode existir)
        docs = list(vagas_col.find({"$or": [{"titulo": regex}, {"descricao": regex}]}, {"titulo":1,"descricao":1,"url":1,"empresa":1,"site":1,"embedding":1}).limit(100))
        candidate_vagas.extend(docs)
    # se poucos candidatos, pegar algumas vagas gerais (limitado)
    if not candidate_vagas:
        candidate_vagas = list(vagas_col.find({}, {"titulo":1,"descricao":1,"url":1,"empresa":1,"site":1,"embedding":1}).limit(200))
    # processar embeddings e retornar top_k
    return processar_com_embeddings(texto, candidate_vagas, top_k=top_k)

def comparar_por_llm(texto: str):
    # extrai profissões (cache-aware)
    profissoes = extrair_profissao_principal_cached(texto)
    for profissao in profissoes:
        profissao_nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(profissao_nucleo)
        regex = re.compile(re.escape(term), re.IGNORECASE)
        logging.info(f"[LLM] procurando vagas para profissão núcleo: '{profissao_nucleo}' (termo: '{term}')")
        vagas = list(vagas_col.find({
            "$or": [{"titulo": regex}, {"descricao": regex}]
        }).limit(5))
        if len(vagas) == 0:
            logging.info("[LLM] Nenhuma vaga encontrada para essa profissão, tentando próxima...")
            continue
        # aqui processa com LLM (poucas vagas)
        return processar_com_llm(texto, vagas)
    return []

def comparar_misto(texto: str, top_k_emb: int = 10, top_n_llm: int = TOP_N_LLM):
    """
    Pipeline misto:
    - tenta vagas do DB por profissão (usando cache de profissões)
    - se insuficiente -> scrap (guardado em scrap_cache e DB)
    - roda embeddings para rankear (top_k_emb)
    - roda LLM apenas nas top_n_llm vagas
    """
    inicio = time.time()
    # 1) garantir vagas para profissões (busca no DB + scrap)
    vagas = garantir_vagas_para_profissao(texto, min_por_profissao=3)

    if not vagas:
        logging.info("[MIXTO] Nenhuma vaga encontrada após tentativas.")
        return []

    # 2) ranking via embeddings (rápido)
    top_vagas = processar_com_embeddings(texto, vagas, top_k=top_k_emb)

    # 3) chamar LLM só para as N melhores (reduz custo/tempo)
    top_docs_for_llm = [v["_raw_doc"] for v in top_vagas]  # recuperar docs brutos
    final = processar_com_llm(texto, top_docs_for_llm, max_vagas_llm=top_n_llm)

    # 4) compor resultado final: combinar scores do embedding e do llm (se desejar)
    # neste exemplo, usamos o resultado LLM ordenado; caso queira usar ambos, podemos mesclar.
    duracao = time.time() - inicio
    logging.info(f"[MIXTO] processado em {duracao:.2f}s — vagas consideradas: {len(vagas)} — top_emb: {len(top_vagas)} — top_llm: {len(final)}")
    return final

# -----------------------
# WORKERS / ENDPOINTS SIMPLIFICADOS
# -----------------------
def worker_run_once():
    curriculo = curriculos_col.find_one_and_update(
        {"status": "pendente"},
        {"$set": {"status": "processando"}}
    )
    if not curriculo:
        logging.info("[worker] nenhum currículo pendente")
        return {"mensagem": "nenhum currículo pendente"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one({"_id": curriculo["_id"]}, {"$set": {"status": "erro", "resultado": []}})
        return {"erro": "currículo sem texto"}

    resultado = comparar_misto(texto)
    curriculos_col.update_one({"_id": curriculo["_id"]}, {"$set": {"status": "concluido", "resultado": resultado}})
    return {"mensagem": "processado", "id": str(curriculo["_id"]), "total_vagas": len(resultado)}

def worker_comparar_misto(email: str):
    curriculo = curriculos_col.find_one_and_update({"email": email}, {"$set": {"status": "processando"}})
    if not curriculo:
        logging.info("[worker] nenhum currículo pendente")
        return {"mensagem": "nenhum currículo pendente"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one({"_id": curriculo["_id"]}, {"$set": {"status": "erro"}})
        return {"erro": "currículo sem texto"}

    resultado = comparar_misto(texto)
    curriculos_col.update_one({"_id": curriculo["_id"]}, {"$set": {"status": "concluido", "resultado": resultado}})
    return {"mensagem": "processado", "id": str(curriculo["_id"]), "total_vagas": len(resultado)}

# Opcional: função para forçar reextrair profissões (por exemplo se o usuário atualizou o currículo)
def reextrair_e_salvar_profissoes(texto_curriculo: str):
    doc_hash = hash_curriculo(texto_curriculo)
    profs = extrair_profissao_principal(texto_curriculo, max_profissoes=5)
    curriculos_col.update_one({"doc_hash": doc_hash}, {"$set": {"profissoes_detectadas": profs, "last_profession_extract_ts": time.time()}}, upsert=True)
    return profs

# -----------------------
# Se quiser rodar em modo debug
# -----------------------
if __name__ == "__main__":
    sample_text = "Seu currículo de teste aqui: Desenvolvedor Python com experiência em Django, Docker, AWS."
    result = comparar_misto(sample_text)
    print("Resultado (ex):", json.dumps(result, indent=2, ensure_ascii=False))
