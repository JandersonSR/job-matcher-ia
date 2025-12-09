import os
import json
import time
import requests
import hashlib
import logging
from typing import List, Tuple, Optional

import torch
from dotenv import load_dotenv
import pymongo
import re
import unicodedata

from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer, util
from webscrapping_vagas_multi import scrap_vagascom

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL não encontrado no .env")

client = pymongo.MongoClient(MONGO_URL)
db = client[DB_NAME]
curriculos_col = db["curriculos"]
vagas_col = db["vagas"]
scrap_cache_col = db["scrap_cache"]

# Hugging Face API
HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"} if HUGGINGFACE_API_KEY else {}

# Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM Local (Ollama)
RUN_LOCAL_LLM = os.getenv("RUN_LOCAL_LLM", "false").lower() == "true"

llm_client = None
if RUN_LOCAL_LLM:
    llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    print("⚠️ LLM local habilitado (Ollama)")

# -----------------------
# CONSTANTES / HYPERPARAMS
# -----------------------
SIM_THRESHOLD = 0.45       # limiar para considerar requisito atendido
TOP_K_EMBEDDINGS = 10      # número de vagas consideradas via embeddings
TOP_N_LLM = 3              # número de vagas que serão avaliadas pelo LLM
SCRAP_MAX_PAGES = 2        # páginas a scrapear quando necessário

# recomenda criar índices (executar uma vez)
try:
    vagas_col.create_index([("titulo", "text"), ("descricao", "text")])
    scrap_cache_col.create_index("term", unique=True)
    curriculos_col.create_index("email", unique=False)
except Exception:
    pass

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
        return palavras[0]

    return palavras_filtradas[0]

def sanitizer_vagas_term(term: str) -> str:
    term = term.strip().lower()
    term = term.replace(" ", "-")
    term = re.sub(r"[^a-z0-9\-]+", "", term)
    return term

# -----------------------
# FUNÇÕES DE EMBEDDINGS
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
def extrair_requisitos(texto_descricao):
    """
    Divide a descrição da vaga em possíveis requisitos com base em quebras de linha e tópicos.
    Retorna cada requisito como um item separado.
    """
    linhas = texto_descricao.split("\n")
    requisitos = []
    for linha in linhas:
        linha = linha.strip("-•* \t").strip()
        if len(linha) > 3:
            # quebra em pequenas habilidades, se houver vírgulas ou pontos
            sub_reqs = [r.strip() for r in re.split(r",|;|\.", linha) if len(r.strip()) > 2]
            requisitos.extend(sub_reqs)
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

# # -----------------------
# # COMPARAR REQUISITOS (usa embeddings de habilidades do currículo)
# # -----------------------

# -----------------------
# LLM helper
# -----------------------

def _call_llm(prompt, timeout=30, retries=1, max_tokens=2500):
    """
    GPT-4o-mini via OpenAI API
    """

    # OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurado")

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for attempt in range(retries + 1):

        try:
            body = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2
            }

            resp = requests.post(url, headers=headers, json=body, timeout=timeout)

            resp.raise_for_status()
            data = resp.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[GPT] Tentativa {attempt + 1} falhou: {e}")
            time.sleep(1 + attempt * 2)

    raise RuntimeError("OpenAI API falhou após múltiplas tentativas")

# -----------------------
# PROCESSAMENTO VIA EMBEDDINGS
# - assume que cada vaga tem embedding salvo no DB (campo embedding)
# - compara currículo com vagas via cosine, sem recalcular embeddings das vagas
# -----------------------
def processar_com_embeddings(texto, vagas, top_k=TOP_K_EMBEDDINGS):
    """
    Processa o currículo contra vagas usando embeddings.
    Retorna top_k vagas com compatibilidade, requisitos atendidos, não atendidos e melhorias.
    """
    if not vagas:
        return []

    # Embedding do currículo
    embedding_curriculo = embedding_model.encode(texto, convert_to_tensor=True)

    # Quebra currículo em pequenas habilidades
    habilidades_emb = [s.strip() for s in texto.split("\n") if len(s.strip()) > 3]

    # 2. Gera EMBEDDINGS das habilidades
    if habilidades_emb:
        habilidades_vecs = embedding_model.encode(habilidades_emb, convert_to_tensor=True)
    else:
        habilidades_vecs = None

    # habilidades_emb = None

    resultados = []

    for vaga in vagas:
        descricao = (vaga.get("titulo", "") + " " + vaga.get("descricao", "")).strip()
        if not descricao:
            continue
        # obter embedding salvo OU criar e persistir
        embedding_vaga = vaga_embedding_from_db(vaga)
        if embedding_vaga is None:
            # calculo e salvamento em background (sincrono aqui)
            emb_vec = embedding_model.encode(descricao, convert_to_tensor=False)
            vagas_col.update_one({"_id": vaga["_id"]}, {"$set": {"embedding": emb_vec.tolist()}})
            embedding_vaga = torch.from_numpy(np.array(emb_vec, dtype=np.float32))

        # Compatibilidade geral
        score = util.cos_sim(embedding_curriculo, embedding_vaga).item()

        # Extrai requisitos da vaga
        req_text_list = extrair_requisitos(descricao)
        req_embeddings = []
        if req_text_list:
            req_vecs = embedding_model.encode(req_text_list, convert_to_tensor=False)
            req_embeddings = [np.array(v, dtype=np.float32) for v in req_vecs]

        requisitos_pairs = list(zip(req_text_list, req_embeddings))

        # Compara requisitos com currículo
        requisitos_atendidos, requisitos_nao_atendidos = comparar_requisitos(
            requisitos_pairs,
            habilidades_vecs
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
            "_raw_doc": vaga
        })

    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)
    return resultados[:top_k]

# -----------------------
# Extrair profissões
# -----------------------

def _extract_text_from_hf_response(resp_json):
    if isinstance(resp_json, list) and resp_json:
        first = resp_json[0]
        if isinstance(first, dict):
            return first.get("generated_text") or json.dumps(first)
        return str(first)
    return str(resp_json)

def extrair_profissao_principal(texto_curriculo: str, max_profissoes: int = 5) -> list[str]:
    """
    Extrai até `max_profissoes` possíveis profissões de um currículo.
    Retorna uma lista de palavras únicas representando profissões.
    """
    prompt = f"""
Extraia no minimo 3 e maximo {max_profissoes} possíveis profissões do currículo abaixo.
Retorne APENAS as palavras representando as profissões, separadas por vírgula,
sem qualquer explicação ou texto adicional.
Se encontrar profissões compostas, retorne duas profissões separadas e correlacionadas (ex: "Chefe de Cozinha" vira "Chefe, Cozinheira").
Currículo:
{texto_curriculo}

Retorne no minimo 3 e no maximo {max_profissoes} profissões, separadas por vírgula:
"""
    resp = _call_llm(prompt, timeout=20, retries=0, max_tokens=400)

    # Normaliza resposta: remove espaços extras e separa por vírgula
    profissoes = [p.strip().lower() for p in resp.split(",") if p.strip()]

    # Garante máximo de max_profissoes
    return profissoes[:max_profissoes]

# from bson import ObjectId

def serializar(v):
    if isinstance(v, list):
        return [serializar(i) for i in v]
    if isinstance(v, dict):
        return {k: serializar(v2) for k, v2 in v.items()}
    return v

def reparar_json(raw):
    """
    Tenta reparar JSON incompleto ou mal formatado retornado pelo LLM.
    """
    import re

    # Remove possíveis markdowns como ```json e ```
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

    # Força fechamento de aspas abertas
    raw = raw.replace("\n", "")

    # Tenta heurística simples: garantir que termina com ] ou }
    if not raw.endswith("]") and "]" in raw:
        raw = raw[:raw.rfind("]")+1]

    return raw


# -----------------------
# PROCESSAR COM LLM (apenas top-N vagas)
# - recebe apenas as melhores vagas de embeddings
# -----------------------

def processar_com_llm(texto, vagas, max_vagas_llm=5):
    """
    Avalia várias vagas em UMA única chamada ao LLM.
    O retorno é uma lista de análises, uma por vaga.
    """

    if not vagas:
        return []

    vagas = vagas[:max_vagas_llm]
    # garantir campos essenciais
    vagas_prepared = []
    for v in vagas:
        vagas_prepared.append({
            "vaga_id": str(v.get("_id") or v.get("vaga_id") or v.get("_uid")),
            "titulo": v.get("titulo", ""),
            "empresa": v.get("empresa", ""),
            "descricao": v.get("descricao") or v.get("titulo", ""),
            "url": v.get("url"),
            "site": v.get("site")
        })

    prompt = f"""
Você é um especialista em Recrutamento e Seleção.

Analise o CURRÍCULO abaixo contra as seguintes {len(vagas_prepared)} vagas.

Para **cada vaga**, siga as regras:

1. Extraia uma lista REAL de requisitos da vaga (somente o que está escrito).
2. Compare cada requisito com o CURRÍCULO.
3. Calcule "compatibilidade" entre 0 e 1 (0 e 1 absolutos nunca devem ser usados; sempre valores como 0.77, 0.53 etc.).
4. NÃO invente requisitos.
5. NÃO invente habilidades do currículo.
6. Retorne **apenas JSON**, sem textos fora do JSON.
7. O retorno deve ser uma lista JSON, com 1 item por vaga.

CURRÍCULO:
{texto}

VAGAS (lista JSON com ID único):
{json.dumps(serializar(vagas_prepared), ensure_ascii=False)}

RESPOSTA ESPERADA (somente JSON):
[
  {{
    "vaga_id": "id_da_vaga_1",
    "compatibilidade": 0.73,
    "requisitos_atendidos": [...],
    "requisitos_nao_atendidos": [...],
    "melhorias_sugeridas": [...]
  }},
  ...
]

NÃO QUEBRE O JSON.
NÃO INTERROMPA O TEXTO NO MEIO.
A saída DEVE ser um JSON completo válido.
Se necessário, RESUMA os requisitos para caber no limite, mas sempre entregue JSON fechado.

"""

    try:
        # chamada ao LLM
        resp_text = _extract_text_from_hf_response(_call_llm(prompt))

        # tentar extrair lista JSON completa
        start = resp_text.find("[")
        end = resp_text.rfind("]")
        if start != -1 and end != -1:
            json_payload = resp_text[start:end+1]
        else:
            raise ValueError("Resposta do LLM não contém lista JSON válida.")

        reparado = json_payload
        # tenta fazer parse
        try:
            lista = json.loads(json_payload)
        except:
            reparado = reparar_json(json_payload)

        try:
            lista = json.loads(reparado)
        except Exception:
            print("[LLM] ERRO FATAL NO JSON:", reparado)
            raise ValueError("JSON inválido mesmo após reparo.")

        resultados = []

        # combinar dados originais
        vaga_map = {v["vaga_id"]: v for v in vagas_prepared}

        for item in lista:
            vid = item.get("vaga_id")
            origem = vaga_map.get(vid, {})

            resultados.append({
                "vaga_id": vid,
                "titulo": origem.get("titulo", ""),
                "empresa": origem.get("empresa", ""),
                "url": origem.get("url"),
                "site": origem.get("site"),
                "compatibilidade": round(float(item.get("compatibilidade", 0)), 2),
                "requisitos_atendidos": item.get("requisitos_atendidos", []),
                "requisitos_nao_atendidos": item.get("requisitos_nao_atendidos", []),
                "melhorias_sugeridas": item.get("melhorias_sugeridas", []),
                "parsed": item,
                "json_text": json.dumps(item, ensure_ascii=False)
            })

        # ordenar pela compatibilidade
        resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)

        return resultados[:5]

    except Exception as e:
        print(f"[LLM] ERRO GERAL: {e}")
        return []

def buscar_vagas_rapido(term: str, limit: int = 50):
    """
    Busca vagas usando índice text em título e descrição.
    Retorna no máximo 'limit' documentos.
    """
    term_sanitizado = term.strip().lower()

    docs = list(
        vagas_col.find(
            {"$text": {"$search": term_sanitizado}},
            {"titulo": 1, "descricao": 1, "url": 1, "empresa": 1, "site": 1, "embedding": 1}
        ).limit(limit)
    )
    return docs


# ============================================================
#  MODOS DE COMPARAÇÃO
# ============================================================
def comparar_por_embeddings(email: str, texto: str, top_k: int = TOP_K_EMBEDDINGS):
    # buscar uma amostra de vagas relevantes (por texto) - aqui usamos text search simples com as profissões
    profs = extrair_profissao_principal_cached(email, texto)
    candidate_vagas = []
    for profissao in profs:
        nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(nucleo)
        # projeção leve (embedding pode existir)
        docs = buscar_vagas_rapido(term, limit=50)
        candidate_vagas.extend(docs)
    # se poucos candidatos, pegar algumas vagas gerais (limitado)
    if not candidate_vagas:
        candidate_vagas = list(vagas_col.find({}, {"titulo":1,"descricao":1,"url":1,"empresa":1,"site":1,"embedding":1}).limit(100))
    # processar embeddings e retornar top_k
    return processar_com_embeddings(texto, candidate_vagas, top_k=top_k)

def comparar_por_llm(email: str, texto: str):
    # extrai profissões (cache-aware)
    profissoes = extrair_profissao_principal_cached(email, texto)
    for profissao in profissoes:
        profissao_nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(profissao_nucleo)
        logging.info(f"[LLM] procurando vagas para profissão núcleo: '{profissao_nucleo}' (termo: '{term}')")

        vagas_profissao =  buscar_vagas_otimizado(term, limit=5)

        vagas = vagas_profissao if len(vagas_profissao) > 0 else garantir_vagas_para_profissao(email, texto, 0)

        if len(vagas) == 0:
            logging.info("[LLM] Nenhuma vaga encontrada para essa profissão, tentando próxima...")
            continue
        # aqui processa com LLM (poucas vagas)
        return processar_com_llm(texto, vagas)
    return []

def comparar_misto(email: str, texto: str, top_k_emb: int = 10, top_n_llm: int = TOP_N_LLM):
    """
    Pipeline misto:
    - tenta vagas do DB por profissão (usando cache de profissões)
    - se insuficiente -> scrap (guardado em scrap_cache e DB)
    - roda embeddings para rankear (top_k_emb)
    - roda LLM apenas nas top_n_llm vagas
    """
    inicio = time.time()
    # 1) garantir vagas para profissões (busca no DB + scrap)

    profissoes = extrair_profissao_principal_cached(email, texto)
    vagas = []
    for profissao in profissoes:
        profissao_nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(profissao_nucleo)
        logging.info(f"[LLM] procurando vagas para profissão núcleo: '{profissao_nucleo}' (termo: '{term}')")

        vagas_profissao =  buscar_vagas_otimizado(term, limit=5)
        print(f"[MIXTO] Vagas encontradas para '{term}': {len(vagas_profissao)}")
        if len(vagas_profissao) > 0 :
            vagas = vagas_profissao
        else:
            vagas = garantir_vagas_para_profissao(email, texto, min_por_profissao=0)


    print(f"[MIXTO] Vagas encontradas para '{term}': {len(vagas)}")
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


# ============================================================
#  WORKER para processar currículos pendentes
# ============================================================
def worker_run_once():
    curriculo = curriculos_col.find_one_and_update(
        {"status": "pendente"},
        {"$set": {"status": "processando"}}
    )
    email = curriculo.get("email") if curriculo else "jandersonrodriguesir@gmail.com"

    if not curriculo:
        print("[worker] nenhum currículo pendente", email)
        return {"mensagem": "nenhum currículo pendente"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro", "resultado": []}}
        )
        return {"erro": "currículo sem texto"}

    # usa pipeline llm como padrão
    resultado = comparar_por_llm(email, texto)

    curriculos_col.update_one(
        {"_id": curriculo["_id"]},
        {"$set": {"status": "concluido", "resultado": resultado}}
    )

    return {
        "mensagem": "processado",
        "id": str(curriculo["_id"]),
        "total_vagas": len(resultado)
    }

def worker_comparar_embeddings(email: str):
    curriculo = curriculos_col.find_one_and_update(
        {"email": email},
        {"$set": {"status": "processando"}}
    )

    if not curriculo:
        print("[worker] nenhum currículo pendente!", email)
        return {"mensagem": "nenhum currículo pendente!"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro"}}
        )
        return {"erro": "currículo sem texto"}

    # usa pipeline misto como padrão
    resultado = comparar_por_embeddings_otimizado(email, texto) or comparar_por_embeddings(email, texto)

    curriculos_col.update_one(
        {"_id": curriculo["_id"]},
        {"$set": {"status": "concluido", "resultado": resultado}}
    )

    return {
        "mensagem": "processado",
        "id": str(curriculo["_id"]),
        "total_vagas": len(resultado)
    }

def worker_comparar_llm(email: str):
    curriculo = curriculos_col.find_one_and_update(
        {"email": email},
        {"$set": {"status": "processando"}}
    )

    if not curriculo:
        print("[worker] nenhum currículo pendente.", email)
        return {"mensagem": "nenhum currículo pendente."}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro"}}
        )
        return {"erro": "currículo sem texto"}

    # usa pipeline misto como padrão
    resultado = comparar_por_llm(email, texto)

    curriculos_col.update_one(
        {"_id": curriculo["_id"]},
        {"$set": {"status": "concluido", "resultado": resultado}}
    )

    return {
        "mensagem": "processado",
        "id": str(curriculo["_id"]),
        "total_vagas": len(resultado)
    }

def worker_comparar_misto(email: str):
    curriculo = curriculos_col.find_one_and_update(
        {"email": email},
        {"$set": {"status": "processando"}}
    )

    if not curriculo:
        print("[worker] nenhum currículo pendente..", email)
        return {"mensagem": "nenhum currículo pendente.."}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")

    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro"}}
        )
        return {"erro": "currículo sem texto"}

    top_vagas = comparar_por_embeddings_otimizado(email, texto, top_k=3, usar_sugestoes=False) or comparar_por_embeddings(email, texto)

    # usa pipeline misto como padrão
    print(f"[MIXTO WORKER] Vagas finais selecionadas: {len(top_vagas) if top_vagas else 0}")
    logging.info(f"[LLM] Processando com LLM - Misto para {len(top_vagas)} vagas")

    if not top_vagas:
        vagas = garantir_vagas_para_profissao(email, texto)
        top_vagas = comparar_por_embeddings_otimizado(email, texto, top_k=3, usar_sugestoes=False) or comparar_por_embeddings(email, texto)

    resultado = processar_com_llm(texto, top_vagas)

    curriculos_col.update_one(
        {"_id": curriculo["_id"]},
        {"$set": {"status": "concluido", "resultado": resultado}}
    )

    return {
        "mensagem": "processado",
        "id": str(curriculo["_id"]),
        "total_vagas": len(resultado)
    }

def worker_scrapping(email: str = ""):
    search_curriculos = {"_id": {"$exists": True}}
    if email:
        search_curriculos["email"] = email

    curriculos = curriculos_col.find(
        search_curriculos
    )
    profissoes = set()
    for c in curriculos:
        profs = c.get("profissoes_detectadas", [])
        for p in profs:
            profissoes.add(p)

    print(f"[SCRAP WORKER] profissões únicas detectadas: {len(profissoes)}")
    for profissao in profissoes:
        profissao_nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(profissao_nucleo)
        logging.info(f"[SCRAP WORKER] procurando vagas para profissão núcleo: '{profissao_nucleo}' (termo: '{term}')")
        scrap_vagascom(term=term, max_pages=SCRAP_MAX_PAGES)


    return { "mensagem": "scrap concluído" }


# Se não houver vagas suficientes para a profissão extraída do currículo,
# realiza scraping adicional no Vagas.com para garantir variedade.
# -----------------------
# GARANTIR VAGAS PARA PROFISSÃO
# - tenta no banco (filtro eficiente), se não houver -> usar scrap_cache -> scrap e inserir no banco
# -----------------------
def garantir_vagas_para_profissao(email: str, texto_curriculo: str, min_por_profissao: int = 5) -> List[dict]:
    profissoes = extrair_profissao_principal_cached(email, texto_curriculo)
    todas_vagas = []

    for profissao in profissoes:
        nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(nucleo)

        # Busca vagas com embedding já calculado e índice text
        docs = buscar_vagas_otimizado(term, limit=20)
        todas_vagas.extend(docs)

        if not todas_vagas:
            cache_regex = re.compile(re.escape(term), re.IGNORECASE)
            # busca no cache de scrapping primeiro (usando term normalizado)
            logging.info(f"[CACHE] Buscando vagas cacheadas para '{cache_regex}'")
            cache = scrap_cache_col.find_one({"term": cache_regex})
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
                logging.info(f"[DB] ELSE Buscando vagas no DB para '{term}'")
                vagas_existentes = buscar_vagas_otimizado(term, limit=50)  # limitar scans

            if len(vagas_existentes) >= min_por_profissao:
                logging.info(f"[OK] Encontradas {len(vagas_existentes)} vagas para '{term}' no DB")
                todas_vagas.extend(vagas_existentes)
                continue

            # se não tem vagas suficientes -> verificar scrap_cache para o termo (se não houve, scrap)
            if cache is None:
                logging.info(f"[SCRAP] Necessário scrap para termo '{term}'")
                # scrap_vagascom espera term (string)
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
# Otimização
# -----------------------
def buscar_vagas_otimizado(term: str, limit: int = 50) -> List[dict]:
    """
    Busca vagas usando índice text em título e descrição, limitado a 'limit' documentos.
    Apenas retorna documentos com embedding já calculado.
    """
    term_sanitizado = term.strip().lower()

    # Busca usando índice text
    docs = list(
        vagas_col.aggregate([
            {"$match": {
                "$text": {"$search": term_sanitizado},
                "embedding": {"$exists": True}
            }},
            {"$group": {
                "_id": "$_uid",      # agrupa por UID
                "doc": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$doc"}},
            {"$limit": limit}
        ])
    )

    if not docs:
        docs = list(
            vagas_col.aggregate([
                {"$match": {
                    "$or": [
                        {"titulo": {"$regex": term_sanitizado, "$options": "i"}},
                        {"descricao": {"$regex": term_sanitizado, "$options": "i"}},
                    ]
                }},
                {"$group": {
                    "_id": "$_uid",
                    "doc": {"$first": "$$ROOT"}
                }},
                {"$replaceRoot": {"newRoot": "$doc"}},
                {"$limit": limit}
            ])
        )

    return docs


def comparar_por_embeddings_otimizado(email: str, texto: str, top_k: int = TOP_K_EMBEDDINGS, usar_sugestoes: bool = True) -> List[dict]:
    """
    Busca vagas relevantes usando $text + embeddings pré-calculadas.
    Retorna as top_k vagas mais compatíveis.
    """

    # Extrair profissões do currículo (cache-aware)
    profissoes = extrair_profissao_principal_cached(email, texto)
    candidate_vagas = []

    for profissao in profissoes:
        nucleo = reduzir_profissao(profissao)

        term = sanitizer_vagas_term(nucleo)

        # Busca vagas com embedding já calculado e índice text
        docs = buscar_vagas_otimizado(term, limit=20)
        candidate_vagas.extend(docs)

    if not candidate_vagas:
        # fallback: pegar vagas gerais com embedding
        candidate_vagas = list(
            vagas_col.find(
                {"embedding": {"$exists": True}},
                {"titulo":1,"descricao":1,"url":1,"empresa":1,"site":1,"embedding":1}
            ).limit(50)
        )

    if not candidate_vagas:
        return []

    # Embedding do currículo
    embedding_curriculo = embedding_model.encode(texto, convert_to_tensor=True)

    # Calcula similaridade e mantém top_k
    vagas_com_scores = []

    unique_ids = set()
    unique_candidate_vagas = []

    for vaga in candidate_vagas:
        uid = f"{vaga.get('titulo','')}-{vaga.get('empresa','')}-{vaga.get('site','')}".lower()

        if not uid:
            continue   # vaga inválida

        if uid in unique_ids:
            continue   # duplicada -> ignora

        unique_ids.add(uid)
        unique_candidate_vagas.append(vaga)

    for vaga in unique_candidate_vagas:
        if not vaga.get("embedding"):
            continue
        emb_vaga = torch.tensor(vaga["embedding"])
        score = util.cos_sim(embedding_curriculo, emb_vaga).item()
        vaga["_score"] = score
        vagas_com_scores.append(vaga)

    # Ordena pelo score e pega top_k
    top_vagas = sorted(vagas_com_scores, key=lambda x: x["_score"], reverse=True)[:top_k]

    if not usar_sugestoes:
            resultados = []
            req_embeddings = []

            for vaga in top_vagas:
                descricao = f"{vaga.get('titulo','')} {vaga.get('descricao','')}".strip()
                req_text_list = extrair_requisitos(descricao)
                if req_text_list:
                    req_vecs = embedding_model.encode(req_text_list, convert_to_tensor=False)
                    req_embeddings = [np.array(v, dtype=np.float32) for v in req_vecs]

                # Quebra currículo em pequenas habilidades
                requisitos_pairs = list(zip(req_text_list, req_embeddings))

                # Compara requisitos com currículo

                resultados.append({
                    "vaga_id": vaga.get("_id"),
                    "titulo": vaga.get("titulo"),
                    "empresa": vaga.get("empresa"),
                    "descricao": descricao,
                    "url": vaga.get("url"),
                    "site": vaga.get("site"),
                    "compatibilidade": round(vaga["_score"], 4),
                    "requisitos": req_text_list,
                    "requisitos_atendidos": vaga.get("requisitos_atendidos", []),
                    "requisitos_nao_atendidos": vaga.get("requisitos_nao_atendidos", []),
                    "melhorias_sugeridas": vaga.get("melhorias_sugeridas", []),
                    "_raw_doc": vaga
                })
            return resultados

    habilidades_emb = [s.strip() for s in texto.split("\n") if len(s.strip()) > 3]

    # 2. Gera EMBEDDINGS das habilidades
    if habilidades_emb:
        habilidades_vecs = embedding_model.encode(habilidades_emb, convert_to_tensor=True)
    else:
        habilidades_vecs = None

    # Extrai requisitos apenas para top_k final
    resultados = []
    req_embeddings = []

    for vaga in top_vagas:
        descricao = f"{vaga.get('titulo','')} {vaga.get('descricao','')}".strip()
        req_text_list = extrair_requisitos(descricao)
        if req_text_list:
            req_vecs = embedding_model.encode(req_text_list, convert_to_tensor=False)
            req_embeddings = [np.array(v, dtype=np.float32) for v in req_vecs]

        # Quebra currículo em pequenas habilidades
        requisitos_pairs = list(zip(req_text_list, req_embeddings))

        # Compara requisitos com currículo
        requisitos_atendidos, requisitos_nao_atendidos = comparar_requisitos(
            requisitos_pairs,
            habilidades_vecs
        )

        melhorias = gerar_melhorias(requisitos_nao_atendidos)

        resultados.append({
            "vaga_id": vaga.get("_id"),
            "titulo": vaga.get("titulo"),
            "empresa": vaga.get("empresa"),
            "descricao": descricao,
            "url": vaga.get("url"),
            "site": vaga.get("site"),
            "compatibilidade": round(vaga["_score"], 4),
            "requisitos": req_text_list,
            "requisitos_atendidos": requisitos_atendidos,
            "requisitos_nao_atendidos": requisitos_nao_atendidos,
            "melhorias_sugeridas": melhorias,
            "_raw_doc": vaga
        })


    return resultados

# -----------------------
# Helper: extrair profissões com cache no curriculo
# -----------------------
def extrair_profissao_principal_cached(email: str, texto_curriculo: str, max_profissoes: int = 5) -> List[str]:
    # tenta achar documento de currículo pelo hash
    curr = curriculos_col.find_one({"email": email}, {"profissoes_detectadas": 1})

    if curr and curr.get("profissoes_detectadas"):
        logging.info("[CACHE] Profissões encontradas no currículo (cache)")
        return curr["profissoes_detectadas"]

    # se não há, extrai via LLM
    profs = extrair_profissao_principal(texto_curriculo, max_profissoes=max_profissoes)
    # salva no documento do currículo (upsert baseado no hash)
    curriculos_col.update_one(
        {"email": email},
        {"$set": {"profissoes_detectadas": profs, "email": email, "last_profession_extract_ts": time.time()}},
        upsert=True
    )
    logging.info("[LLM] Profissões extraídas e salvas no currículo")
    return profs

# Função para forçar reextrair profissões (por exemplo se o usuário atualizou o currículo)
def reextrair_e_salvar_profissoes(email: str, texto_curriculo: str):
    profs = extrair_profissao_principal(texto_curriculo, max_profissoes=5)
    curriculos_col.update_one({"email": email}, {"$set": {"profissoes_detectadas": profs, "last_profession_extract_ts": time.time()}}, upsert=True)
    return profs