import os
import json
import time
import requests

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


# ============================================================
#  EMBEDDINGS
# ============================================================

SIM_THRESHOLD = 0.4   # LIMIAR DE SIMILARIDADE PARA CONSIDERAR "ATENDIDO"

def extrair_requisitos(texto_descricao):
    """
    Divide a descrição da vaga em possíveis requisitos com base em quebras de linha e tópicos.
    """
    linhas = texto_descricao.split("\n")
    requisitos = []

    for linha in linhas:
        linha = linha.strip("-•* \t").strip()
        if len(linha) > 3:
            requisitos.append(linha)

    return requisitos


def comparar_requisitos(requisitos, habilidades_curriculo_embeddings):
    requisitos_atendidos = []
    requisitos_nao_atendidos = []

    for req_text, req_emb in requisitos:
        # garante que req_emb é tensor
        if isinstance(req_emb, np.ndarray):
            req_emb = torch.tensor(req_emb)

        # transforma em [1, dim] se necessário
        if req_emb.ndim == 1:
            req_emb = req_emb.unsqueeze(0)

        sims = util.cos_sim(req_emb, habilidades_curriculo_embeddings)
        max_sim = sims.max().item()

        if max_sim >= SIM_THRESHOLD:
            requisitos_atendidos.append(req_text)
        else:
            requisitos_nao_atendidos.append(req_text)

    return requisitos_atendidos, requisitos_nao_atendidos


def gerar_melhorias(requisitos_nao_atendidos):
    """
    Gera textos simples de melhoria com base nos requisitos não atendidos.
    """
    return [
        f"Adicionar experiência com {req.lower()}."
        for req in requisitos_nao_atendidos
    ]


def processar_com_embeddings(texto, vagas, top_k=10):
    """
    Retorna top_k vagas (lista de dicts) ordenadas por similaridade.
    Agora inclui: requisitos_atendidos, requisitos_nao_atendidos, melhorias_sugeridas
    """
    if not vagas:
        return []

    #  Embedding do texto do currículo inteiro
    embedding_curriculo = embedding_model.encode(texto, convert_to_tensor=True)

    #  Quebra o currículo em pequenas frases (habilidades)
    habilidades = [s.strip() for s in texto.split("\n") if len(s.strip()) > 3]
    habilidades_emb = embedding_model.encode(habilidades, convert_to_tensor=True)

    resultados = []

    for vaga in vagas:
        descricao = vaga.get("titulo", "") + " " + vaga.get("descricao", "")

        if not descricao.strip():
            continue

        # Embedding da vaga inteira (compatibilidade geral)
        embedding_vaga = embedding_model.encode(descricao, convert_to_tensor=True)
        score = util.cos_sim(embedding_curriculo, embedding_vaga).item()

        # EXTRAÇÃO DE REQUISITOS
        req_text_list = extrair_requisitos(descricao)

        # Embeddings individuais de cada requisito
        req_embeddings = embedding_model.encode(req_text_list)

        # Associa cada requisito ao embedding
        requisitos_pairs = list(zip(req_text_list, req_embeddings))

        # COMPARAÇÃO COM O CURRÍCULO
        requisitos_atendidos, requisitos_nao_atendidos = comparar_requisitos(
            requisitos_pairs,
            habilidades_emb
        )

        # MELHORIAS
        melhorias = gerar_melhorias(requisitos_nao_atendidos)

        resultados.append({
            "vaga_id": vaga.get("_uid") or str(vaga.get("_id")),
            "titulo": vaga.get("titulo", "Sem título"),
            "empresa": vaga.get("empresa", "Desconhecida"),
            "descricao": descricao,
            "url": vaga.get("url"),
            "site": vaga.get("site", "Desconhecido"),
            "compatibilidade": round(float(score), 4),

            # NOVOS CAMPOS
            "requisitos_atendidos": requisitos_atendidos,
            "requisitos_nao_atendidos": requisitos_nao_atendidos,
            "melhorias_sugeridas": melhorias,
        })

    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)
    return resultados[:top_k]

# ============================================================
#  LLM API (HuggingFace)
# ============================================================
def _call_llm(prompt, timeout=30, retries=1):
    """
    Agora usando GPT-4o-mini via OpenAI API, no formato compatível com chat
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
                "max_tokens": 400,
                "temperature": 0.4
            }

            resp = requests.post(url, headers=headers, json=body, timeout=timeout)

            resp.raise_for_status()
            data = resp.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[GPT] Tentativa {attempt + 1} falhou: {e}")
            time.sleep(1 + attempt * 2)

    raise RuntimeError("OpenAI API falhou após múltiplas tentativas")


def _extract_text_from_hf_response(resp_json):
    if isinstance(resp_json, list) and resp_json:
        first = resp_json[0]
        if isinstance(first, dict):
            return first.get("generated_text") or json.dumps(first)
        return str(first)
    return str(resp_json)

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


def extrair_profissao_principal(texto_curriculo: str) -> str:
    prompt = f"""
Extraia SOMENTE a profissão principal do currículo abaixo.
Retorne apenas UMA palavra representando a profissão, exemplo: 'desenvolvedor', 'analista', 'gerente', 'chefe'.

Currículo:
{texto_curriculo}

Retorne APENAS a profissão:
"""

    resp = _call_llm(prompt)
    profissao = resp.strip().split()[0]  # pega só a primeira palavra
    return profissao

def processar_com_llm(texto, vagas):
    """
    Avaliação usando LLM para cada vaga já filtrada.
    """
    resultados = []

    for vaga in vagas:
        descricao = vaga.get("descricao") or vaga.get("titulo", "")

        prompt = f"""
Você é um especialista em Recrutamento e Seleção.

Analise cuidadosamente a VAGA e extraia uma lista de REQUISITOS a partir dela
(somente itens realmente mencionados no texto).

Depois, compare cada requisito com o CURRÍCULO.

REGRAS IMPORTANTES:
- NÃO invente requisitos. Use apenas o que está escrito na vaga.
- NÃO invente habilidades do currículo.
- Não inclua comentários fora do JSON.
- Sempre calcule compatibilidade entre o currículo e a vaga retornando um número entre 0 e 1, como: 0.8. Nunca enviar 0 nem 1 como compatibilidade.

CURRÍCULO:
{texto}

VAGA:
{descricao}


Retorne APENAS um JSON válido SEM texto fora do JSON. Exemplo de saída:
{{"compatibilidade": 0.73, "requisitos_atendidos":["Python"], "requisitos_nao_atendidos":["Docker"], "melhorias_sugeridas":["Curso Docker básico"]}}
"""

        # fallback seguro para ID da vaga
        vaga_id = vaga.get("vaga_id") or vaga.get("_uid") or None

        try:
            resp_json = _call_llm(prompt)
            raw = _extract_text_from_hf_response(resp_json)

            # tenta extrair trecho JSON
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
                "json_text": json_text,
                "parsed": parsed,
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
            print(f"[LLM] erro ao processar vaga {vaga.get('titulo')}: {e}")

            resultados.append({
                "vaga_id": vaga_id,
                "titulo": vaga.get("titulo"),
                "empresa": vaga.get("empresa"),
                "compatibilidade": 0.0,
                "requisitos_atendidos": [],
                "requisitos_nao_atendidos": [],
                "melhorias_sugeridas": []
            })

    # ordena por compatibilidade
    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)

    return resultados[:5]


# ============================================================
#  MODOS DE COMPARAÇÃO
# ============================================================
def comparar_por_embeddings(texto):
    vagas = list(vagas_col.find())
    return processar_com_embeddings(texto, vagas, top_k=10)


def comparar_por_llm(texto):
    profissao = extrair_profissao_principal(texto)
    profissao_nucleo = reduzir_profissao(profissao)
    term = sanitizer_vagas_term(profissao_nucleo)
    pattern = re.compile(re.escape(term), re.IGNORECASE)

    # Filtra vagas onde título ou descrição contém a profissão
    vagas = list(vagas_col.find({
        "$or": [
            {"titulo": {"$regex": pattern}},
            {"descricao": {"$regex": pattern}}
        ]
    }).limit(5))
    # top_vagas = processar_com_embeddings(texto, vagas, top_k=5)
    return processar_com_llm(texto, vagas)


def comparar_misto(texto):
    profissao = extrair_profissao_principal(texto)
    profissao_nucleo = reduzir_profissao(profissao)
    term = sanitizer_vagas_term(profissao_nucleo)
    pattern = re.compile(re.escape(term), re.IGNORECASE)

    # Filtra vagas onde título ou descrição contém a profissão
    vagas = list(vagas_col.find({
        "$or": [
            {"titulo": {"$regex": pattern}},
            {"descricao": {"$regex": pattern}}
        ]
    }))
    # vagas = list(vagas_col.find())

    if not vagas or len(vagas) < 5:
        print("[IA] Poucas vagas no banco — tentando buscar novas no Vagas.com")

        profissao = extrair_profissao_principal(texto)
        profissao_nucleo = reduzir_profissao(profissao)
        term = sanitizer_vagas_term(profissao_nucleo)

        print(f"[IA] Profissão detectada: {profissao} → núcleo: {profissao_nucleo} → slug: {term}")

        novas = scrap_vagascom(term=term, max_pages=3)

        print(f"[IA] {len(novas)} novas vagas coletadas para '{term}'")

        vagas = list(vagas_col.find())

    top_vagas = processar_com_embeddings(texto, vagas, top_k=5)
    return processar_com_llm(texto, top_vagas)


# ============================================================
#  WORKER para processar currículos pendentes
# ============================================================
def worker_run_once():
    curriculo = curriculos_col.find_one_and_update(
        {"status": "pendente"},
        {"$set": {"status": "processando"}}
    )

    if not curriculo:
        print("[worker] nenhum currículo pendente")
        return {"mensagem": "nenhum currículo pendente"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro", "resultado": []}}
        )
        return {"erro": "currículo sem texto"}

    # usa pipeline misto como padrão
    resultado = comparar_misto(texto)

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
        print("[worker] nenhum currículo pendente")
        return {"mensagem": "nenhum currículo pendente"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro"}}
        )
        return {"erro": "currículo sem texto"}

    # usa pipeline misto como padrão
    resultado = comparar_por_embeddings(texto)

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
        print("[worker] nenhum currículo pendente")
        return {"mensagem": "nenhum currículo pendente"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro"}}
        )
        return {"erro": "currículo sem texto"}

    # usa pipeline misto como padrão
    resultado = comparar_por_llm(texto)

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
        print("[worker] nenhum currículo pendente")
        return {"mensagem": "nenhum currículo pendente"}

    texto = curriculo.get("conteudo") or curriculo.get("texto", "")
    if not texto:
        curriculos_col.update_one(
            {"_id": curriculo["_id"]},
            {"$set": {"status": "erro"}}
        )
        return {"erro": "currículo sem texto"}

    vagas = garantir_vagas_para_profissao(texto)

    # usa pipeline misto como padrão
    top_vagas = processar_com_embeddings(texto, vagas, top_k=5)
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

# Se não houver vagas suficientes para a profissão extraída do currículo,
# realiza scraping adicional no Vagas.com para garantir variedade.
def garantir_vagas_para_profissao(texto_curriculo):
    # 1. Extrair profissão
    profissao = extrair_profissao_principal(texto_curriculo)
    print(f"[PROFISSÃO EXTRAÍDA] {profissao}")

    # 2. Verificar se existe vaga dessa profissão no banco
    vagas_existentes = list(vagas_col.find({"titulo": {"$regex": profissao, "$options": "i"}}))

    if len(vagas_existentes) >= 5:
        print(f"[OK] Já existem {len(vagas_existentes)} vagas para '{profissao}' no banco.")
        return vagas_existentes

    print(f"[SCRAP NECESSÁRIO] Buscando vagas de '{profissao}' no Vagas.com...")
    novas = scrap_vagascom(term=profissao, max_pages=2)

    print(f"[SCRAP FEITO] {len(novas)} vagas novas adicionadas.")

    return list(vagas_col.find({"titulo": {"$regex": profissao, "$options": "i"}}))
