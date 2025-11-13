# jobs.py
import os
import json
import time
import requests
from dotenv import load_dotenv
import pymongo

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL não encontrado no .env")

client = pymongo.MongoClient(MONGO_URL)
db = client[DB_NAME]
curriculos_col = db["curriculos"]
vagas_col = db["vagas"]
config_col = db["aiConfig"]

# Hugging Face API (opcional)
HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"} if HUGGINGFACE_API_KEY else {}

# Embeddings (local)
from sentence_transformers import SentenceTransformer, util
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM local (Ollama) — opcional
from openai import OpenAI
RUN_LOCAL_LLM = os.getenv("RUN_LOCAL_LLM", "false").lower() == "true"
llm_client = None
if RUN_LOCAL_LLM:
    # Assumindo Ollama rodando localmente na porta 11434
    llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    print("⚠️ LLM local habilitado (Ollama)")

def modo_ia():
    config = config_col.find_one({})
    if config:
        return config.get("modoIA", "EMBEDDINGS")
    return "EMBEDDINGS"

def processar_com_embeddings(texto, vagas, top_k=10):
    """
    Retorna top_k vagas (lista de dicts) ordenadas por similaridade por cosine
    """
    if not vagas:
        return []

    embedding_curriculo = embedding_model.encode(texto, convert_to_tensor=True)
    resultados = []

    for vaga in vagas:
        descricao = vaga.get("descricao", "") or vaga.get("titulo", "")
        if not descricao:
            continue
        embedding_vaga = embedding_model.encode(descricao, convert_to_tensor=True)
        score = util.cos_sim(embedding_curriculo, embedding_vaga).item()
        resultados.append({
            "vaga_id": vaga.get("_uid") or str(vaga.get("_id")),
            "titulo": vaga.get("titulo", "Sem título"),
            "empresa": vaga.get("empresa", "Desconhecida"),
            "descricao": descricao,
            "url": vaga.get("url"),
            "site": vaga.get("site", "Desconhecido"),
            "compatibilidade": float(score)
        })

    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)
    # normaliza pontuação para 0..1 (opcional) ou mantemos como está. Mantemos como está e arredondamos depois.
    for r in resultados:
        r["compatibilidade"] = round(r["compatibilidade"], 4)
    return resultados[:top_k]

def _call_huggingface(prompt, timeout=30, retries=2):
    """
    Chamada robusta para HuggingFace Inference API (ou outro endpoint que o usuário passar).
    Retorna string gerada.
    """
    if not HUGGINGFACE_API_URL:
        raise RuntimeError("HUGGINGFACE_API_URL não configurado")
    for attempt in range(retries + 1):
        try:
            resp = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json={"inputs": prompt}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[HF] tentativa {attempt+1} falhou: {e}")
            time.sleep(2 + attempt * 2)
    raise RuntimeError("Hugging Face API falhou após retries")

def _extract_text_from_hf_response(resp_json):
    """
    Extrai texto gerado de diferentes formatos de resposta HF.
    """
    if isinstance(resp_json, list) and len(resp_json) > 0:
        first = resp_json[0]
        if isinstance(first, dict):
            # comum: {"generated_text": "..."}
            return first.get("generated_text") or first.get("generated_text", "") or json.dumps(first)
        # fallback
        return str(first)
    return str(resp_json)

def processar_com_llm_api(texto, vagas):
    """
    Usa LLM via HuggingFace API para analisar cada vaga e retornar JSON com compatibilidade e explicações.
    Recebe vagas já filtradas (ex: top 10 por embeddings).
    """
    resultados = []

    if not vagas:
        return resultados

    for vaga in vagas:
        descricao = vaga.get("descricao", "") or vaga.get("titulo", "")
        prompt = f"""
Você é um especialista em Recrutamento e Seleção.
Avalie a compatibilidade entre o currículo e a vaga. Retorne APENAS um JSON válido com as chaves:
- compatibilidade: número decimal entre 0 e 1
- requisitos_atendidos: lista curta de requisitos encontrados no currículo
- requisitos_nao_atendidos: lista curta de requisitos da vaga que NÃO aparecem no currículo
- melhorias_sugeridas: lista curta de sugestões (cursos, habilidades) reais e relevantes

Currículo:
{texto}

Vaga:
{descricao}

EXEMPLO DE SAÍDA:
{{"compatibilidade": 0.73, "requisitos_atendidos":["Python","REST"], "requisitos_nao_atendidos":["Docker"], "melhorias_sugeridas":["Curso Docker básico"]}}
"""
        try:
            resp_json = _call_huggingface(prompt)
            text = _extract_text_from_hf_response(resp_json)
            # tentar extrair JSON da string gerada
            try:
                # às vezes o modelo retorna texto extra; procurar primeira ocorrência de '{' ... '}'
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    json_text = text[start:end+1]
                else:
                    json_text = text
                parsed = json.loads(json_text)
                score = float(parsed.get("compatibilidade", 0))
                req_ok = parsed.get("requisitos_atendidos", [])
                req_not = parsed.get("requisitos_nao_atendidos", [])
                sugest = parsed.get("melhorias_sugeridas", [])
            except Exception as e:
                print(f"[LLM_API] falha ao parsear JSON da resposta: {e}. Resposta bruta: {text}")
                score, req_ok, req_not, sugest = 0.0, [], [], []

            resultados.append({
                "vaga_id": vaga.get("vaga_id") or vaga.get("vaga_id") or vaga.get("vaga_id"),
                "titulo": vaga.get("titulo"),
                "empresa": vaga.get("empresa"),
                "url": vaga.get("url"),
                "site": vaga.get("site"),
                "compatibilidade": round(float(score), 2),
                "requisitos_atendidos": req_ok,
                "requisitos_nao_atendidos": req_not,
                "melhorias_sugeridas": sugest
            })
        except Exception as e:
            print(f"[LLM_API] erro ao avaliar vaga {vaga.get('titulo')}: {e}")
            resultados.append({
                "vaga_id": vaga.get("vaga_id") or vaga.get("_uid"),
                "titulo": vaga.get("titulo"),
                "empresa": vaga.get("empresa"),
                "compatibilidade": 0.0,
                "requisitos_atendidos": [],
                "requisitos_nao_atendidos": [],
                "melhorias_sugeridas": []
            })

    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)
    return resultados[:5]

def processar_curriculo_dict(curriculo_doc):
    """
    Pipeline híbrido: triagem embeddings -> refinamento LLM API (se configurado).
    Retorna lista de resultados.
    """
    texto = curriculo_doc.get("conteudo", "") or curriculo_doc.get("texto", "")
    if not texto:
        return []

    # pegar todas as vagas do banco
    vagas = list(vagas_col.find())
    # triagem embeddings
    top_vagas = processar_com_embeddings(texto, vagas, top_k=10)
    print(f"[processar_curriculo] triagem com embeddings finalizada — top {len(top_vagas)} vagas")

    modo = modo_ia()
    if modo == "LLM_API" and HUGGINGFACE_API_URL:
        print("[processar_curriculo] refinando com LLM via API (Hugging Face)...")
        return processar_com_llm_api(texto, top_vagas)
    elif modo == "LLM_LOCAL" and RUN_LOCAL_LLM and llm_client:
        # Aqui poderia chamar processar_com_llm_local (não implementado em detalhe)
        print("[processar_curriculo] modo LLM_LOCAL ativado — mas atualmente não implementado aqui")
        return processar_com_embeddings(texto, vagas, top_k=5)  # fallback
    else:
        print("[processar_curriculo] retornando resultado apenas de embeddings (modo EMBEDDINGS)")
        # converte para formato de exibição simples
        return [{
            "vaga_id": v.get("vaga_id") if v.get("vaga_id") else v.get("vaga_id"),
            "titulo": v["titulo"],
            "empresa": v["empresa"],
            "compatibilidade": round(float(v["compatibilidade"]), 2),
            "url": v.get("url"),
            "site": v.get("site")
        } for v in top_vagas]

def worker_run_once():
    """
    Processa um único currículo pendente (status == "pendente").
    Atualiza o documento com status e resultado.
    Retorna mensagem resumo para a rota.
    """
    curriculo = curriculos_col.find_one({"status": "pendente"})
    if not curriculo:
        print("[worker_run_once] nenhum currículo pendente")
        return {"mensagem": "nenhum curriculo pendente"}

    print(f"[worker_run_once] processando currículo id={curriculo['_id']}")
    resultado = processar_curriculo_dict(curriculo)

    curriculos_col.update_one({"_id": curriculo["_id"]}, {"$set": {"status": "concluido", "resultado": resultado}})
    print(f"[worker_run_once] currículo {curriculo['_id']} processado. resultados: {len(resultado)} vagas")
    return {"mensagem": "curriculo processado", "curriculo_id": str(curriculo["_id"]), "top_resultados": len(resultado)}
