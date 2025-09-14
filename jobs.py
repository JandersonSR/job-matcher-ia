import requests
import os
from dotenv import load_dotenv
import time
import pymongo
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Carregar variáveis de ambiente
load_dotenv()

# Conexão com MongoDB
MONGO_URI = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
curriculos_col = db["curriculos"]
vagas_col = db["vagas"]
config_col = db["aiConfig"]

# Ex: "https://api-inference.huggingface.co/models/your-model"
HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cliente para Ollama (simula OpenAI API, mas roda local)
llm_client = OpenAI(
  base_url="http://localhost:11434/v1",
  api_key="ollama"  # placeholder, Ollama não exige autenticação
)

RUN_LOCAL_LLM = os.getenv("RUN_LOCAL_LLM", "false").lower = "true"

if RUN_LOCAL_LLM:
  print("⚠️ Rodando LLM local")
else:
  print("✅ LLM local desativado")

def modo_ia():
  """Retorna qual modo de IA usar a partir do MongoDB"""
  config = config_col.find_one({})
  if config:
      return config.get("modoIA", "EMBEDDINGS")
  return "EMBEDDINGS"


def processar_com_embeddings(texto, vagas):
  """Processa currículo e vagas usando embeddings"""
  embedding_curriculo = embedding_model.encode(texto, convert_to_tensor=True)

  resultados = []
  for vaga in vagas:
      descricao = vaga.get("descricao", "")
      if not descricao:
          continue

      embedding_vaga = embedding_model.encode(descricao, convert_to_tensor=True)
      score = util.cos_sim(embedding_curriculo, embedding_vaga).item()

      resultados.append({
          "vaga_id": str(vaga["_id"]),
          "titulo": vaga.get("titulo", "Sem título"),
          "empresa": vaga.get("empresa", "Desconhecida"),
          "compatibilidade": round(float(score), 2)
      })

  return sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)[:5]

def processar_com_llm_local(texto, vagas):
  """Processa currículo e vagas usando LLM (exemplo com OpenAI)"""
  resultados = []
  for vaga in vagas:
    descricao = vaga.get("descricao", "")
    if not descricao:
      continue

    prompt = f"""
    Você é um especialista em RH.
    Compare o seguinte currículo com a vaga e retorne apenas um número entre 0 e 1
    representando a compatibilidade.

    Currículo:
    {texto}

    Vaga:
    {descricao}

    Responda apenas com um número decimal entre 0 e 1.
    """

    try:
      resp = llm_client.chat.completions.create(
        model="llama3",  # modelo baixado no Ollama
        messages=[{"role": "user", "content": prompt}],
        temperature=0
      )

      score_str = resp.choices[0].message.content.strip()
      score = float(score_str)

      resultados.append({
        "vaga_id": str(vaga["_id"]),
        "titulo": vaga.get("titulo", "Sem título"),
        "empresa": vaga.get("empresa", "Desconhecida"),
        "compatibilidade": round(score, 2)
      })
    except Exception as e:
      print(f"⚠️ Erro no LLM: {e}")

  return sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)[:5]

def processar_com_llm_api(texto, vagas):
  """Processa currículo e vagas usando LLM via API gratuita (Hugging Face)"""
  resultados = []

  for vaga in vagas:
    descricao = vaga.get("descricao", "")
    if not descricao:
      continue

    prompt = f"""
Você é um especialista em RH.
Compare o seguinte currículo com a vaga e dê uma pontuação de compatibilidade entre 0 e 1.

Currículo:
{texto}

Vaga:
{descricao}

Responda apenas com um número decimal entre 0 e 1.
"""

  try:
    payload = {"inputs": prompt}
    response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()

    result_text = response.json()

    # Dependendo do modelo da Hugging Face, a resposta pode vir em diferentes formatos
    if isinstance(result_text, list) and len(result_text) > 0:
      # Alguns modelos retornam: [{"generated_text": "..."}]
      text = result_text[0].get("generated_text", "").strip()
    else:
      text = str(result_text).strip()

    # Tenta extrair número
    try:
      score = float(text)
    except:
      score = 0.0

    resultados.append({
      "vaga_id": str(vaga["_id"]),
      "titulo": vaga.get("titulo", "Sem título"),
      "empresa": vaga.get("empresa", "Desconhecida"),
      "compatibilidade": round(score, 2)
    })

  except Exception as e:
    print(f"⚠️ Erro ao chamar LLM API: {e}")
    resultados.append({
      "vaga_id": str(vaga["_id"]),
      "titulo": vaga.get("titulo", "Sem título"),
      "empresa": vaga.get("empresa", "Desconhecida"),
      "compatibilidade": 0.0
    })

  # Ordena do mais compatível para o menos
  return sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)[:5]

def processar_curriculo(curriculo):
  texto = curriculo.get("conteudo", "")
  if not texto:
    return []

  vagas = list(vagas_col.find())
  modo = modo_ia()

  if modo == "LLM_LOCAL" and RUN_LOCAL_LLM:
    print("🤖 Usando LLM local (Ollama/LLaMA3)")
    return processar_com_llm_local(texto, vagas)
  elif modo == "LLM_API":
    print("🌐 Usando LLM via API gratuita")
    return processar_com_llm_api(texto, vagas)
  else:
    print("⚡ Usando Embeddings")
    return processar_com_embeddings(texto, vagas)

def worker_loop():
  """Loop do worker que processa currículos pendentes."""
  while True:
    curriculo = curriculos_col.find_one({"status": "pendente"})
    if curriculo:
      print(f"📄 Processando currículo {curriculo['_id']}...")

      # Extrair dados e calcular compatibilidade
      resultado = processar_curriculo(curriculo)

      # Atualizar no MongoDB
      curriculos_col.update_one(
        {"_id": curriculo["_id"]},
        {"$set": {"status": "concluido", "resultado": resultado}}
      )

      print(f"✅ Currículo {curriculo['_id']} processado.")
    else:
      print("⏳ Nenhum currículo pendente. Aguardando...")
      time.sleep(5)

if __name__ == "__main__":
  worker_loop()
