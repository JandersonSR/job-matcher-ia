import os
from dotenv import load_dotenv
import time
import pymongo
from sentence_transformers import SentenceTransformer, util

# Carregar variáveis de ambiente
load_dotenv()

# Conexão com MongoDB
MONGO_URI = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
curriculos_col = db["curriculos"]
vagas_col = db["vagas"]

# Modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def processar_curriculo(curriculo):
    """Extrai dados do currículo e calcula matching com vagas."""
    texto = curriculo.get("conteudo", "")
    if not texto:
        return []

    # Embedding do currículo
    embedding_curriculo = model.encode(texto, convert_to_tensor=True)

    resultados = []
    for vaga in vagas_col.find():
        descricao = vaga.get("descricao", "")
        if not descricao:
            continue

        embedding_vaga = model.encode(descricao, convert_to_tensor=True)
        score = util.cos_sim(embedding_curriculo, embedding_vaga).item()

        resultados.append({
            "vaga_id": str(vaga["_id"]),
            "titulo": vaga.get("titulo", "Sem título"),
            "empresa": vaga.get("empresa", "Desconhecida"),
            "compatibilidade": round(float(score), 2)
        })

    # Ordenar do mais compatível para o menos
    resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)

    # Top 5 vagas
    return resultados[:5]

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
