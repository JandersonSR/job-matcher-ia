# jobs_runner.py
import os
import pymongo
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import traceback

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "jobmatcher")

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
curriculos_col = db["curriculos"]
vagas_col = db["vagas"]
config_col = db["aiConfig"]

# modelo de embeddings (pré-carregado)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_modo_ia():
    cfg = config_col.find_one({})
    if cfg:
        return cfg.get("modoIA", "EMBEDDINGS")
    return "EMBEDDINGS"

def processar_um_curriculo(curriculo_doc):
    """
    Processa um documento de currículo (single run).
    Retorna o resultado que será salvo.
    """
    try:
        texto = curriculo_doc.get("conteudo", "")
        if not texto:
            return []

        # usando embeddings
        emb_curr = embedding_model.encode(texto, convert_to_tensor=True)

        resultados = []
        for vaga in vagas_col.find():
            descricao = vaga.get("descricao", "")
            if not descricao:
                continue
            emb_vaga = embedding_model.encode(descricao, convert_to_tensor=True)
            score = util.cos_sim(emb_curr, emb_vaga).item()
            resultados.append({
                "vaga_id": str(vaga["_id"]),
                "titulo": vaga.get("titulo", "Sem título"),
                "empresa": vaga.get("empresa", "Desconhecida"),
                "compatibilidade": round(float(score), 4)
            })
        resultados = sorted(resultados, key=lambda x: x["compatibilidade"], reverse=True)[:10]
        return resultados
    except Exception as e:
        print("Erro no processamento do currículo:", e)
        traceback.print_exc()
        return []

def process_pending_once():
    """
    Pega um currículo pendente e processa (uma execução).
    Retorna dict com info do job (id, status, results) para logs.
    """
    curr = curriculos_col.find_one_and_update(
        {"status": "pendente"},
        {"$set": {"status": "processing", "processing_at": pymongo.datetime.datetime.utcnow()}},
        return_document=pymongo.ReturnDocument.AFTER
    )
    if not curr:
        return {"ok": False, "msg": "Nenhum currículo pendente"}

    print(f"Processando currículo {curr['_id']}")
    results = processar_um_curriculo(curr)
    curriculos_col.update_one(
        {"_id": curr["_id"]},
        {"$set": {"status": "concluido", "resultado": results, "updated_at": pymongo.datetime.datetime.utcnow()}}
    )
    return {"ok": True, "id": str(curr["_id"]), "count_results": len(results)}
