from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, time, json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from utils.config import RAGConfig
from utils.ingest import ensure_whoosh, upsert
from utils.retrieve import retrieve as base_retrieve 
from utils.ingestion_log import update_log

# ---------------- Config & Globals ----------------
Cfg = RAGConfig()
DATA_DIR = os.path.join(getattr(Cfg, "data_dir", "./data"), "pdfs")
load_dotenv()
qdrant_client: QdrantClient | None = None
whoosh_index = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CTX_CHARS", "18000"))

# ---------------- Lifespan ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, whoosh_index

    # Startup
    qdrant_client = QdrantClient(host=Cfg.qdrant_host, port=Cfg.qdrant_port)
    cols = [c.name for c in qdrant_client.get_collections().collections]
    if Cfg.collection not in cols:
        qdrant_client.recreate_collection(
            collection_name=Cfg.collection,
            vectors_config=VectorParams(size=Cfg.dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=Cfg.hnsw_M, ef_construct=Cfg.hnsw_ef_construct),
        )
    whoosh_index = ensure_whoosh(Cfg.index_dir)

    yield

    # Shutdown
    if qdrant_client is not None:
        qdrant_client.close()

# ---------------- App ----------------
app = FastAPI(
    title="Hybrid RAG (Qdrant + BM25 + Reranker)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- Helpers ----------------
def _ensure_ready():
    if qdrant_client is None or whoosh_index is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")

def _fmt_citation(item: Dict[str, Any]) -> str:
    pages = item.get("pages") or item.get("page_nums") or []
    p = f" p.{pages[0]}" if pages else ""
    return f"[{item.get('chunk_id', item.get('id', '?'))}{p}]"

def _build_context(retrieved: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    used = 0
    for it in retrieved:
        header = f"- [{it.get('doc_title','')}] {it.get('section_title','')} • p.{','.join(map(str, it.get('pages', it.get('page_nums', []))))} • chunk_id={it.get('chunk_id', it.get('id',''))}"
        body = (it.get("text") or "").strip()
        seg = header + "\n" + body + "\n\n"
        if used + len(seg) > max_chars:
            break
        parts.append(seg)
        used += len(seg)
    return "".join(parts)

def _generate_answer(question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns {answer, citations, provider}
    Tries OpenAI if OPENAI_API_KEY is set; otherwise uses a deterministic extractive fallback.
    """
    if not contexts:
        return {"answer": "Not found.", "citations": [], "provider": "none"}

    # Try OpenAI
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            system = (
                "You are a careful assistant. Answer using ONLY the supplied passages. "

                "If unknown from passages, say 'Not found.'"
            )
            user = f"Question: {question}\n\nPassages:\n{_build_context(contexts)}\n"

            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.2
            )
            answer = resp.choices[0].message.content.strip()
            cits = [{"chunk_id": c.get("chunk_id", c.get("id","?")),
                     "doc_title": c.get("doc_title",""),
                     "pages": c.get("pages", c.get("page_nums", []))}
                    for c in contexts]
            return {"answer": answer, "citations": cits, "provider": "openai"}
        except Exception as e:
            # fall through to deterministic fallback
            pass

    # Fallback: stitch a concise extract with citations from top 3 chunks
    lines = []
    for it in contexts[:3]:
        text = (it.get("text") or "").strip()
        excerpt = " ".join(text.split()[:80])
        if excerpt:
            lines.append(excerpt + " " + _fmt_citation(it))
    answer = " ".join(lines) if lines else "Not found."
    cits = [{"chunk_id": c.get("chunk_id", c.get("id","?")),
             "doc_title": c.get("doc_title",""),
             "pages": c.get("pages", c.get("page_nums", []))}
            for c in contexts]
    return {"answer": answer.strip(), "citations": cits, "provider": "fallback"}

def _filter_mode(results: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """
    Soft filter for 'text' | 'table' | 'hybrid'.
    If items lack 'content_type', returns as-is.
    """
    if mode not in ("text", "table"):
        return results
    out = [r for r in results if r.get("content_type") == mode]
    return out or results  # graceful fallback

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    _ensure_ready()
    return {
        "ok": True,
        "qdrant": True,
        "whoosh": True,
        "collection": Cfg.collection,
        "openai": bool(OPENAI_API_KEY),
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    _ensure_ready()
    path = os.path.join(DATA_DIR, file.filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()
    return {"filename": file.filename, "message": "Uploaded"}

@app.post("/ingest")
def ingest_documents():
    _ensure_ready()
    new_files = []
    for name in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, name)
        if not os.path.isfile(full):
            continue
        try:
            info = upsert(qdrant_client, whoosh_index, full, doc_title=name)
            new_files.append(name)
        except Exception as e:
            # continue ingesting others but report failures
            new_files.append(f"{name} (FAILED: {e})")

    if new_files:
        try:
            update_log(set([n for n in new_files if "(FAILED" not in n]))
        except Exception:
            pass
    return {"status": "ok", "ingested_files": new_files}

@app.post("/query")
def query_endpoint(payload: dict = Body(...)):
    _ensure_ready()
    question = (payload.get("question") or "").strip()
    k = int(payload.get("k", Cfg.final_top_k))
    mode = (payload.get("mode") or "hybrid").lower()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        hits = base_retrieve(qdrant_client, whoosh_index, question, final_k=k)
        hits = _filter_mode(hits, mode)
        return {"query": question, "k": k, "mode": mode, "contexts": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieve failed: {e}")

@app.post("/answer")
def answer_endpoint(payload: dict = Body(...)):
    _ensure_ready()
    question = (payload.get("question") or "").strip()
    k = int(payload.get("k", Cfg.final_top_k))
    mode = (payload.get("mode") or "hybrid").lower()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        hits = base_retrieve(qdrant_client, whoosh_index, question, final_k=k)
        hits = _filter_mode(hits, mode)
        pack = _generate_answer(question, hits)
        # include retrieved chunks and a tiny latency trace
        return {
            "query": question,
            "mode": mode,
            "k": k,
            "answer": pack["answer"],
            "citations": pack["citations"],
            "provider": pack["provider"],
            "chunks": hits,
            "ts": int(time.time())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answering failed: {e}")

@app.get("/documents")
def documents():
    _ensure_ready()
    docs = {}
    with whoosh_index.searcher() as s:
        for fields in s.all_stored_fields():
            did = fields.get("doc_id"); title = fields.get("title")
            if did and title:
                docs[did] = title
    out = [{"doc_id": k, "title": v} for k, v in docs.items()]
    out.sort(key=lambda x: x["title"].lower())
    return {"ok": True, "documents": out}

@app.get("/delete")
def delete(mode: str = "collection", recreate: bool = True, wipe_whoosh: bool = False):
    """
    Clear Qdrant data used by this RAG.

    Query params:
      - mode: "collection" (drop the collection) | "points" (delete all points only)
      - recreate: when mode="collection", recreate empty collection after delete (default: True)
      - wipe_whoosh: also delete & recreate the Whoosh index directory (default: False)
    """
    global qdrant_client, whoosh_index
    if qdrant_client is None:
        return {"ok": False, "error": "Qdrant client not initialized"}

    result = {"ok": True, "collection": Cfg.collection, "actions": []}

    try:
        if mode == "points":
            # Delete all vectors but keep the collection schema
            qdrant_client.delete(collection_name=Cfg.collection, points_selector={"filter": {}})
            result["actions"].append("qdrant_points_deleted")
        else:
            # Drop the entire collection
            qdrant_client.delete_collection(Cfg.collection)
            result["actions"].append("qdrant_collection_deleted")

            if recreate:
                # Recreate with the same vector config / HNSW settings
                qdrant_client.recreate_collection(
                    collection_name=Cfg.collection,
                    vectors_config=VectorParams(size=Cfg.dim, distance=Distance.COSINE),
                    hnsw_config=HnswConfigDiff(m=Cfg.hnsw_M, ef_construct=Cfg.hnsw_ef_construct),
                )
                result["actions"].append("qdrant_collection_recreated")

        # Optionally wipe Whoosh index, then recreate/open
        if wipe_whoosh:
            try:
                import shutil, os
                idx_dir = Cfg.index_dir
                if os.path.isdir(idx_dir):
                    shutil.rmtree(idx_dir)
                # Recreate empty index
                whoosh_index = ensure_whoosh(Cfg.index_dir)
                result["actions"].append("whoosh_recreated")
            except Exception as e:
                result["ok"] = False
                result["whoosh_error"] = str(e)

        return result

    except Exception as e:
        return {"ok": False, "error": f"Delete failed: {e}"}
