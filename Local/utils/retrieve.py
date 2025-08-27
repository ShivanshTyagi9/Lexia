from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from whoosh.qparser import QueryParser
from whoosh import scoring
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils.config import RAGConfig

Cfg = RAGConfig()

_embedder = None
_reranker = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(Cfg.embed_model, device="cpu")
    return _embedder

def reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(Cfg.reranker_model, device="cpu")
    return _reranker

def dense_search(qc: QdrantClient, query_vec, top_k):
    hits = qc.search(
        collection_name=Cfg.collection,
        query_vector=query_vec,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=Cfg.hnsw_ef_search)
    )
    out = []
    for r in hits:
        p = r.payload
        out.append({
            "id": r.id, "score": float(r.score),
            "doc_id": p["doc_id"], "doc_title": p["doc_title"],
            "section_title": p.get("section_title",""),
            "page_nums": p.get("page_nums", []),
            "chunk_index": p.get("chunk_index", -1),
        })
    return out

def bm25_search(wi, query: str, top_k):
    from whoosh import index as windex
    with wi.searcher(weighting=scoring.BM25F(B=0.75, K1=1.5)) as s:
        qp = QueryParser("content", schema=s.schema)
        q = qp.parse(query)
        results = s.search(q, limit=top_k)
        out = []
        for r in results:
            out.append({
                "id": r["chunk_id"], "score": float(r.score),
                "doc_id": r["doc_id"], "doc_title": r["title"],
                "section_title": r["section"],
                "page_nums": [int(x) for x in r["page"].split(",") if x],
                "chunk_index": int(r["chunk_id"].split(":")[-1]),
            })
        return out

def rrf_fuse(dense, sparse, k=60):
    def ranks(items): return {item["id"]: i for i, item in enumerate(items)}
    rd, rs = ranks(dense), ranks(sparse)
    ids = set(rd) | set(rs)
    fused = []
    for pid in ids:
        score = 1.0/(k + rd.get(pid, 10_000)) + 1.0/(k + rs.get(pid, 10_000))
        meta = next((x for x in dense if x["id"] == pid), None) or next((x for x in sparse if x["id"] == pid), None)
        fused.append({**meta, "rrf": score})
    fused.sort(key=lambda x: x["rrf"], reverse=True)
    return fused

def _load_chunk_text(wi, chunk_id: str) -> str:
    with wi.searcher() as s:
        doc = s.document(chunk_id=str(chunk_id))
        if doc: return doc["content"]
        return ""

def _diverse_head(items, limit):
    seen, out = set(), []
    for it in items:
        if it["doc_id"] in seen: continue
        out.append(it); seen.add(it["doc_id"])
        if len(out) == limit: return out
    for it in items:
        if len(out) == limit: break
        out.append(it)
    return out[:limit]

def retrieve(qc, wi, query: str, final_k=Cfg.final_top_k):
    """
    Hybrid retrieval with robust text loading + table-aware reranking.

    - If a candidate's `id` is a Qdrant UUID (dense side), we derive the Whoosh chunk_id
      as f"{doc_id}:{chunk_index}" so _load_chunk_text() can always fetch content.
    - If a candidate has payload-derived `content_type == "table"` (available on dense hits
      when ingest stored it), we prepend "[TABLE]\\n" to the reranker text for better scoring.
    """
    def _whoosh_id_from_meta(item: Dict[str, Any]) -> str:
        # If id already looks like our whoosh key "doc_id:chunk_index", keep it.
        cid = str(item.get("id", ""))
        if ":" in cid and len(cid.split(":")) == 2:
            return cid
        # Otherwise derive from payload fields (provided by dense_search meta).
        did = item.get("doc_id")
        cidx = item.get("chunk_index")
        if did is not None and cidx is not None and cidx != -1:
            return f"{did}:{cidx}"
        # Fallback: return raw id (may fail to load from Whoosh, but won't crash).
        return cid

    # 1) Dense + Sparse
    qvec = embedder().encode([query], normalize_embeddings=True)[0]
    d = dense_search(qc, qvec, Cfg.dense_top_k)
    s = bm25_search(wi, query, Cfg.bm25_top_k)

    # 2) Fuse (RRf)
    fused = rrf_fuse(d, s, k=60)
    top_for_rerank = fused[:Cfg.rerank_top_k]

    # 3) Build CrossEncoder pairs with table hint (when available)
    pairs = []
    for t in top_for_rerank:
        wid = _whoosh_id_from_meta(t)
        chunk_text = _load_chunk_text(wi, wid)
        # If dense side supplied payload with 'content_type', use it to prefix
        prefix = "[TABLE]\n" if t.get("content_type") == "table" else ""
        pairs.append(
            (query, "[" + t['doc_title'] + "] " + t.get('section_title', "") + "\n" + prefix + chunk_text)
        )

    # 4) Cross-encode reranking
    if pairs:
        scores = reranker().predict(pairs)
        for t, sc in zip(top_for_rerank, scores):
            t["rerank"] = float(sc)
        top_for_rerank.sort(key=lambda x: x["rerank"], reverse=True)

    # 5) Diversity head + final assembly
    final = _diverse_head(top_for_rerank, limit=final_k)
    results = []
    for t in final:
        wid = _whoosh_id_from_meta(t)
        results.append({
            "doc_title": t["doc_title"],
            "section_title": t.get("section_title", ""),
            "pages": t.get("page_nums", []),
            "chunk_index": t.get("chunk_index", -1),
            "chunk_id": wid,  # unified id usable in Whoosh
            "content_type": t.get("content_type", "text"),  # present if it came from dense payload
            "text": _load_chunk_text(wi, wid)
        })
    return results
