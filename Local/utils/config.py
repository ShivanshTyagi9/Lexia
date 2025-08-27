from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Storage
    collection: str = "notes_papers"
    data_dir: str = "data"
    index_dir: str = "whoosh_index"

    # Models
    embed_model: str = "intfloat/e5-small-v2"        # or "Alibaba-NLP/gte-small"
    reranker_model: str = "BAAI/bge-reranker-base"  # or "-base" for more accuracy

    # Qdrant
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    dim: int = 384  # e5-small/gte-small
    hnsw_M: int = 32
    hnsw_ef_construct: int = 256
    hnsw_ef_search: int = 64

    # Retrieval
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rerank_top_k: int = 30
    final_top_k: int = 8

    # Chunking
    target_tokens_min: int = 400
    target_tokens_max: int = 800
    overlap_ratio: float = 0.18