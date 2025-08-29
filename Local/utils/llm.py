# utils/llm.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv
load_dotenv()
# ---------- Config via env ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()     # "auto" | "openai" | "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")
OLLAMA_BASE   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CTX_CHARS", "18000"))

# ---------- Public API ----------
def generate_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    *,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Returns: {"answer": str, "citations": List[...], "provider": "openai"|"ollama"|"fallback"}
    """
    if not contexts:
        return {"answer": "Not found.", "citations": [], "provider": "none"}

    provider = _select_provider()

    # Build prompt
    system = (
        "You are a careful assistant. Answer using ONLY the supplied passages. "
        "If the answer isn't in the passages, reply exactly: 'Not found.'"
    )
    user = f"Question: {question}\n\nPassages:\n{_build_context(contexts, max_context_chars)}\n"

    # Try provider
    if provider == "openai":
        try:
            answer = _call_openai(system, user, temperature)
            return {"answer": answer, "citations": _mk_citations(contexts), "provider": "openai"}
        except Exception:
            pass

    if provider in ("ollama", "auto"):
        try:
            answer = _call_ollama(system, user, temperature)
            return {"answer": answer, "citations": _mk_citations(contexts), "provider": "ollama"}
        except Exception:
            pass 

    
    answer = _fallback_answer(contexts)
    return {"answer": answer, "citations": _mk_citations(contexts), "provider": "fallback"}

# ---------- Internals ----------
def _select_provider() -> str:
    if LLM_PROVIDER in ("openai", "ollama"):
        return LLM_PROVIDER
    # auto: prefer OpenAI if key is present, else Ollama, else fallback
    if OPENAI_API_KEY:
        return "openai"
    return "ollama" 

def _call_openai(system: str, user: str, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()

def _call_ollama(system: str, user: str, temperature: float) -> str:
    import requests
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    # schema: {"message": {"content": "..."} , ...}
    msg = (data.get("message", {}) or {}).get("content", "")
    if not msg:
        msgs = data.get("messages") or []
        msg = (msgs[-1]["content"] if msgs else "") or ""
    return msg.strip()

def _build_context(retrieved: List[Dict[str, Any]], max_chars: int) -> str:
    parts, used = [], 0
    for it in retrieved:
        pages = it.get("pages", it.get("page_nums", []))
        header = (
            f"- [{it.get('doc_title','')}] {it.get('section_title','')} • "
            f"p.{','.join(map(str, pages))} • chunk_id={it.get('chunk_id', it.get('id',''))}"
        )
        body = (it.get("text") or "").strip()
        seg = header + "\n" + body + "\n\n"
        if used + len(seg) > max_chars:
            break
        parts.append(seg)
        used += len(seg)
    return "".join(parts)

def _fmt_citation(item: Dict[str, Any]) -> str:
    pages = item.get("pages") or item.get("page_nums") or []
    p = f" p.{pages[0]}" if pages else ""
    return f"[{item.get('chunk_id', item.get('id', '?'))}{p}]"

def _mk_citations(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": c.get("chunk_id", c.get("id", "?")),
            "doc_title": c.get("doc_title", ""),
            "pages": c.get("pages", c.get("page_nums", [])),
        }
        for c in contexts
    ]

def _fallback_answer(contexts: List[Dict[str, Any]]) -> str:
    lines = []
    for it in contexts[:3]:
        text = (it.get("text") or "").strip()
        excerpt = " ".join(text.split()[:80])
        if excerpt:
            lines.append(excerpt + " " + _fmt_citation(it))
    return (" ".join(lines) if lines else "Not found.").strip()
