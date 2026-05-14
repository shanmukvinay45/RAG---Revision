"""
Hybrid RAG — FastAPI (API-only: no local embedding/reranker model downloads).

- Embeddings: Hugging Face Inference API (BGE-M3 feature extraction)
- Reranker: Hugging Face Inference API (bge-reranker-v2-m3)
- LLM: Groq (LangChain)

Env:
  GROQ_API_KEY   — required
  HF_TOKEN       — required (token must allow Inference Providers)
  GROQ_MODEL     — optional (default from rag_chatbot_groq)

Run:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

First startup indexes docs via HF (one API call per document for embeddings).

Sessions:
  POST /sessions → { "session_id": "..." }
  POST /sessions/{session_id}/chat  body: { "query": "..." }
  GET /sessions/{session_id} → stored history
  DELETE /sessions/{session_id}
"""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_chatbot_groq import (
    DEFAULT_GROQ_MODEL,
    HYBRID_K,
    TOP_K,
    HuggingFaceBgeM3Embeddings,
    answer,
    as_documents,
    build_bm25,
    build_llm_chains,
    build_vectordb,
    hybrid_search,
    rephrase_query,
    rerank_hf,
)

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment or .env")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in environment or .env")

MAX_SESSION_MESSAGES = 10

_sessions: Dict[str, List[Dict[str, str]]] = {}
_sessions_lock = threading.Lock()

_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _state
    groq_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    embedding = HuggingFaceBgeM3Embeddings(hf_token=HF_TOKEN)
    db = build_vectordb(embedding)
    documents = as_documents()
    bm25 = build_bm25(documents)
    hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
    rephrase_chain, answer_chain = build_llm_chains(GROQ_API_KEY, groq_model)
    _state = {
        "db": db,
        "bm25": bm25,
        "hf_client": hf_client,
        "rephrase_chain": rephrase_chain,
        "answer_chain": answer_chain,
        "groq_model": groq_model,
    }
    yield
    _state.clear()


app = FastAPI(
    title="Hybrid RAG API",
    version="1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)


class SessionChatRequest(BaseModel):
    query: str


@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "mode": "api_only",
        "groq_model": _state.get("groq_model", "starting…"),
        "endpoints": {
            "stateless_chat": "POST /chat",
            "create_session": "POST /sessions",
            "session_chat": "POST /sessions/{session_id}/chat",
            "get_history": "GET /sessions/{session_id}",
            "delete_session": "DELETE /sessions/{session_id}",
        },
    }


def _run_rag(query: str, history: List[Dict[str, str]]) -> Tuple[str, str, List[Document]]:
    db = _state["db"]
    bm25 = _state["bm25"]
    hf_client = _state["hf_client"]
    rephrase_chain = _state["rephrase_chain"]
    answer_chain = _state["answer_chain"]

    rewritten_query = rephrase_query(query, history, rephrase_chain)
    candidates = hybrid_search(rewritten_query, db, bm25, k=HYBRID_K)
    top_chunks = rerank_hf(rewritten_query, candidates, hf_client, top_k=TOP_K)
    response = answer(rewritten_query, top_chunks, history, answer_chain)
    return rewritten_query, response, top_chunks


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    rewritten_query, response, top_chunks = _run_rag(req.query, req.history)

    return {
        "query": req.query,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": [
            {
                "id": d.metadata.get("id"),
                "content": d.page_content,
            }
            for d in top_chunks
        ],
        "answer": response,
    }


@app.post("/sessions")
def create_session() -> dict:
    sid = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[sid] = []
    return {"session_id": sid}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        history = list(_sessions[session_id])
    return {"session_id": session_id, "history": history}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    with _sessions_lock:
        removed = _sessions.pop(session_id, None) is not None
    if not removed:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return {"session_id": session_id, "deleted": True}


@app.post("/sessions/{session_id}/chat")
def chat_session(session_id: str, body: SessionChatRequest) -> dict:
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        history = list(_sessions[session_id])

    rewritten_query, response, top_chunks = _run_rag(body.query, history)

    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session expired or deleted")
        _sessions[session_id].append({"role": "user", "content": body.query})
        _sessions[session_id].append({"role": "assistant", "content": response})
        _sessions[session_id] = _sessions[session_id][-MAX_SESSION_MESSAGES:]

    return {
        "session_id": session_id,
        "query": body.query,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": [
            {
                "id": d.metadata.get("id"),
                "content": d.page_content,
            }
            for d in top_chunks
        ],
        "answer": response,
    }
