RAG Revision

Hybrid RAG for policy-style Q&A using BGE-M3 + Chroma for dense retrieval, BM25 for sparse retrieval, fusion-based ranking, Hugging Face reranker, and Groq for answer generation through LangChain. FastAPI service is exposed in app.py and CLI support is available in rag_chatbot_groq.py.

Setup:
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

Copy .env.example to .env and configure:
GROQ_API_KEY
HF_TOKEN (requires Hugging Face Inference Providers access)

Do not commit .env.

Run CLI:
python rag_chatbot_groq.py

First startup re-indexes Chroma and may take a few minutes because of Hugging Face inference calls.

Run API:
uvicorn app:app --reload --host 0.0.0.0 --port 8000

Swagger Docs:
http://127.0.0.1:8000/docs

Docker:
docker build -t rag-api .

docker run --rm -p 8000:8000 -e GROQ_API_KEY=... -e HF_TOKEN=... rag-api

Evaluation:
python eval_ragas.py

Results are saved to:
ragas_results.csv

By default, dependencies are configured for API-only inference (USE_HF_INFERENCE=true).
For local BGE embedding/reranking, install sentence-transformers separately.
