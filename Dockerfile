# API-only service: Groq (LLM) + Hugging Face Inference (embeddings + reranker).
# Pass secrets at run time only — never COPY a .env or bake keys into the image.
#
# Required:  -e GROQ_API_KEY=...  -e HF_TOKEN=...
# Optional:  -e GROQ_MODEL=llama-3.3-70b-versatile
#
# Example:
#   docker build -t rag-api .
#   docker run --rm -p 8000:8000 -e GROQ_API_KEY="$GROQ_API_KEY" -e HF_TOKEN="$HF_TOKEN" rag-api

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY requirements-api.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-api.txt

COPY app.py rag_chatbot_groq.py ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=8s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
