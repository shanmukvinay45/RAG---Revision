"""
RAGAS evaluation for the hybrid RAG pipeline (same logic as rag_chatbot_groq / app).

IMPORTANT: Do not name this file `ragas.py` — it shadows the `ragas` package on import.

Install:
    pip install ragas datasets pandas

Env (.env):
    GROQ_API_KEY
    HF_TOKEN                    required — Inference Providers (embed + rerank + RAGAS judge)
    GROQ_MODEL                  optional

Run from project directory:
    python eval_ragas.py

Outputs:
    ragas_results.csv           per-row scores + inputs
    Console summary             mean metrics + weak spots
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag_chatbot_groq import (
    DEFAULT_GROQ_MODEL,
    HYBRID_K,
    TOP_K,
    HuggingFaceBgeM3Embeddings,
    as_documents,
    build_bm25,
    build_llm_chains,
    build_vectordb,
    hybrid_search,
    rerank_hf,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
HF_TOKEN = os.environ.get("HF_TOKEN")

if not GROQ_API_KEY:
    raise SystemExit("Missing GROQ_API_KEY in .env")
if not HF_TOKEN:
    raise SystemExit(
        "Missing HF_TOKEN in .env. eval_ragas.py is API-only: "
        "HF Inference for index embeddings, reranker, and RAGAS judge embeddings."
    )


# Golden Q&A — extend as you add real policy docs.
GOLDEN_DATA = [
    {
        "question": "Can I return electronics after 30 days?",
        "ground_truth": "No; electronics returns are within 30 days of purchase with original packaging and receipt.",
    },
    {
        "question": "Can I return clothing items?",
        "ground_truth": "Yes, within 60 days with tags attached and unworn; sale items are final sale.",
    },
    {
        "question": "What if my food delivery is spoiled?",
        "ground_truth": "Report within 24 hours with photo evidence for a replacement or store credit.",
    },
    {
        "question": "Is return shipping free for online orders?",
        "ground_truth": "Free only for defective items; non-defective returns have a $5.99 label fee deducted.",
    },
    {
        "question": "Can I return a personalized product?",
        "ground_truth": "No; customized or personalized products are not returnable.",
    },
    {
        "question": "How long does a refund take?",
        "ground_truth": "Refunds process within 5-7 business days to the original payment method.",
    },
    {
        "question": "What if I used a discount code?",
        "ground_truth": "Refunds are at the discounted price actually paid.",
    },
    {
        "question": "Can I return opened video games?",
        "ground_truth": "Opened games are non-refundable; defective discs may be exchanged within 7 days with proof of purchase.",
    },
]


def setup_pipeline():
    """Build Chroma + BM25 + HF Inference client (no local embedding/reranker packages)."""
    embedding = HuggingFaceBgeM3Embeddings(hf_token=HF_TOKEN)
    embedding.embed_query("ping")
    hf_infer = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
    print("[+] RAG index: HF Inference embeddings + HF reranker (eval is API-only)")

    documents = as_documents()
    db = build_vectordb(embedding)
    bm25 = build_bm25(documents)
    return db, bm25, hf_infer


def run_rag_retrieval(question: str, db, bm25, hf_infer: InferenceClient) -> list[str]:
    candidates = hybrid_search(question, db, bm25, k=HYBRID_K)
    top_chunks = rerank_hf(question, candidates, hf_infer, top_k=TOP_K)
    return [d.page_content for d in top_chunks]


def generate_answer(question: str, contexts: list[str], answer_chain) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    return answer_chain.invoke(
        {
            "short_history": "",
            "context": context_text,
            "query": question,
        }
    ).strip()


def build_eval_dataset(
    golden_data, db, bm25, hf_infer: InferenceClient, answer_chain
) -> Dataset:
    questions, answers, contexts_list, ground_truths = [], [], [], []

    print("\n[*] Running your RAG on golden questions...")
    for i, item in enumerate(golden_data, 1):
        q, gt = item["question"], item["ground_truth"]
        print(f"    [{i}/{len(golden_data)}] {q[:70]}...")
        ctxs = run_rag_retrieval(q, db, bm25, hf_infer)
        ans = generate_answer(q, ctxs, answer_chain)
        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctxs)
        ground_truths.append(gt)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )


def run_ragas(dataset: Dataset) -> None:
    judge_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # RAGAS judge embeddings via same HF Inference BGE-M3 (no local MiniLM).
    judge_embed = LangchainEmbeddingsWrapper(
        HuggingFaceBgeM3Embeddings(hf_token=HF_TOKEN)
    )

    print("\n[*] Running RAGAS metrics (Groq judge + HF BGE-M3 for answer_relevancy)...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=judge_embed,
        raise_exceptions=False,
    )

    df = results.to_pandas()
    metric_cols = [
        c
        for c in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
        if c in df.columns
    ]

    print("\n" + "=" * 50)
    print(" RAGAS SUMMARY (row means)")
    print("=" * 50)
    for col in metric_cols:
        m = float(np.nanmean(df[col].astype(float)))
        print(f"  {col:20s}: {m:.3f}")
    print("=" * 50)

    # RAGAS may rename columns when converting to internal schema (e.g. user_input vs question).
    text_col = next(
        (c for c in ("question", "user_input", "query") if c in df.columns),
        None,
    )
    print("\n[Per-question]")
    show_cols = ([text_col] if text_col else []) + metric_cols
    show_cols = [c for c in show_cols if c in df.columns]
    if show_cols:
        print(df[show_cols].to_string(index=False))
    else:
        print("(no text column found; columns:", list(df.columns), ")")
        print(df.to_string(index=False))

    out_path = Path(__file__).resolve().parent / "ragas_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[+] Saved: {out_path}")

    print("\n[!] Metrics below 0.7 (mean):")
    weak = False
    for col in metric_cols:
        m = float(np.nanmean(df[col].astype(float)))
        if m < 0.7:
            print(f"    {col}: {m:.3f}")
            weak = True
    if not weak:
        print("    (none)")


def main() -> None:
    db, bm25, hf_infer = setup_pipeline()
    _, answer_chain = build_llm_chains(GROQ_API_KEY, GROQ_MODEL)
    dataset = build_eval_dataset(GOLDEN_DATA, db, bm25, hf_infer, answer_chain)
    run_ragas(dataset)


if __name__ == "__main__":
    main()
