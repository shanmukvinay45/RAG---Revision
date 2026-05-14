"""
RAGAS Evaluation for RAG Pipeline
===================================
pip install ragas datasets langchain-groq sentence-transformers

.env file needs:
    GROQ_API_KEY=your_groq_key
    HF_TOKEN=your_hf_token          (optional, only if USE_HF_INFERENCE=true)
    USE_HF_INFERENCE=false           (true = HF API embeddings, false = local)
    GROQ_MODEL=llama-3.3-70b-versatile

Run:
    python eval_ragas.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Import your RAG pipeline ──
from RAG import (
    build_vectordb,
    build_bm25,
    as_documents,
    hybrid_search,
    rerank_local,
    rerank_hf,
    HuggingFaceBgeM3Embeddings,
    HF_EMBED_MODEL,
    HF_RERANK_MODEL,
    HYBRID_K,
    TOP_K,
)
from sentence_transformers import CrossEncoder
from huggingface_hub import InferenceClient


# ─────────────────────────── CONFIG ──────────────────────────────────

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN")
USE_HF       = os.environ.get("USE_HF_INFERENCE", "false").lower() in ("1", "true", "yes")

if not GROQ_API_KEY:
    raise EnvironmentError("Missing GROQ_API_KEY in .env")


# ─────────────────────────── GOLDEN DATASET ──────────────────────────
# These are your hand-labeled Q&A pairs.
# ground_truth = what the correct answer should be (you write this once).
# Add more rows as your doc count grows.

GOLDEN_DATA = [
    {
        "question":     "Can I return electronics after 30 days?",
        "ground_truth": "No, electronics must be returned within 30 days of purchase with original packaging and receipt.",
    },
    {
        "question":     "Can I return clothing items?",
        "ground_truth": "Yes, clothing can be returned within 60 days with tags attached and items unworn. Sale items are final sale.",
    },
    {
        "question":     "What happens if my food delivery is spoiled?",
        "ground_truth": "Report within 24 hours with photo evidence to receive a replacement or store credit.",
    },
    {
        "question":     "Is return shipping free for online orders?",
        "ground_truth": "Return shipping is free only for defective items. Non-defective returns have a $5.99 shipping label fee.",
    },
    {
        "question":     "Can I return a personalized product?",
        "ground_truth": "No, customized or personalized products cannot be returned.",
    },
    {
        "question":     "How long does a refund take?",
        "ground_truth": "Refunds are processed within 5-7 business days to the original payment method.",
    },
    {
        "question":     "What if I used a discount code for my purchase?",
        "ground_truth": "Items purchased with a discount code are refunded at the discounted price paid.",
    },
]


# ─────────────────────────── PIPELINE SETUP ──────────────────────────

def setup_pipeline():
    """Initialize embeddings, vectordb, bm25, and reranker."""
    if USE_HF and HF_TOKEN:
        embedding   = HuggingFaceBgeM3Embeddings(hf_token=HF_TOKEN)
        hf_infer    = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
        cross_enc   = None
        print("[+] Using HF Inference API for embeddings & reranker")
    else:
        embedding   = SentenceTransformerEmbeddings(model_name=HF_EMBED_MODEL)
        hf_infer    = None
        cross_enc   = CrossEncoder(HF_RERANK_MODEL)
        print("[+] Using local sentence-transformers for embeddings & reranker")

    documents = as_documents()
    db        = build_vectordb(embedding)
    bm25      = build_bm25(documents)

    return db, bm25, hf_infer, cross_enc


def run_pipeline(question: str, db, bm25, hf_infer, cross_enc) -> tuple[str, list[str]]:
    """
    Run full RAG pipeline for one question.
    Returns: (answer_placeholder, list_of_context_strings)
    
    Note: We return contexts here for RAGAS.
    The actual LLM answer is generated separately by RAGAS eval
    OR you can generate it here using your answer_chain.
    """
    # Hybrid search → 7 candidates
    candidates = hybrid_search(question, db, bm25, k=HYBRID_K)

    # Rerank → top 3
    if hf_infer:
        top_chunks = rerank_hf(question, candidates, hf_infer, top_k=TOP_K)
    else:
        top_chunks = rerank_local(question, candidates, cross_enc, top_k=TOP_K)

    contexts = [doc.page_content for doc in top_chunks]
    return contexts


def generate_answer(question: str, contexts: list[str], answer_chain) -> str:
    """Generate answer using your existing answer_chain from RAG.py."""
    from RAG import build_llm_chains
    context_text = "\n\n---\n\n".join(contexts)
    result = answer_chain.invoke({
        "short_history": "",
        "context":       context_text,
        "query":         question,
    })
    return result.strip()


# ─────────────────────────── BUILD EVAL DATASET ──────────────────────

def build_eval_dataset(golden_data, db, bm25, hf_infer, cross_enc, answer_chain):
    """Run pipeline on all golden questions and collect results."""
    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    print("\n[*] Running pipeline on golden questions...")
    for i, item in enumerate(golden_data, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"    [{i}/{len(golden_data)}] {q}")

        contexts = run_pipeline(q, db, bm25, hf_infer, cross_enc)
        answer   = generate_answer(q, contexts, answer_chain)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(gt)

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })


# ─────────────────────────── RAGAS EVAL ──────────────────────────────

def run_ragas_eval(dataset: Dataset) -> None:
    """Run RAGAS evaluation using Groq as the judge LLM."""

    # Groq as judge LLM
    groq_llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )

    # Local embeddings for answer_relevancy metric
    embed_model = SentenceTransformerEmbeddings(model_name=HF_EMBED_MODEL)

    ragas_llm   = LangchainLLMWrapper(groq_llm)
    ragas_embed = LangchainEmbeddingsWrapper(embed_model)

    print("\n[*] Running RAGAS evaluation (Groq as judge)...")
    print(f"    Judge LLM : {GROQ_MODEL}")
    print(f"    Embeddings: {HF_EMBED_MODEL} (local)\n")

    results = evaluate(
        dataset  = dataset,
        metrics  = [
            faithfulness,       # LLM stuck to context? (no hallucination)
            answer_relevancy,   # Answer actually addresses the question?
            context_precision,  # Retrieved chunks relevant to question?
            context_recall,     # Chunks cover what's needed for answer?
        ],
        llm        = ragas_llm,
        embeddings = ragas_embed,
    )

    # ── Print summary ──
    print("\n" + "="*45)
    print("         RAGAS EVALUATION RESULTS")
    print("="*45)
    print(f"  Faithfulness       : {results['faithfulness']:.3f}  (0=hallucination, 1=grounded)")
    print(f"  Answer Relevancy   : {results['answer_relevancy']:.3f}  (0=off-topic, 1=on-point)")
    print(f"  Context Precision  : {results['context_precision']:.3f}  (0=junk chunks, 1=all relevant)")
    print(f"  Context Recall     : {results['context_recall']:.3f}  (0=missing info, 1=complete)")
    print("="*45)

    # ── Per-question breakdown ──
    print("\n[Per-question breakdown]")
    df = results.to_pandas()
    cols = ["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(df[cols].to_string(index=False))

    # ── Save to CSV ──
    df.to_csv("ragas_results.csv", index=False)
    print("\n[+] Full results saved to ragas_results.csv")

    # ── Flag weak spots ──
    print("\n[!] Weak spots (score < 0.7):")
    weak = False
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        score = results[metric]
        if score < 0.7:
            print(f"    ⚠  {metric}: {score:.3f}  ← needs improvement")
            weak = True
    if not weak:
        print("    ✅ All metrics look good!")


# ─────────────────────────── MAIN ────────────────────────────────────

def main():
    # 1. Setup pipeline
    db, bm25, hf_infer, cross_enc = setup_pipeline()

    # 2. Build answer chain (reuse from RAG.py)
    from RAG import build_llm_chains
    _, answer_chain = build_llm_chains(GROQ_API_KEY, GROQ_MODEL)

    # 3. Build eval dataset by running pipeline on golden questions
    dataset = build_eval_dataset(GOLDEN_DATA, db, bm25, hf_infer, cross_enc, answer_chain)

    # 4. Run RAGAS
    run_ragas_eval(dataset)


if __name__ == "__main__":
    main()