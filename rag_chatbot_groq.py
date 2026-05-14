import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# pip install langchain langchain-community langchain-chroma langchain-groq
#     chromadb huggingface_hub python-dotenv numpy rank-bm25
# Optional (CLI / local embeddings + reranker): sentence-transformers

# --------------------------- CONFIG ----------------------------------
CHROMA_PATH = "./chroma_db_bge_m3"
COLLECTION = "company_policies"

# Dense vectors for Chroma (same Hub model; use feature-extraction API).
HF_EMBED_MODEL = "BAAI/bge-m3"
# Reranker via HF Inference text-classification (query + passage in one string).
HF_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Groq chat LLM (rephrase + answer chains only).
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

HYBRID_K = 7
TOP_K = 3
SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ----------------------- SAMPLE DOCS ---------------------------------
DOCS = [
    {
        "id": "doc1",
        "text": (
            "Return Policy – Electronics: Customers may return electronics within 30 days of purchase "
            "with original packaging and receipt. Items must be in working condition. Damaged or tampered "
            "products are not eligible. Refunds are processed within 5-7 business days to the original payment method."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Return Policy – Clothing & Apparel: Clothing items can be returned within 60 days. "
            "Tags must be attached and items unworn. Sale items are final sale and cannot be returned. "
            "Exchange for a different size is allowed within 90 days with receipt."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "Return Policy – Perishables & Food: Perishable items including food, beverages, and fresh "
            "produce cannot be returned once purchased. If an item is defective or spoiled upon delivery, "
            "customers must report within 24 hours with photo evidence to receive a replacement or store credit."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "Return Policy – Online Orders: Online purchases can be returned by mail or in-store within 45 days. "
            "Return shipping is free for defective items. For non-defective returns, a $5.99 shipping label fee "
            "is deducted from the refund. Original shipping charges are non-refundable."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "Refund Process & Exceptions: Refunds are issued to the original payment method. "
            "Store credit is offered if the original payment method is unavailable. "
            "Gift receipts qualify for exchange or store credit only. "
            "Items purchased with a discount code are refunded at the discounted price paid. "
            "No returns are accepted on customized or personalized products."
        ),
    },
    {
        "id": "doc6",
        "text": (
            "Return Policy – Furniture & Large Items: Furniture may be returned within 14 days of delivery "
            "if unassembled and in original packaging. Assembled or used furniture is non-returnable except "
            "for manufacturing defects reported within 48 hours with photos. White-glove delivery fees are non-refundable."
        ),
    },
    {
        "id": "doc7",
        "text": (
            "Return Policy – Jewelry & Watches: Unworn jewelry with tags and certificate of authenticity may be "
            "returned within 30 days. Watches must include all links, box, and warranty card. Engraved or resized "
            "items are final sale. High-value pieces may require manager approval before refund."
        ),
    },
    {
        "id": "doc8",
        "text": (
            "Return Policy – Books & Media: New books and sealed media can be returned within 21 days with receipt. "
            "Opened software, digital codes, and e-books are non-refundable once redeemed or opened. "
            "Defective discs may be exchanged for the same title within 14 days."
        ),
    },
    {
        "id": "doc9",
        "text": (
            "Return Policy – Sports & Outdoor Equipment: Unused equipment in original packaging may be returned "
            "within 45 days. Climbing safety gear and helmets are final sale once packaging is opened for liability "
            "reasons. Bicycles may be returned unridden within 7 days with original receipt only."
        ),
    },
    {
        "id": "doc10",
        "text": (
            "Return Policy – Beauty & Cosmetics: Unopened products may be returned within 30 days. "
            "Opened cosmetics, skincare, and fragrance are non-returnable unless an allergic reaction is "
            "documented by a physician within 72 hours of first use and approved by customer care."
        ),
    },
    {
        "id": "doc11",
        "text": (
            "Return Policy – Toys & Games: Toys must be unopened for return within 30 days. Collectibles and "
            "limited editions are final sale. Video game discs may be exchanged if defective within 7 days "
            "with proof of purchase; opened games are non-refundable."
        ),
    },
    {
        "id": "doc12",
        "text": (
            "Return Policy – Automotive Accessories: Car accessories in original packaging may be returned "
            "within 30 days if unused. Installed parts, fluids, and batteries are non-returnable once installed "
            "or opened. Core charges on batteries are refunded only when an eligible core is returned."
        ),
    },
    {
        "id": "doc13",
        "text": (
            "Return Policy – Home Appliances: Major appliances may be returned within 48 hours of delivery "
            "if unused and in original packaging; after that, only manufacturer warranty applies. "
            "Small countertop appliances may be returned within 30 days unused with receipt."
        ),
    },
    {
        "id": "doc14",
        "text": (
            "Return Policy – Mattresses & Bedding: Mattresses qualify for a 100-night trial only when purchased "
            "with a mattress protector; stains or damage void the trial. Pillows and bedding are returnable "
            "within 30 days if unopened. Adjustable bases are final sale after delivery."
        ),
    },
    {
        "id": "doc15",
        "text": (
            "Return Policy – Software & Digital Goods: Downloadable software and subscription renewals are "
            "non-refundable after delivery or activation. Prepaid cards and gift cards are non-refundable. "
            "Billing disputes must be raised within 60 days of the charge."
        ),
    },
    {
        "id": "doc16",
        "text": (
            "Return Policy – Plants & Garden: Live plants are non-returnable. Defective seeds or bulbs may be "
            "replaced within 14 days with receipt and photo. Tools and planters may be returned unused within 30 days."
        ),
    },
    {
        "id": "doc17",
        "text": (
            "Return Policy – Seasonal & Holiday Items: Seasonal merchandise may be returned until December 31 "
            "if purchased between November 1 and December 24, with receipt and original packaging. "
            "After December 31, seasonal items are final sale."
        ),
    },
    {
        "id": "doc18",
        "text": (
            "Return Policy – Warranties & Extended Plans: Manufacturer warranties are handled by the manufacturer "
            "after the store return window. Extended protection plans are refundable within 30 days of plan purchase "
            "if no claim has been filed; after 30 days plans are non-refundable."
        ),
    },
    {
        "id": "doc19",
        "text": (
            "Return Policy – In-Store Pickup: Items not picked up within 7 days of notification may be restocked "
            "and refunded automatically minus a 10% restocking fee. Pickup orders follow the same return window "
            "as in-store purchases from the pickup date."
        ),
    },
    {
        "id": "doc20",
        "text": (
            "Return Policy – B2B & Wholesale: Business accounts may return eligible stock within 15 days with "
            "RMA approval. Volume purchases may incur restocking fees up to 20%. Net-30 invoices must be current "
            "before credits are issued."
        ),
    },
    {
        "id": "doc21",
        "text": (
            "Return Policy – Damaged in Transit: If packaging shows visible damage at delivery, refuse the package "
            "or note damage with the carrier before signing. Hidden damage must be reported within 48 hours with "
            "photos to qualify for free return shipping and full refund or replacement."
        ),
    },
    {
        "id": "doc22",
        "text": (
            "Return Policy – Price Adjustments: If an item’s price drops within 14 days of purchase, a one-time "
            "price match credit may be issued with receipt. Clearance and flash-sale items are excluded. "
            "Price adjustments are not combined with other promotions."
        ),
    },
    {
        "id": "doc23",
        "text": (
            "Return Policy – Membership & Services: Annual membership fees are refundable within 3 days of "
            "purchase if no member benefits have been used. Installed services and delivery fees are non-refundable "
            "after service completion."
        ),
    },
    {
        "id": "doc24",
        "text": (
            "Return Policy – International Orders: International shipments may be returned within 30 days; "
            "customer pays return shipping and any customs duties unless the item is defective. "
            "Refunds exclude original import taxes paid to third parties."
        ),
    },
    {
        "id": "doc25",
        "text": (
            "Return Policy – Fraud & Abuse: Returns may be denied if patterns suggest abuse, including excessive "
            "returns, mismatch of serial numbers, or missing proof of purchase. The company reserves the right to "
            "limit returns per customer or household at its discretion."
        ),
    },
]


class HuggingFaceBgeM3Embeddings(Embeddings):
    """
    BGE-M3 embeddings via Hugging Face Inference API (feature extraction).
    Chroma needs dense vectors; use feature_extraction, not sentence_similarity.
    """

    def __init__(self, hf_token: str, model: str = HF_EMBED_MODEL) -> None:
        self.model = model
        self._client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )

    def embed_query(self, text: str) -> List[float]:
        raw = self._client.feature_extraction(
            text,
            model=self.model,
            truncate=True,
        )
        return _pool_embedding(raw)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # One request per doc avoids ambiguous batch tensor shapes across providers.
        return [self.embed_query(t) for t in texts]


def _pool_embedding(raw: Any) -> List[float]:
    """Mean-pool token vectors to a single sentence embedding."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim == 2:
        return arr.mean(axis=0).tolist()
    if arr.ndim == 3:
        return arr.mean(axis=1).mean(axis=0).tolist()
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).tolist()


def _local_embedding_and_reranker() -> Tuple[Embeddings, Any]:
    """Local BGE-M3 + CrossEncoder; imports sentence-transformers only when used."""
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from sentence_transformers import CrossEncoder

    embedding = SentenceTransformerEmbeddings(model_name=HF_EMBED_MODEL)
    cross_encoder = CrossEncoder(HF_RERANK_MODEL)
    return embedding, cross_encoder


def as_documents() -> List[Document]:
    """Convert sample records to LangChain Document objects."""
    return [
        Document(page_content=d["text"], metadata={"id": d["id"]})
        for d in DOCS
    ]


def build_vectordb(embedding: Embeddings) -> Chroma:
    """
    Build persistent Chroma DB using a simple ingestion pipeline:
    Chroma.from_documents(...)
    """
    documents = as_documents()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
    )
    return db


def build_bm25(documents: List[Document]) -> BM25Retriever:
    """BM25 retriever over the same documents."""
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = HYBRID_K
    return bm25


def hybrid_search(
    query: str,
    db: Chroma,
    bm25: BM25Retriever,
    k: int = HYBRID_K,
) -> List[Document]:
    """Hybrid retrieval with weighted score fusion: 0.6 semantic + 0.4 BM25."""
    sem_results = db.similarity_search_with_relevance_scores(query, k=k)
    sem_score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc, score in sem_results:
        doc_id = doc.metadata.get("id", doc.page_content[:60])
        sem_score_map[doc_id] = float(max(0.0, min(1.0, score)))
        doc_map[doc_id] = doc

    bm25_scores_raw = bm25.vectorizer.get_scores(query.lower().split())
    if len(bm25_scores_raw) > 0:
        min_b = min(bm25_scores_raw)
        max_b = max(bm25_scores_raw)
        denom = (max_b - min_b) if (max_b - min_b) != 0 else 1.0
    else:
        min_b = 0.0
        denom = 1.0

    bm25_score_map: Dict[str, float] = {}
    for doc, raw_score in zip(bm25.docs, bm25_scores_raw):
        doc_id = doc.metadata.get("id", doc.page_content[:60])
        bm25_score_map[doc_id] = float((raw_score - min_b) / denom)
        doc_map[doc_id] = doc

    candidate_ids = set(sem_score_map.keys()) | set(bm25_score_map.keys())
    fused_scores = {
        doc_id: (
            SEMANTIC_WEIGHT * sem_score_map.get(doc_id, 0.0)
            + BM25_WEIGHT * bm25_score_map.get(doc_id, 0.0)
        )
        for doc_id in candidate_ids
    }

    ranked_ids = sorted(candidate_ids, key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in ranked_ids[:k]]


def _rerank_score_from_classification(out: Any) -> float:
    """Turn HF text_classification output into a single scalar score."""
    if isinstance(out, list):
        if not out:
            return 0.0
        return max(float(getattr(x, "score", 0.0)) for x in out)
    return float(getattr(out, "score", 0.0))


def rerank_hf(
    query: str,
    candidates: List[Document],
    hf_client: InferenceClient,
    top_k: int = TOP_K,
) -> List[Document]:
    """Rerank with BAAI/bge-reranker-v2-m3 via HF Inference text_classification."""
    if not candidates:
        return []

    scores: List[float] = []
    for doc in candidates:
        pair = f"{query.strip()}\n{doc.page_content.strip()}"
        out = hf_client.text_classification(pair, model=HF_RERANK_MODEL)
        scores.append(_rerank_score_from_classification(out))

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


def rerank_local(
    query: str,
    candidates: List[Document],
    cross_encoder: Any,
    top_k: int = TOP_K,
) -> List[Document]:
    """Rerank with local CrossEncoder (same Hub model id, runs on your machine)."""
    if not candidates:
        return []
    pairs = [(query, d.page_content) for d in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    return [doc for doc, _ in ranked[:top_k]]


def build_llm_chains(groq_api_key: str, groq_model: str) -> Tuple[Any, Any]:
    """LCEL: prompt | llm | StrOutputParser(), then chain.invoke({...})."""
    llm = ChatGroq(
        model=groq_model,
        api_key=groq_api_key,
        temperature=0.1,
    )

    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite follow-up user questions to be fully self-contained. "
                "Resolve pronouns such as it, that, they, this using chat history. "
                "Return only the rewritten question.",
            ),
            (
                "human",
                "Conversation:\n{history_text}\n\nQuestion: {question}",
            ),
        ]
    )
    rephrase_chain = rephrase_prompt | llm.bind(max_tokens=120) | StrOutputParser()

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant. "
                "Use only the policy context provided by the user. "
                "If the answer is not present in context, say you do not have that information.",
            ),
            (
                "human",
                "Recent conversation:\n{short_history}\n\n"
                "Policy context:\n{context}\n\n"
                "Question: {query}",
            ),
        ]
    )
    answer_chain = answer_prompt | llm.bind(max_tokens=512) | StrOutputParser()

    return rephrase_chain, answer_chain


def rephrase_query(
    user_q: str,
    history: List[Dict],
    rephrase_chain: Any,
) -> str:
    """Rewrite question into a self-contained query using recent history."""
    if not history:
        return user_q

    recent = history[-4:]
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent
    )

    rewritten = rephrase_chain.invoke(
        {"history_text": history_text, "question": user_q}
    )
    return (rewritten or user_q).strip() or user_q


def answer(
    query: str,
    chunks: List[Document],
    history: List[Dict],
    answer_chain: Any,
) -> str:
    """Generate final answer from retrieved context and chat history."""
    context = "\n\n---\n\n".join([d.page_content for d in chunks]) if chunks else ""

    short_history = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
    )

    return answer_chain.invoke(
        {
            "short_history": short_history,
            "context": context,
            "query": query,
        }
    ).strip()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    use_hf_inference = _env_truthy("USE_HF_INFERENCE")
    hf_token = os.environ.get("HF_TOKEN")
    if use_hf_inference and not hf_token:
        raise EnvironmentError(
            "USE_HF_INFERENCE is enabled but HF_TOKEN is missing. "
            "Use a Hugging Face token that is allowed to call Inference Providers, "
            "or set USE_HF_INFERENCE=false to run BGE-M3 and the reranker locally."
        )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "Missing GROQ_API_KEY (needed for the chat LLM). "
            "Add it to .env or export it in your shell."
        )

    groq_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    rephrase_chain, answer_chain = build_llm_chains(groq_api_key, groq_model)

    hf_infer: InferenceClient | None = None
    cross_encoder: Any = None
    inference_label = ""

    if use_hf_inference and hf_token:
        embedding = HuggingFaceBgeM3Embeddings(hf_token=hf_token)
        try:
            _ = embedding.embed_query("ping")
        except HfHubHTTPError as err:
            status = getattr(getattr(err, "response", None), "status_code", None)
            if status == 403 or "403" in str(err):
                print(
                    "[HF] Inference Providers returned 403 (token missing "
                    "'Make calls to Inference Providers'). Using local BGE-M3 + reranker."
                )
                use_hf_inference = False
                embedding, cross_encoder = _local_embedding_and_reranker()
                inference_label = "local sentence-transformers (HF 403 fallback)"
            else:
                raise
        else:
            hf_infer = InferenceClient(provider="hf-inference", api_key=hf_token)
            inference_label = "HF Inference API"
    else:
        embedding, cross_encoder = _local_embedding_and_reranker()
        inference_label = "local sentence-transformers"

    documents = as_documents()
    db = build_vectordb(embedding)
    bm25 = build_bm25(documents)

    print("\n=== Hybrid RAG (Groq LLM + BGE-M3 + BGE reranker) ===")
    print(f"Embeddings / reranker: {inference_label}")
    print(f"Groq model: {groq_model}")
    print("Type 'quit' to exit.\n")

    history: List[Dict[str, str]] = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        rephrased = rephrase_query(user_input, history, rephrase_chain)
        if rephrased != user_input:
            print(f"[Rephrased] {rephrased}")

        candidates = hybrid_search(rephrased, db, bm25, k=HYBRID_K)
        if hf_infer is not None:
            top_chunks = rerank_hf(rephrased, candidates, hf_infer, top_k=TOP_K)
        else:
            assert cross_encoder is not None
            top_chunks = rerank_local(
                rephrased, candidates, cross_encoder, top_k=TOP_K
            )

        bot_reply = answer(rephrased, top_chunks, history, answer_chain)
        print(f"\nBot: {bot_reply}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": bot_reply})
        history = history[-10:]


if __name__ == "__main__":
    main()
