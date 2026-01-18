# vectordb.py
import os
from typing import List, Dict, Any, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_DIR = "faiss_db"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_emb_instance = None


def _emb() -> HuggingFaceEmbeddings:
    global _emb_instance
    if _emb_instance is None:
        _emb_instance = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return _emb_instance


def _load_db() -> FAISS | None:
    if os.path.exists(DB_DIR):
        return FAISS.load_local(DB_DIR, _emb(), allow_dangerous_deserialization=True)
    return None


def _save_db(db: FAISS) -> None:
    db.save_local(DB_DIR)


def add_mistake(entry: Dict[str, Any]) -> None:
    """
    Store only WRONG cases for cross-session retrieval.
    Keep content compact to avoid over-matching on numbers.
    """
    db = _load_db()

    question = (entry.get("question") or "").strip()
    error_desc = (entry.get("error_description") or "").strip()
    rule = (entry.get("rule") or "").strip()
    corrected_code = (entry.get("corrected_code") or "").strip()
    wrong_code = (entry.get("wrong_code") or "").strip()

    page = (
        "PAST_MISTAKE\n"
        f"QUESTION: {question}\n"
        f"ERROR: {error_desc}\n"
    )
    if rule:
        page += f"RULE: {rule}\n"

    page += (
        "FIX_PATTERN: Follow the corrected code pattern below (do not copy any numbers from memory).\n"
        f"CORRECTED_CODE:\n{corrected_code}\n"
        f"WRONG_CODE (for reference only):\n{wrong_code}\n"
    )

    doc = Document(page_content=page, metadata={"type": "mistake"})

    if db is None:
        db = FAISS.from_documents([doc], _emb())
    else:
        db.add_documents([doc])

    _save_db(db)


def search_similar(
    query: str,
    k: int = 3,
    fetch_k: int = 12,
    min_similarity_score: float = 0.35,
) -> List[str]:
    """
    Retrieve top-k similar past mistakes for prompt injection.
    Uses a simple distance -> similarity heuristic.
    """
    db = _load_db()
    if db is None:
        return []

    pairs = db.similarity_search_with_score(query, k=fetch_k)
    docs = [(doc, float(score)) for doc, score in pairs if doc.metadata.get("type") == "mistake"]
    if not docs:
        return []

    ranked = sorted(docs, key=lambda x: x[1])  # ascending distance
    ranked_sim = [((1.0 / (1.0 + dist)), doc) for doc, dist in ranked]
    filtered = [(sim, doc) for sim, doc in ranked_sim if sim >= min_similarity_score]
    filtered = filtered[:k]
    return [doc.page_content for _, doc in filtered]
