import json
import os
import re
import time
from typing import Any, Dict, List
from uuid import uuid4


RULES_FILE = os.getenv("RULES_FILE", "rules.json")

STOPWORDS = {
    "what", "was", "were", "the", "and", "for", "with", "from", "that", "this", "into",
    "between", "how", "much", "total", "about", "which", "had", "did", "does", "not",
    "in", "on", "of", "to", "by", "all", "as", "a", "an", "is", "are", "be", "it",
}


def _terms(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    terms = [t for t in tokens if t not in STOPWORDS]
    # Preserve order and uniqueness.
    return list(dict.fromkeys(terms))[:12]


def sanitize_code(code: str) -> str:
    return re.sub(r"\b\d+(\.\d+)?\b", "<NUM>", code or "")


def load_rules() -> List[Dict[str, Any]]:
    if not os.path.exists(RULES_FILE):
        return []
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_rules(rules: List[Dict[str, Any]]) -> None:
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)


def add_rules(new_rules: List[Dict[str, Any]]) -> None:
    rules = load_rules()
    rules.extend(new_rules)
    save_rules(rules)


def build_rule(question: str, error_description: str, corrected_code: str, wrong_code: str) -> Dict[str, Any]:
    return {
        "id": str(uuid4()),
        "created_at": int(time.time()),
        "enabled": True,
        "trigger_terms": _terms(question or ""),
        "error_description": error_description or "",
        "corrected_code": sanitize_code(corrected_code),
        "wrong_code": sanitize_code(wrong_code),
    }


def match_rules(question: str, min_overlap: int = 2, limit: int = 3) -> List[Dict[str, Any]]:
    q_terms = set(_terms(question or ""))
    if not q_terms:
        return []

    scored = []
    for r in load_rules():
        if not r.get("enabled", True):
            continue
        terms = set(r.get("trigger_terms", []))
        score = len(q_terms & terms)
        if score >= min_overlap:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def format_rules(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return ""

    block = "Learned fixes (use if relevant):\n"
    for r in rules:
        terms = ", ".join(r.get("trigger_terms", []))
        if terms:
            block += f"- Trigger terms: {terms}\n"
        if r.get("error_description"):
            block += f"  Avoid: {r['error_description']}\n"
        if r.get("corrected_code"):
            block += "  Corrected pandas pattern (no numbers):\n"
            block += f"{r['corrected_code']}\n"
    block += "Reminder: use these as patterns only, never copy numeric outputs from memory.\n"
    return block
