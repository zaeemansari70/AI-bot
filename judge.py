# judge.py
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

JUDGE_SYSTEM = """You are a strict evaluator of pandas-based CSV analysis.

You will be given:
- dataset_card (schema + important uniques)
- user_question
- worker_code (pandas code attempt)
- worker_tool_result (output dict from running the code, or error)

Return ONLY valid JSON:
{
  "error": "yes" | "no",
  "error_type": "semantic" | "logic" | "runtime" | "none",
  "error_description": "string",
  "corrected_code": "string"
}

Rules:
- If the question references a product/country/line item not in dataset_card uniques, error=yes semantic.
- Prefer Amount in USD for money unless user explicitly asks local.
- If worker_tool_result.ok is false, error=yes runtime.
- If error=yes, provide corrected pandas code that sets `result`.
"""

def judge(
    groq_api_key: str,
    dataset_card: dict,
    user_question: str,
    worker_code: str,
    worker_tool_result: dict,
    model: str = "openai/gpt-oss-120b",
    dataset_context: str | None = None,
) -> dict:
    llm = ChatGroq(
        model=model,
        temperature=0,
        groq_api_key=groq_api_key,
    )

    system_text = JUDGE_SYSTEM
    if dataset_context:
        system_text = system_text + "\nDataset context:\n" + dataset_context + "\n"

    payload = {
        "dataset_card": dataset_card,
        "user_question": user_question,
        "worker_code": worker_code,
        "worker_tool_result": worker_tool_result,
    }

    resp = llm.invoke(
        [
            SystemMessage(content=system_text),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )

    try:
        return json.loads(resp.content)
    except Exception:
        return {
            "error": "yes",
            "error_type": "semantic",
            "error_description": "Judge returned invalid JSON.",
            "corrected_code": "",
        }
