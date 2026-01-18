# main.py
import os
import json
import re
import time
from dotenv import load_dotenv
from rich import print

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

import tools as T
import vectordb
from judge import judge
from chat_memory import ChatState

WORKER_SYSTEM_BASE = """You are a CSV analysis assistant.

Hard rules:
- ALWAYS use tools to compute. Never guess numbers.
- You MUST call pandas_exec for EVERY user question and return a tool call.
- You may call helper tools (search_uniques, search_fsline_l1/l2, value_exists) before pandas_exec,
  but you must always call pandas_exec in the same request.
- For compare/percentage/growth/margin/YoY you MUST use pandas_exec.
- If you are unsure about exact labels (e.g., "Gross Revenue", "Cost of Goods Sold"),
  you MUST call search_uniques on the relevant column (usually "FSLine Statement L2")
  to find the exact value used in the dataset.
- For high-level categories (Net Revenue, Cost of Goods Sold, OPEX, Other Income/Expenses),
  use FSLine Statement L1. Use search_fsline_l1 to confirm labels.
- Product names: do not assume "Product X" exists; verify using value_exists or search_uniques("Product", ...).
- Do not copy any numeric outputs from memory. Memory is patterns only.
- Always answer the current user question; do not repeat a previous answer unless explicitly asked.
- Always produce a structured response:
  1) Final answer
  2) Filters used (Fiscal Year, Fiscal Quarter, Product, Country, Line items)
  3) Method
  4) Sanity check (rows used or brief validation)
"""

# Allow enough iterations for label search + pandas_exec.
MAX_TOOL_ITERS = 6

WORKER_MODEL = "qwen/qwen3-32b"
JUDGE_MODEL = "openai/gpt-oss-120b"

RISK_PATTERNS = [
    r"\byoy\b",
    r"year[- ]over[- ]year",
    r"\bgrowth\b",
    r"\bmargin\b",
    r"\bpercent\b",
    r"\bpercentage\b",
    r"\bcompare\b",
    r"\bdifference\b",
    r"\bdelta\b",
    r"\brate\b",
    r"\baverage\b",
    r"\bmedian\b",
    r"\bnet revenue\b",
]

def is_high_risk_question(q: str) -> bool:
    s = (q or "").lower()
    return any(re.search(p, s) for p in RISK_PATTERNS)


def safe_llm_invoke(llm, messages, retries: int = 1):
    last_err = None
    tool_fix = (
        "Your last tool call failed because the arguments were not valid JSON. "
        "If you call a tool, return a well-formed JSON object for arguments."
    )
    def ensure_tool_fix_message():
        for msg in messages:
            if isinstance(msg, SystemMessage) and tool_fix in (msg.content or ""):
                return
        messages.append(SystemMessage(content=tool_fix))

    for _ in range(max(0, retries) + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            msg = str(e)
            if "tool_use_failed" in msg or "Failed to parse tool call arguments" in msg:
                ensure_tool_fix_message()
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 or "429" in str(e):
                time.sleep(1.5)
    raise last_err  # type: ignore


def should_store(decision: dict, corrected_out: dict) -> bool:
    if decision.get("error") != "yes":
        return False
    if not decision.get("corrected_code"):
        return False
    if not corrected_out.get("ok", False):
        return False

    if decision.get("error_type") == "runtime":
        desc = (decision.get("error_description") or "").lower()
        allow = ["unknown column", "missing column", "keyerror", "column not found"]
        return any(x in desc for x in allow)

    return decision.get("error_type") in {"semantic", "logic"}


def extract_rules_from_retrieval(retrieved_docs: list[str]) -> str:
    rules = []
    patterns = []
    for doc in retrieved_docs:
        for line in doc.splitlines():
            line = line.strip()
            if line.startswith("RULE:"):
                rules.append(line.replace("RULE:", "").strip())
        if not any("RULE:" in x for x in doc.splitlines()):
            for line in doc.splitlines():
                line = line.strip()
                if line.startswith("ERROR:"):
                    rules.append("Avoid: " + line.replace("ERROR:", "").strip())
                    break
        if "CORRECTED_CODE:" in doc:
            lines = doc.splitlines()
            start = None
            for i, line in enumerate(lines):
                if line.strip().startswith("CORRECTED_CODE:"):
                    start = i + 1
                    break
            if start is not None:
                buf = []
                for line in lines[start:]:
                    if line.strip().startswith("WRONG_CODE"):
                        break
                    buf.append(line)
                code = "\n".join(buf).strip()
                if code:
                    code = re.sub(r"\b\d+(\.\d+)?\b", "<NUM>", code)
                    patterns.append(code)

    rules = [r for r in rules if r][:6]
    patterns = [p for p in patterns if p][:2]
    if not rules and not patterns:
        return ""

    block = "Learned rules (MUST follow these patterns):\n"
    for i, r in enumerate(rules, 1):
        block += f"{i}. {r}\n"
    if patterns:
        block += "Example corrected pandas patterns (do not copy numbers):\n"
        for i, p in enumerate(patterns, 1):
            block += f"Pattern {i}:\n{p}\n"
    block += "Reminder: use these as patterns only, never copy numeric outputs from memory.\n"
    return block


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing in .env")

    debug_tools = os.getenv("DEBUG_TOOLS", "0").strip() == "1"

    worker = ChatGroq(
        model=WORKER_MODEL,
        temperature=0.2,
        groq_api_key=api_key,
    )

    @tool
    def get_dataset_card() -> dict:
        """Compact dataset summary (schema, dtypes, some uniques, head)."""
        return T.dataset_card()

    @tool
    def get_uniques(column: str, limit: int = 50) -> dict:
        """Return unique values for a column."""
        return T.get_uniques(column, limit)

    @tool
    def search_uniques(column: str, contains: str = "", limit: int = 50) -> dict:
        """Search unique values in a column by substring (case-insensitive)."""
        return T.search_uniques(column, contains, limit)

    @tool
    def value_exists(column: str, value: str) -> dict:
        """Check if a value exists in a column."""
        return T.value_exists(column, value)

    @tool
    def filter_count(filters: dict) -> dict:
        """Count rows matching filters. Supports list values and ranges."""
        return T.filter_count(filters)

    @tool
    def pandas_exec(code: str) -> dict:
        """Run safe pandas code; must set `result`."""
        return T.run_pandas(code)
    
    @tool
    def search_fsline_l2(query: str, limit: int = 10) -> dict:
        """Search FSLine Statement L2 values (deterministic substring search)."""
        return T.search_fsline_l2(query, limit)

    @tool
    def search_fsline_l1(query: str, limit: int = 10) -> dict:
        """Search FSLine Statement L1 values (deterministic substring search)."""
        return T.search_fsline_l1(query, limit)

    worker = worker.bind_tools([
        get_dataset_card,
        get_uniques,
        search_uniques,
        value_exists,
        filter_count,
        pandas_exec,
        search_fsline_l1,
        search_fsline_l2,
    ])

    print(f"[bold green]CSV Bot[/bold green] (Worker: {WORKER_MODEL} | Judge: {JUDGE_MODEL})")
    print("Type 'exit' to quit.\n")

    state = ChatState(system_base=WORKER_SYSTEM_BASE)
    state.init()
    pending_mistakes = []
    disable_write = os.getenv("VDB_DISABLE_WRITE", "0").strip() == "1"

    def run_tool_loop(resp):
        nonlocal last_code, last_tool_result, dataset_card
        tool_iters = 0
        while getattr(resp, "tool_calls", None):
            tool_iters += 1
            if tool_iters > MAX_TOOL_ITERS:
                fail_msg = (
                    f"I started looping on tool calls and stopped to avoid hanging. "
                    f"(max iterations={MAX_TOOL_ITERS})\n"
                    "Tip: The model is probably failing to find the exact dataset label for a metric. "
                    "Try asking with exact FSLine names or enable DEBUG_TOOLS=1 to see the tool loop.\n"
                )
                state.add_ai(fail_msg)
                print("\n[bold cyan]Bot:[/bold cyan]")
                print(fail_msg)
                return None

            state.messages.append(resp)

            for call in resp.tool_calls:
                name = call["name"]
                args = call["args"]

                if debug_tools:
                    print(f"\n[bold magenta]TOOL CALL[/bold magenta] {name} args={args}")

                try:
                    if name == "get_dataset_card":
                        out = get_dataset_card.invoke(args)
                        dataset_card = out

                    elif name == "pandas_exec":
                        last_code = args.get("code", "")
                        out = pandas_exec.invoke(args)
                        last_tool_result = out

                    elif name == "get_uniques":
                        out = get_uniques.invoke(args)

                    elif name == "search_uniques":
                        out = search_uniques.invoke(args)

                    elif name == "value_exists":
                        out = value_exists.invoke(args)

                    elif name == "filter_count":
                        out = filter_count.invoke(args)

                    elif name == "search_fsline_l2":
                        out = search_fsline_l2.invoke(args)

                    elif name == "search_fsline_l1":
                        out = search_fsline_l1.invoke(args)

                    else:
                        out = {"ok": False, "error_type": "TOOL_INPUT", "error": "Unknown tool"}

                except Exception as e:
                    out = {"ok": False, "error_type": "TOOL_EXEC", "error": f"{type(e).__name__}: {e}"}

                if debug_tools:
                    print(f"[bold magenta]TOOL OUT[/bold magenta] {name} -> {out}")

                state.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(out)})

            resp = safe_llm_invoke(worker, state.messages, retries=1)

        return resp

    def force_pandas_exec():
        nonlocal last_code, last_tool_result, dataset_card
        force = (
            "You must call pandas_exec to answer this question. "
            "Return a tool call with valid JSON args. Do not answer in prose."
        )
        force_msg = SystemMessage(content=force)
        state.messages.append(force_msg)
        resp = safe_llm_invoke(worker, state.messages, retries=1)
        state.messages.remove(force_msg)

        last_code = ""
        last_tool_result = {}
        dataset_card = {}
        return run_tool_loop(resp)

    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        q_lower = q.lower()
        if q_lower == "exit" or q_lower.startswith("exit "):
            print("\n[bold cyan]Bot:[/bold cyan] Exiting and saving learning (if any).")
            break

        retrieved = vectordb.search_similar(q, k=3)
        learned_rules_block = extract_rules_from_retrieval(retrieved)

        state.rebuild_system_message()
        system_text = state.system_prompt()
        if learned_rules_block:
            system_text = system_text + "\n\n" + learned_rules_block
        state.messages[0] = SystemMessage(content=system_text)

        state.add_user(q)

        last_code = ""
        last_tool_result = {}
        dataset_card = {}
        resp = safe_llm_invoke(worker, state.messages, retries=1)
        resp = run_tool_loop(resp)

        if resp is None:
            continue

        if not last_code:
            resp = force_pandas_exec()
            if resp is None:
                continue

        worker_answer = resp.content
        state.add_ai(worker_answer)

        if not dataset_card:
            dataset_card = T.dataset_card()

        need_judge = False
        if last_tool_result and (last_tool_result.get("ok") is False):
            need_judge = True
        if is_high_risk_question(q):
            need_judge = True
        if last_code:
            need_judge = True

        if need_judge:
            decision = judge(
                groq_api_key=api_key,
                dataset_card=dataset_card,
                user_question=q,
                worker_code=last_code,
                worker_tool_result=last_tool_result,
            )
            if decision.get("error") == "yes" and decision.get("corrected_code"):
                corrected_out = T.run_pandas(decision["corrected_code"])
                if should_store(decision, corrected_out):
                    pending_mistakes.append(
                        {
                            "question": q,
                            "wrong_code": last_code,
                            "error_description": f"{decision.get('error_type')}: {decision.get('error_description')}",
                            "rule": (decision.get("error_description") or "").strip(),
                            "corrected_code": decision["corrected_code"],
                        }
                    )
                    print("\n[bold yellow]Learning event:[/bold yellow] Queued fix for FAISS.")

        if not last_code:
            guard = (
                "I need to run a pandas query to answer this question, but no pandas code was executed. "
                "Please re-ask the question so I can compute it using tools."
            )
            state.add_ai(guard)
            print("\n[bold cyan]Bot:[/bold cyan]")
            print(guard)
            print()
            continue

        print("\n[bold cyan]Bot:[/bold cyan]")
        print(worker_answer)
        print()

    if pending_mistakes and not disable_write:
        print(f"\n[bold yellow]Learning event:[/bold yellow] Storing {len(pending_mistakes)} fix(es) in FAISS...")
        for entry in pending_mistakes:
            vectordb.add_mistake(entry)
        print(f"[bold yellow]Learning event:[/bold yellow] Stored {len(pending_mistakes)} fix(es) in FAISS.")
    elif pending_mistakes and disable_write:
        print(f"\n[bold yellow]Learning event:[/bold yellow] Skipped storing {len(pending_mistakes)} fix(es) (VDB_DISABLE_WRITE=1).")
    else:
        print("\n[bold yellow]Learning event:[/bold yellow] No queued fixes to store.")



if __name__ == "__main__":
    main()
