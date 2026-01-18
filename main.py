# main.py
import os
import json
import re
import time
from dotenv import load_dotenv
from rich import print

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

import tools as T
import vectordb
from judge import judge
from chat_memory import ChatState

WORKER_SYSTEM_BASE = """You are a CSV analysis assistant.

Hard rules:
- ALWAYS use tools to compute. Never guess numbers.
- For computational questions, you MUST call pandas_exec and return a tool call.
- If a required entity (product/country/line item) does not exist, verify via value_exists/search_uniques
  and respond with a structured error without calling pandas_exec.
- You may call helper tools (search_uniques, search_fsline_l1/l2, value_exists) before pandas_exec,
  but if no validation error is found you must call pandas_exec in the same request.
- For compare/percentage/growth/margin/YoY you MUST use pandas_exec.
- For dataset metadata (rows/columns/schema), call get_dataset_card; pandas_exec is not required.
- If you are unsure about exact labels (e.g., "Gross Revenue", "Cost of Goods Sold"),
  you MUST call search_uniques on the relevant column (usually "FSLine Statement L2")
  to find the exact value used in the dataset.
- Use the dataset context below; do not invent column names or values.
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
MAX_TOOL_ITERS = 10

WORKER_MODEL = "qwen/qwen3-32b"
JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_HISTORY_MESSAGES = 18

ANSWER_SYSTEM = """You are a CSV analysis assistant.

Use the provided computed result exactly. Do not call tools or change numbers.
Return a structured response with:
1) Final answer
2) Filters used
3) Method
4) Sanity check
If the computed result is a string error message, treat it as the final answer and explain the validation."""


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
    return True


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


def trim_history(state: ChatState, keep_last: int) -> None:
    if keep_last < 2:
        return
    if len(state.messages) <= keep_last + 1:
        return
    system = state.messages[0]
    tail = state.messages[-keep_last:]
    state.messages = [system] + tail


def _format_list(values: list[str], limit: int = 30) -> str:
    if not values:
        return ""
    vals = [str(v) for v in values]
    suffix = " ..." if len(vals) > limit else ""
    return ", ".join(vals[:limit]) + suffix


def dataset_context_text(card: dict) -> str:
    if not card:
        return ""
    lines = ["Dataset context (use exact column names/values):"]
    cols = card.get("columns") or []
    if cols:
        lines.append("Columns: " + ", ".join(cols))
    if "fiscal_year_min" in card and "fiscal_year_max" in card:
        lines.append(f"Fiscal Years: {card['fiscal_year_min']} - {card['fiscal_year_max']}")
    for label, key, limit in [
        ("Products", "unique_Product", 20),
        ("Countries", "unique_Country", 20),
        ("Fiscal Quarters", "unique_Fiscal Quarter", 20),
        ("FSLine L1", "unique_FSLine Statement L1", 30),
        ("FSLine L2", "unique_FSLine Statement L2", 30),
    ]:
        vals = card.get(key) or []
        if vals:
            lines.append(f"{label}: {_format_list(vals, limit)}")
    return "\n".join(lines)


def format_structured_response(
    final_answer: str,
    filters: list[tuple[str, str]] | None = None,
    method: str | None = None,
    sanity: str | None = None,
) -> str:
    lines = [final_answer.strip()]
    if filters:
        lines.append("")
        lines.append("**Filters used:**")
        for k, v in filters:
            lines.append(f"- {k}: {v}")
    if method:
        lines.append("")
        lines.append("**Method:**")
        lines.append(method)
    if sanity:
        lines.append("")
        lines.append("**Sanity check:**")
        lines.append(sanity)
    return "\n".join(lines)


def preflight_validation(q: str, card: dict) -> str | None:
    q_text = (q or "").strip()
    q_lower = q_text.lower()
    products = [str(p) for p in (card.get("unique_Product") or [])]
    if products:
        for m in re.finditer(r"\bproduct\s+([a-z0-9]+)\b", q_text, flags=re.I):
            token = m.group(1)
            candidate = f"Product {token.upper()}"
            if not any(p.lower() == candidate.lower() for p in products):
                final = f"{candidate} does not exist in the dataset."
                filters = [("Product", candidate)]
                method = "Validated Product against dataset uniques before analysis."
                sanity = "Available products: " + ", ".join(products)
                return format_structured_response(final, filters, method, sanity)

    l2_vals = [str(v) for v in (card.get("unique_FSLine Statement L2") or [])]
    if "headcount" in q_lower and "expense" not in q_lower:
        if any(v.lower() == "headcount expenses" for v in l2_vals):
            final = "Employee Headcount is not in the dataset (only Headcount Expenses exists)."
            filters = [("Metric", "Employee Headcount")]
            method = "Validated FSLine items against dataset uniques before analysis."
            sanity = "Closest available line item: Headcount Expenses (expense, not headcount)."
            return format_structured_response(final, filters, method, sanity)

    return None


def metadata_response(q: str, card: dict) -> str | None:
    q_lower = (q or "").lower()
    if not card:
        return None

    summary_terms = [
        "summary of the csv",
        "summary of csv",
        "csv summary",
        "dataset summary",
        "describe the dataset",
        "schema",
        "headers",
    ]
    wants_summary = any(term in q_lower for term in summary_terms)
    wants_rows = bool(
        re.search(r"\bhow many rows\b|\brow count\b|\bnumber of rows\b|\brows and columns\b", q_lower)
    )
    wants_cols = bool(
        re.search(r"\bhow many columns\b|\bcolumn count\b|\bnumber of columns\b|\brows and columns\b", q_lower)
    )
    if "rows" in q_lower and "columns" in q_lower:
        wants_rows = True
        wants_cols = True
    if "column names" in q_lower or ("columns" in q_lower and any(w in q_lower for w in ["csv", "dataset", "file"])):
        wants_cols = True

    wants_head = bool(re.search(r"\bpreview\b|\bhead\b|\bfirst few rows\b|\bsample rows\b", q_lower))

    if not (wants_summary or wants_rows or wants_cols or wants_head):
        return None

    rows = card.get("rows")
    cols = card.get("cols")
    columns = card.get("columns") or []
    head = card.get("head") or []
    csv_path = os.getenv("CSV_PATH", "")
    dataset_name = os.path.basename(csv_path) if csv_path else "CSV dataset"

    lines = []
    if wants_summary or (wants_rows and wants_cols):
        if rows is not None and cols is not None:
            lines.append(f"The CSV has {rows} rows and {cols} columns.")
    elif wants_rows and rows is not None:
        lines.append(f"The CSV has {rows} rows.")
    elif wants_cols and cols is not None:
        lines.append(f"The CSV has {cols} columns.")

    if (wants_summary or wants_cols) and columns:
        lines.append("Columns: " + ", ".join(columns))

    if wants_head and head:
        lines.append("Preview (first 3 rows): " + json.dumps(head, indent=2))

    final = "\n".join(lines).strip() if lines else "Dataset metadata is unavailable."
    filters = [("Dataset", dataset_name)]
    method = "Used dataset metadata from cached dataset card; no filtering applied."
    sanity = "Rows: {}, Columns: {}.".format(rows, cols) if rows is not None and cols is not None else "Metadata read."
    return format_structured_response(final, filters, method, sanity)


def _find_longest_match(values: list[str], q_lower: str) -> str:
    matches = [v for v in values if v.lower() in q_lower]
    if not matches:
        return ""
    return max(matches, key=len)


def extract_simple_filters(q: str, card: dict) -> dict | None:
    q_text = (q or "").strip()
    q_lower = q_text.lower()
    complex_patterns = [
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
    ]
    if any(re.search(p, q_lower) for p in complex_patterns):
        return None

    year = None
    m_year = re.search(r"\b(20\d{2})\b", q_lower)
    if m_year:
        year = int(m_year.group(1))

    quarter = None
    m_q = re.search(r"\bq[1-4]\b", q_lower)
    if m_q:
        quarter = m_q.group(0).upper()

    products = [str(p) for p in (card.get("unique_Product") or [])]
    countries = [str(c) for c in (card.get("unique_Country") or [])]
    l1_vals = [str(v) for v in (card.get("unique_FSLine Statement L1") or [])]
    l2_vals = [str(v) for v in (card.get("unique_FSLine Statement L2") or [])]

    product = _find_longest_match(products, q_lower)
    if not product:
        m_prod = re.search(r"\bproduct\s+([a-z0-9]+)\b", q_text, flags=re.I)
        if m_prod:
            cand = f"Product {m_prod.group(1).upper()}"
            if any(p.lower() == cand.lower() for p in products):
                product = cand

    country = _find_longest_match(countries, q_lower)

    l2 = _find_longest_match(l2_vals, q_lower)
    l1 = ""
    if not l2:
        l1 = _find_longest_match(l1_vals, q_lower)

    if not l2 and not l1:
        if "total revenue" in q_lower or ("revenue" in q_lower and "gross" not in q_lower and "net" not in q_lower):
            if any(v.lower() == "net revenue" for v in l1_vals):
                l1 = "Net Revenue"
        elif "opex" in q_lower and any(v.lower() == "opex" for v in l1_vals):
            l1 = "OPEX"
        elif "cogs" in q_lower and any(v.lower() == "cost of goods sold" for v in l1_vals):
            l1 = "Cost of Goods Sold"

    if not l2 and not l1:
        return None

    amount_col = "Amount in USD"
    if "local currency" in q_lower or ("local" in q_lower and "currency" in q_lower):
        amount_col = "Amount in Local Currency"

    filter_dict: dict[str, object] = {}
    filters: list[tuple[str, str]] = []
    if year is not None:
        filter_dict["Fiscal Year"] = year
        filters.append(("Fiscal Year", str(year)))
    if quarter:
        filter_dict["Fiscal Quarter"] = quarter
        filters.append(("Fiscal Quarter", quarter))
    if product:
        filter_dict["Product"] = product
        filters.append(("Product", product))
    if country:
        filter_dict["Country"] = country
        filters.append(("Country", country))
    if l2:
        filter_dict["FSLine Statement L2"] = l2
        filters.append(("FSLine Statement L2", l2))
    elif l1:
        filter_dict["FSLine Statement L1"] = l1
        filters.append(("FSLine Statement L1", l1))

    return {
        "filters": filters,
        "filter_dict": filter_dict,
        "amount_col": amount_col,
        "metric": l2 or l1,
    }


def auto_answer_simple(q: str, card: dict) -> dict | None:
    parsed = extract_simple_filters(q, card)
    if not parsed:
        return None

    filter_dict = parsed["filter_dict"]
    amount_col = parsed["amount_col"]
    metric = parsed["metric"]

    code_lines = ["d = df"]
    for col, val in filter_dict.items():
        code_lines.append(f"d = d[d[{col!r}] == {val!r}]")
    code_lines.append(f"result = d[{amount_col!r}].sum()")
    code = "\n".join(code_lines)
    out = T.run_pandas(code)
    if not out.get("ok"):
        return None

    result = out.get("result")
    value_str = str(result)
    try:
        num = float(result)
        if amount_col == "Amount in USD":
            value_str = f"${num:,.2f} USD"
        else:
            value_str = f"{num:,.2f} (local currency)"
    except (TypeError, ValueError):
        pass

    final = f"The total {metric} is {value_str}."
    method = "Auto-parsed simple filters and summed the amount column with pandas."
    sanity = "Rows matched: unknown."
    count_out = T.filter_count(filter_dict)
    if count_out.get("ok"):
        sanity = f"Rows matched: {count_out.get('count')}."

    fallback_answer = format_structured_response(final, parsed["filters"], method, sanity)
    return {
        "code": code,
        "tool_result": out,
        "fallback_answer": fallback_answer,
    }


def render_answer(llm, question: str, pandas_code: str, tool_result: dict) -> str:
    payload = {
        "question": question,
        "pandas_code": pandas_code,
        "tool_result": tool_result,
    }
    resp = llm.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )
    return (resp.content or "").strip()


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing in .env")

    debug_tools = os.getenv("DEBUG_TOOLS", "0").strip() == "1"
    history_keep = int(os.getenv("MAX_HISTORY_MESSAGES", str(DEFAULT_HISTORY_MESSAGES)))
    worker_model = os.getenv("WORKER_MODEL", WORKER_MODEL)
    judge_model = os.getenv("JUDGE_MODEL", JUDGE_MODEL)

    base_dataset_card = T.dataset_card()
    dataset_ctx = dataset_context_text(base_dataset_card)
    system_base = WORKER_SYSTEM_BASE
    if dataset_ctx:
        system_base = system_base + "\n\n" + dataset_ctx

    worker = ChatGroq(
        model=worker_model,
        temperature=0.2,
        groq_api_key=api_key,
    )
    writer = ChatGroq(
        model=worker_model,
        temperature=0,
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

    print(f"[bold green]CSV Bot[/bold green] (Worker: {worker_model} | Judge: {judge_model})")
    print("Type 'exit' to quit.\n")

    state = ChatState(system_base=system_base)
    state.init()
    pending_mistakes = []
    disable_write = os.getenv("VDB_DISABLE_WRITE", "0").strip() == "1"

    def emit_answer(text: str) -> None:
        state.add_ai(text)
        print("\n[bold cyan]Bot:[/bold cyan]")
        print(text)
        print()
        trim_history(state, history_keep)

    def run_tool_loop(resp):
        nonlocal last_code, last_tool_result, dataset_card, tool_loop_exceeded
        tool_iters = 0
        while getattr(resp, "tool_calls", None):
            tool_iters += 1
            if tool_iters > MAX_TOOL_ITERS:
                tool_loop_exceeded = True
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
        nonlocal last_code, last_tool_result, dataset_card, tool_loop_exceeded
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
        tool_loop_exceeded = False
        return run_tool_loop(resp)

    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        q_lower = q.lower()
        if q_lower == "exit" or q_lower.startswith("exit "):
            print("\n[bold cyan]Bot:[/bold cyan] Exiting and saving learning (if any).")
            break

        meta_msg = metadata_response(q, base_dataset_card)
        if meta_msg:
            state.add_user(q)
            emit_answer(meta_msg)
            continue

        validation_msg = preflight_validation(q, base_dataset_card)
        if validation_msg:
            state.add_user(q)
            emit_answer(validation_msg)
            continue

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
        tool_loop_exceeded = False
        fallback_bundle = None
        resp = safe_llm_invoke(worker, state.messages, retries=1)
        resp = run_tool_loop(resp)

        if tool_loop_exceeded:
            fallback_bundle = auto_answer_simple(q, base_dataset_card)
            if fallback_bundle:
                last_code = fallback_bundle["code"]
                last_tool_result = fallback_bundle["tool_result"]
                tool_loop_exceeded = False
            else:
                fail_msg = (
                    f"I started looping on tool calls and stopped to avoid hanging. "
                    f"(max iterations={MAX_TOOL_ITERS})\n"
                    "Tip: The model is probably failing to find the exact dataset label for a metric. "
                    "Try asking with exact FSLine names or enable DEBUG_TOOLS=1 to see the tool loop.\n"
                )
                emit_answer(fail_msg)
                continue

        if resp is None and not last_code:
            continue

        if not last_code:
            resp = force_pandas_exec()
            if tool_loop_exceeded:
                fallback_bundle = auto_answer_simple(q, base_dataset_card)
                if fallback_bundle:
                    last_code = fallback_bundle["code"]
                    last_tool_result = fallback_bundle["tool_result"]
                    tool_loop_exceeded = False
                else:
                    fail_msg = (
                        f"I started looping on tool calls and stopped to avoid hanging. "
                        f"(max iterations={MAX_TOOL_ITERS})\n"
                        "Tip: The model is probably failing to find the exact dataset label for a metric. "
                        "Try asking with exact FSLine names or enable DEBUG_TOOLS=1 to see the tool loop.\n"
                    )
                    emit_answer(fail_msg)
                    continue
            if resp is None and not last_code:
                continue

        if not dataset_card:
            dataset_card = base_dataset_card

        if not last_code:
            fallback_bundle = auto_answer_simple(q, base_dataset_card)
            if fallback_bundle:
                last_code = fallback_bundle["code"]
                last_tool_result = fallback_bundle["tool_result"]
            else:
                guard = (
                    "I need to run a pandas query to answer this question, but no pandas code was executed. "
                    "Please re-ask the question so I can compute it using tools."
                )
                emit_answer(guard)
                continue
        if not last_tool_result:
            guard = (
                "I need to run a pandas query to answer this question, but no pandas code was executed. "
                "Please re-ask the question so I can compute it using tools."
            )
            emit_answer(guard)
            continue

        final_code = last_code
        final_result = last_tool_result
        decision = judge(
            groq_api_key=api_key,
            dataset_card=dataset_card,
            user_question=q,
            worker_code=last_code,
            worker_tool_result=last_tool_result,
            model=judge_model,
            dataset_context=dataset_ctx,
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
            if corrected_out.get("ok"):
                final_code = decision["corrected_code"]
                final_result = corrected_out

        if not final_result.get("ok", True):
            error_text = final_result.get("error") or "Unknown error"
            answer = format_structured_response(
                final_answer=f"Analysis failed: {error_text}",
                filters=None,
                method="Pandas execution reported an error.",
                sanity="No numeric result produced.",
            )
            emit_answer(answer)
            continue

        try:
            answer = render_answer(
                writer,
                question=q,
                pandas_code=final_code,
                tool_result=final_result,
            )
        except Exception:
            if fallback_bundle and fallback_bundle.get("fallback_answer"):
                answer = fallback_bundle["fallback_answer"]
            else:
                answer = format_structured_response(
                    final_answer=str(final_result.get("result")),
                    filters=None,
                    method="Rendered from latest tool result.",
                    sanity="Answer derived from pandas output.",
                )
        emit_answer(answer)

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
