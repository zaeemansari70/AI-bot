# tools.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List
import pandas as pd
from pandas.api.types import is_numeric_dtype

from safety import validate_pandas_code, UnsafeCodeError

# ---------
# CSV cache
# ---------
_DF_CACHE: pd.DataFrame | None = None
_DF_CACHE_PATH: str | None = None
_DF_CACHE_MTIME: float | None = None
_DF_CACHE_LOADED_AT: float | None = None


def _csv_path() -> str:
    p = os.getenv("CSV_PATH")
    if not p:
        raise RuntimeError("CSV_PATH missing in .env")
    return p


def _load_df_fresh(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _df() -> pd.DataFrame:
    global _DF_CACHE, _DF_CACHE_PATH, _DF_CACHE_MTIME, _DF_CACHE_LOADED_AT

    path = _csv_path()
    try:
        mtime = os.path.getmtime(path)
    except Exception as e:
        raise RuntimeError(f"Could not stat CSV_PATH={path}: {type(e).__name__}: {e}")

    if (
        _DF_CACHE is None
        or _DF_CACHE_PATH != path
        or _DF_CACHE_MTIME is None
        or mtime != _DF_CACHE_MTIME
    ):
        _DF_CACHE = _load_df_fresh(path)
        _DF_CACHE_PATH = path
        _DF_CACHE_MTIME = mtime
        _DF_CACHE_LOADED_AT = time.time()

    return _DF_CACHE


def dataset_card() -> dict:
    df = _df()

    card = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "head": df.head(3).to_dict(orient="records"),
    }

    # Give the model some useful enumerations (truncated)
    cat_cols = [
        "Product",
        "Country",
        "Currency",
        "Fiscal Year",
        "Fiscal Quarter",
        "Version",
        "FSLine Statement L1",
        "FSLine Statement L2",
    ]
    for col in cat_cols:
        if col in df.columns:
            vals = df[col].dropna().astype(str).unique().tolist()
            vals = sorted(vals)
            card[f"unique_{col}"] = vals[:50]

    if "Fiscal Year" in df.columns:
        yrs = pd.to_numeric(df["Fiscal Year"], errors="coerce").dropna()
        if len(yrs) > 0:
            card["fiscal_year_min"] = int(yrs.min())
            card["fiscal_year_max"] = int(yrs.max())

    return card


def get_uniques(column: str, limit: int = 50) -> Dict[str, Any]:
    df = _df()
    if column not in df.columns:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": f"Unknown column: {column}", "values": []}

    vals = df[column].dropna().astype(str).unique().tolist()
    vals = sorted(vals)
    vals = vals[: max(1, min(int(limit), 500))]
    return {"ok": True, "values": vals}


def search_uniques(column: str, contains: str, limit: int = 50) -> Dict[str, Any]:
    """
    Search unique string values in a column by substring (case-insensitive).
    This prevents the model looping when it doesn't know the exact label.
    """
    df = _df()
    if column not in df.columns:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": f"Unknown column: {column}", "values": []}

    needle = (contains or "").strip().lower()
    if not needle:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": "contains must be non-empty", "values": []}

    vals = df[column].dropna().astype(str).unique().tolist()
    matches = [v for v in vals if needle in v.lower()]
    matches = sorted(matches)
    matches = matches[: max(1, min(int(limit), 200))]

    return {"ok": True, "values": matches, "match_count": len(matches)}


def value_exists(column: str, value: str) -> Dict[str, Any]:
    df = _df()
    if column not in df.columns:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": f"Unknown column: {column}"}

    s = set(df[column].dropna().astype(str).unique().tolist())
    return {"ok": True, "exists": str(value) in s}


def normalize_filter_value(val: Any):
    if isinstance(val, (list, tuple, set)):
        return ("IN", list(val))
    if isinstance(val, dict) and ("min" in val or "max" in val):
        return ("RANGE", val.get("min"), val.get("max"))
    return ("EQ", val)

def _coerce_scalar_for_column(series: pd.Series, val: Any) -> Any:
    if val is None:
        return val
    if is_numeric_dtype(series):
        if isinstance(val, str):
            v = val.strip()
            if v == "":
                return val
            try:
                if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                    return int(v)
                return float(v)
            except Exception:
                return val
        if isinstance(val, (int, float)):
            return val
    return val


def _coerce_filter_value_for_column(series: pd.Series, raw: Any) -> Any:
    if isinstance(raw, dict) and ("min" in raw or "max" in raw):
        return {
            "min": _coerce_scalar_for_column(series, raw.get("min")),
            "max": _coerce_scalar_for_column(series, raw.get("max")),
        }
    if isinstance(raw, (list, tuple, set)):
        return [_coerce_scalar_for_column(series, v) for v in raw]
    return _coerce_scalar_for_column(series, raw)


def filter_count(filters: Dict[str, Any]) -> Dict[str, Any]:
    df = _df()
    d = df
    try:
        for col, raw in (filters or {}).items():
            if col not in d.columns:
                return {"ok": False, "error_type": "TOOL_INPUT", "error": f"Unknown column: {col}"}

            coerced = _coerce_filter_value_for_column(d[col], raw)
            mode = normalize_filter_value(coerced)

            if mode[0] == "IN":
                d = d[d[col].isin(mode[1])]
            elif mode[0] == "RANGE":
                min_v, max_v = mode[1], mode[2]
                if min_v is not None:
                    d = d[d[col] >= min_v]
                if max_v is not None:
                    d = d[d[col] <= max_v]
            else:
                d = d[d[col] == mode[1]]

        return {"ok": True, "count": int(len(d))}
    except Exception as e:
        return {"ok": False, "error_type": "TOOL_EXEC", "error": f"{type(e).__name__}: {e}"}


def run_pandas(code: str) -> Dict[str, Any]:
    df = _df()
    try:
        validate_pandas_code(code)
    except UnsafeCodeError as e:
        return {"ok": False, "error_type": "SANDBOX", "error": str(e)}

    safe_globals = {
        "__builtins__": {"len": len, "sum": sum, "min": min, "max": max, "round": round, "abs": abs},
        "pd": pd,
        "df": df,
    }
    safe_locals: Dict[str, Any] = {}

    try:
        exec(code, safe_globals, safe_locals)
        result = safe_locals.get("result", None)

        if isinstance(result, pd.DataFrame):
            return {"ok": True, "result": result.head(30).to_dict(orient="records")}
        if isinstance(result, pd.Series):
            return {"ok": True, "result": result.head(30).to_dict()}
        return {"ok": True, "result": result}

    except Exception as e:
        return {"ok": False, "error_type": "TOOL_EXEC", "error": f"{type(e).__name__}: {e}"}
    
def search_fsline_l2(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search FSLine Statement L2 values using simple string matching.
    This is deterministic and avoids LLM guessing loops.
    """
    df = _df()
    col = "FSLine Statement L2"
    if col not in df.columns:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": f"Missing column: {col}", "matches": []}

    q = (query or "").strip().lower()
    if not q:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": "query must be non-empty", "matches": []}
    vals = df[col].dropna().astype(str)

    # Count occurrences for ranking
    counts = vals.value_counts()

    # simple substring match
    matches = []
    for v, c in counts.items():
        if q in v.lower():
            matches.append({"value": v, "count": int(c)})

    return {"ok": True, "matches": matches[: max(1, min(limit, 50))]}


def search_fsline_l1(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search FSLine Statement L1 values using simple string matching.
    """
    df = _df()
    col = "FSLine Statement L1"
    if col not in df.columns:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": f"Missing column: {col}", "matches": []}

    q = (query or "").strip().lower()
    if not q:
        return {"ok": False, "error_type": "TOOL_INPUT", "error": "query must be non-empty", "matches": []}
    vals = df[col].dropna().astype(str)

    counts = vals.value_counts()
    matches = []
    for v, c in counts.items():
        if q in v.lower():
            matches.append({"value": v, "count": int(c)})

    return {"ok": True, "matches": matches[: max(1, min(limit, 50))]}

