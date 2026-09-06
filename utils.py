"""
Shared helper utilities for TruthCheck-LLM.
"""
from __future__ import annotations

import difflib
import re
from typing import Optional, List

import pandas as pd


def is_categorical_column(series: pd.Series) -> bool:
    """
    True for text/category-like columns. Handles both classic pandas
    'object' dtype and the newer pandas>=2.x/3.x native 'string'/'category'
    dtypes, so category detection doesn't silently break across pandas versions.
    """
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
    )


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV/XLSX file into a DataFrame with light cleanup."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fuzzy_match_column(name: str, columns: List[str], cutoff: float = 0.55) -> Optional[str]:
    """
    Map a phrase mentioned in an LLM narrative (e.g. 'sales revenue') to the
    closest real column name (e.g. 'Revenue'). Returns None if nothing is
    close enough, so callers can mark the claim as 'unverifiable' rather than
    silently guessing wrong.
    """
    if not name:
        return None
    name_norm = name.strip().lower()

    # exact match first (cheap and very reliable)
    for c in columns:
        if name_norm == c.strip().lower():
            return c

    # substring match — prefer the LONGEST matching column name, so a short
    # column like 'Region' doesn't win over 'MarketingSpend' just because
    # 'region' happens to appear inside a longer captured phrase like
    # 'marketingspend across all regions'.
    substring_matches = [c for c in columns if name_norm in c.strip().lower() or c.strip().lower() in name_norm]
    if substring_matches:
        return max(substring_matches, key=lambda c: len(c.strip()))

    # fall back to fuzzy string matching
    matches = difflib.get_close_matches(name_norm, [c.lower() for c in columns], n=1, cutoff=cutoff)
    if matches:
        idx = [c.lower() for c in columns].index(matches[0])
        return columns[idx]
    return None


def parse_number(text: str) -> Optional[float]:
    """Parse a human-written number like '$45,230.50' or '12%' or '3.4k' into a float."""
    if text is None:
        return None
    t = text.strip().lower().replace(",", "").replace("$", "").replace("%", "")
    multiplier = 1.0
    if t.endswith("k"):
        multiplier, t = 1_000.0, t[:-1]
    elif t.endswith("m"):
        multiplier, t = 1_000_000.0, t[:-1]
    elif t.endswith("b"):
        multiplier, t = 1_000_000_000.0, t[:-1]
    try:
        return float(t) * multiplier
    except ValueError:
        return None


def pct_diff(actual: float, claimed: float) -> float:
    """Relative percentage difference between an actual and a claimed value."""
    if actual == 0:
        return 0.0 if claimed == 0 else float("inf")
    return abs(actual - claimed) / abs(actual) * 100.0


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter — good enough for narrative paragraphs."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]
