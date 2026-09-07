"""
Deterministic, rule-based verification of structured claims against the
source dataset. No ML, no external knowledge base — every number here is
recomputed directly with pandas/numpy so the result is fully reproducible
and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd

from .utils import fuzzy_match_column, pct_diff, is_categorical_column
from .claim_extractor import Claim

SUPPORTED = "supported"
PARTIAL = "partially_supported"
HALLUCINATED = "hallucinated"
UNVERIFIABLE = "unverifiable"   # e.g. column couldn't be matched, or claim needs the consistency checker

TOLERANCE_SUPPORTED_PCT = 3.0     # within 3% -> supported
TOLERANCE_PARTIAL_PCT = 15.0      # within 15% -> partially supported, beyond -> hallucinated


@dataclass
class VerificationResult:
    claim: Claim
    status: str
    actual_value: Optional[Any] = None
    evidence: str = ""
    corrected_text: Optional[str] = None


def _numeric_verdict(actual: float, claimed: float) -> str:
    diff = pct_diff(actual, claimed)
    if diff <= TOLERANCE_SUPPORTED_PCT:
        return SUPPORTED
    if diff <= TOLERANCE_PARTIAL_PCT:
        return PARTIAL
    return HALLUCINATED


def _find_group_column(df: pd.DataFrame, numeric_col: Optional[str]) -> Optional[str]:
    """Best-effort guess at which column holds category labels (for ranking claims)."""
    candidates = [c for c in df.columns if c != numeric_col and is_categorical_column(df[c])]
    return candidates[0] if candidates else None


def verify_average(claim: Claim, df: pd.DataFrame) -> VerificationResult:
    if not claim.columns or claim.claimed_value is None:
        return VerificationResult(claim, UNVERIFIABLE, evidence="Could not match a numeric column.")
    col = claim.columns[0]
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return VerificationResult(claim, UNVERIFIABLE, evidence=f"Column '{col}' is not numeric.")
    actual = float(df[col].mean())
    status = _numeric_verdict(actual, claim.claimed_value)
    return VerificationResult(
        claim, status, actual_value=round(actual, 2),
        evidence=f"Recomputed mean({col}) = {actual:.2f} over {len(df)} rows; claim stated {claim.claimed_value}.",
        corrected_text=f"The average {col} was {actual:.2f}." if status != SUPPORTED else None,
    )


def verify_sum(claim: Claim, df: pd.DataFrame) -> VerificationResult:
    if not claim.columns or claim.claimed_value is None:
        return VerificationResult(claim, UNVERIFIABLE, evidence="Could not match a numeric column.")
    col = claim.columns[0]
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return VerificationResult(claim, UNVERIFIABLE, evidence=f"Column '{col}' is not numeric.")
    actual = float(df[col].sum())
    status = _numeric_verdict(actual, claim.claimed_value)
    return VerificationResult(
        claim, status, actual_value=round(actual, 2),
        evidence=f"Recomputed sum({col}) = {actual:.2f}; claim stated {claim.claimed_value}.",
        corrected_text=f"The total {col} was {actual:.2f}." if status != SUPPORTED else None,
    )


def verify_percentage(claim: Claim, df: pd.DataFrame) -> VerificationResult:
    if claim.claimed_value is None or not claim.group_value:
        return VerificationResult(claim, UNVERIFIABLE, evidence="Could not parse the percentage claim.")

    # Try: does group_value name a categorical column, and is it referring to a
    # proportion of rows where some column equals a specific category?
    # Heuristic: search all object columns for a value matching claim.group_value.
    target_col = claim.columns[0] if claim.columns else None
    best = None  # (col, category_value, share)
    search_cols = [target_col] if target_col else list(df.columns)
    for col in search_cols:
        if col not in df.columns:
            continue
        if is_categorical_column(df[col]):
            for val in df[col].dropna().unique():
                if str(val).lower() in claim.group_value.lower() or claim.group_value.lower() in str(val).lower():
                    share = (df[col] == val).mean() * 100
                    best = (col, val, share)
                    break
        if best:
            break

    if best is None:
        return VerificationResult(claim, UNVERIFIABLE, evidence=f"Could not locate a category matching '{claim.group_value}'.")

    col, val, actual_pct = best
    status = _numeric_verdict(actual_pct, claim.claimed_value)
    return VerificationResult(
        claim, status, actual_value=round(actual_pct, 2),
        evidence=f"Rows where {col} == '{val}' make up {actual_pct:.2f}% of {len(df)} rows; claim stated {claim.claimed_value}%.",
        corrected_text=f"{val} accounts for {actual_pct:.1f}% of {col}." if status != SUPPORTED else None,
    )


def verify_ranking(claim: Claim, df: pd.DataFrame) -> VerificationResult:
    if not claim.columns or not claim.direction:
        return VerificationResult(claim, UNVERIFIABLE, evidence="Could not match the ranked column.")
    numeric_col = claim.columns[0]
    if numeric_col not in df.columns or not pd.api.types.is_numeric_dtype(df[numeric_col]):
        return VerificationResult(claim, UNVERIFIABLE, evidence=f"Column '{numeric_col}' is not numeric.")
    group_col = _find_group_column(df, numeric_col)
    if group_col is None:
        return VerificationResult(claim, UNVERIFIABLE, evidence="No categorical column found to rank against.")

    agg = df.groupby(group_col)[numeric_col].mean().sort_values(ascending=(claim.direction == "lowest"))
    actual_top = str(agg.index[0])
    claimed_group = (claim.group_value or "").lower()

    if claimed_group and (claimed_group in actual_top.lower() or actual_top.lower() in claimed_group):
        status = SUPPORTED
    else:
        status = HALLUCINATED
    return VerificationResult(
        claim, status, actual_value=actual_top,
        evidence=f"Ranking {group_col} by mean({numeric_col}): actual {claim.direction} is '{actual_top}' "
                 f"({agg.iloc[0]:.2f}); claim named '{claim.group_value}'.",
        corrected_text=f"{actual_top} had the {claim.direction} {numeric_col}." if status != SUPPORTED else None,
    )


def verify_correlation(claim: Claim, df: pd.DataFrame) -> VerificationResult:
    if len(claim.columns) < 2:
        return VerificationResult(claim, UNVERIFIABLE, evidence="Could not match both columns in the correlation claim.")
    c1, c2 = claim.columns[0], claim.columns[1]
    if not (pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2])):
        return VerificationResult(claim, UNVERIFIABLE, evidence="One or both columns are not numeric.")
    r = float(df[c1].corr(df[c2]))
    actual_dir = "positive" if r > 0.1 else ("negative" if r < -0.1 else "no")
    claimed_dir = claim.direction if claim.direction in ("positive", "negative", "no") else (
        "positive" if claim.direction == "strong" else "no"
    )
    status = SUPPORTED if actual_dir == claimed_dir else (PARTIAL if abs(r) < 0.3 else HALLUCINATED)
    return VerificationResult(
        claim, status, actual_value=round(r, 3),
        evidence=f"Pearson correlation({c1}, {c2}) = {r:.3f} -> {actual_dir}; claim stated '{claim.direction}'.",
        corrected_text=f"There is a {actual_dir} correlation between {c1} and {c2} (r={r:.2f})." if status != SUPPORTED else None,
    )


def verify_trend(claim: Claim, df: pd.DataFrame) -> VerificationResult:
    # Without a reliable, auto-detected time column this is inherently the
    # hardest claim type to ground generically — we verify direction only
    # (increase/decrease) using row order as a proxy for time when no
    # explicit date column exists, and flag low confidence via 'partial'.
    if not claim.columns or claim.claimed_value is None or not claim.direction:
        return VerificationResult(claim, UNVERIFIABLE, evidence="Could not parse the trend claim.")
    col = claim.columns[0]
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return VerificationResult(claim, UNVERIFIABLE, evidence=f"Column '{col}' is not numeric.")

    half = len(df) // 2
    if half == 0:
        return VerificationResult(claim, UNVERIFIABLE, evidence="Not enough rows to assess a trend.")
    first_half_mean = df[col].iloc[:half].mean()
    second_half_mean = df[col].iloc[half:].mean()
    if first_half_mean == 0:
        pct_change = float("inf")
    else:
        pct_change = (second_half_mean - first_half_mean) / abs(first_half_mean) * 100
    actual_dir = "increased" if pct_change > 0 else "decreased"
    claimed_dir = "increased" if claim.direction in ("increased", "grew", "rose") else "decreased"

    dir_ok = actual_dir == claimed_dir
    diff = pct_diff(abs(pct_change), abs(claim.claimed_value)) if claim.claimed_value else 100
    if dir_ok and diff <= TOLERANCE_SUPPORTED_PCT:
        status = SUPPORTED
    elif dir_ok:
        status = PARTIAL
    else:
        status = HALLUCINATED
    return VerificationResult(
        claim, status, actual_value=round(pct_change, 2),
        evidence=f"Comparing first half vs second half of the data, {col} {actual_dir} by {abs(pct_change):.2f}% "
                 f"(row order used as a proxy for time — verify a real date column exists for stronger confidence); "
                 f"claim stated {claim.direction} by {claim.claimed_value}%.",
        corrected_text=f"{col} {actual_dir} by {abs(pct_change):.1f}%." if status != SUPPORTED else None,
    )


_DISPATCH = {
    "average": verify_average,
    "sum": verify_sum,
    "percentage": verify_percentage,
    "ranking": verify_ranking,
    "correlation": verify_correlation,
    "trend": verify_trend,
}


def verify_claim(claim: Claim, df: pd.DataFrame) -> VerificationResult:
    fn = _DISPATCH.get(claim.claim_type)
    if fn is None:
        # 'qualitative' claims are handled by the consistency checker, not here.
        return VerificationResult(claim, UNVERIFIABLE, evidence="Not a rule-checkable claim type.")
    try:
        return fn(claim, df)
    except Exception as exc:  # keep the pipeline resilient to unexpected data
        return VerificationResult(claim, UNVERIFIABLE, evidence=f"Rule verifier error: {exc}")
