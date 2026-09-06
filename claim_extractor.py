"""
Extracts structured, checkable claims from an LLM-generated data narrative.

Each sentence in the narrative is scanned against a set of patterns for the
claim types we know how to verify deterministically: average/mean, sum/total,
percentage/proportion, ranking (highest/lowest), correlation, and trend
(increase/decrease). Anything that doesn't match a pattern is kept as a
'qualitative' claim, which the consistency checker (not the rule engine)
is responsible for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .utils import split_sentences, fuzzy_match_column, parse_number

NUM = r"[-+]?\$?\d[\d,]*\.?\d*\s?[kmb%]?"


@dataclass
class Claim:
    id: int
    text: str                     # the original sentence
    claim_type: str                # 'average' | 'sum' | 'percentage' | 'ranking' | 'correlation' | 'trend' | 'qualitative'
    columns: List[str] = field(default_factory=list)   # matched dataframe columns
    claimed_value: Optional[float] = None
    group_value: Optional[str] = None   # e.g. category name in a ranking/percentage claim
    direction: Optional[str] = None     # 'highest' | 'lowest' | 'increase' | 'decrease' | 'positive' | 'negative'


_PATTERNS = [
    # "The average revenue was $45,230" / "mean order value is 120.5"
    (
        "average",
        re.compile(rf"(average|mean)\s+(?P<col>[a-z0-9 _\-]+?)\s+(?:was|is|stood at|of)\s+(?P<val>{NUM})", re.I),
    ),
    # "Total sales amounted to 1.2m" / "the sum of profit was 400000"
    (
        "sum",
        re.compile(rf"(total|sum)\s+(?:of\s+)?(?P<col>[a-z0-9 _\-]+?)\s+(?:was|is|amounted to|totaled|equal(?:s|led)?)\s+(?P<val>{NUM})", re.I),
    ),
    # "North accounts for 45% of Region" -> group='North', col='Region'
    (
        "percentage",
        re.compile(rf"(?P<group>[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){{0,3}}?)\s+(?:accounts? for|represents?|makes? up)\s+(?P<val>{NUM}%)\s+of\s+(?:the\s+)?(?P<col>[a-z0-9 _\-]+)", re.I),
    ),
    # "45% of customers were churned" -> col='customers', group='churned'
    (
        "percentage",
        re.compile(rf"(?P<val>{NUM}%)\s+of\s+(?:the\s+)?(?P<col>[a-z0-9 _\-]+?)\s+(?:were|are|is|was)\s+(?P<group>[a-z0-9 _\-]+)", re.I),
    ),
    # "North accounts for 45%" (no trailing 'of <col>' — column must be inferred)
    (
        "percentage",
        re.compile(rf"(?P<group>[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){{0,3}}?)\s+(?:accounts? for|represents?|makes? up)\s+(?P<val>{NUM}%)", re.I),
    ),
    # "Region X had the highest sales" / "Product Y recorded the lowest average rating"
    # 'col' stops at a boundary word/punctuation so trailing modifiers (e.g.
    # "across all regions") aren't swept into the column name.
    (
        "ranking",
        re.compile(
            r"(?P<group>[A-Z][a-zA-Z0-9 _\-]*?)\s+(?:had|has|recorded|showed)\s+the\s+(?P<dir>highest|lowest)\s+"
            r"(?P<col>[a-z0-9][a-z0-9 _\-]*?)(?=[,.]|\s+(?:across|in|on|for|during|among|within|over|of)\b|$)",
            re.I,
        ),
    ),
    # "There is a positive correlation between marketing spend and sales"
    (
        "correlation",
        re.compile(
            r"(?P<dir>positive|negative|strong|weak|no)\s+correlation\s+between\s+"
            r"(?P<col1>[a-z0-9][a-z0-9 _\-]*?)\s+and\s+(?P<col2>[a-z0-9][a-z0-9 _\-]*?)(?=[,.]|$)",
            re.I,
        ),
    ),
    # "Sales increased by 15% from Q1 to Q2" / "revenue decreased by 8%"
    (
        "trend",
        re.compile(rf"(?P<col>[a-z0-9 _\-]+?)\s+(?P<dir>increased|decreased|grew|dropped|fell|rose)\s+by\s+(?P<val>{NUM})", re.I),
    ),
]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,")


def extract_claims(narrative: str, df: pd.DataFrame) -> List[Claim]:
    sentences = split_sentences(narrative)
    columns = list(df.columns)
    claims: List[Claim] = []
    cid = 0

    for sentence in sentences:
        matched_any = False
        for claim_type, pattern in _PATTERNS:
            m = pattern.search(sentence)
            if not m:
                continue
            matched_any = True
            gd = m.groupdict()
            cid += 1

            if claim_type in ("average", "sum"):
                col = fuzzy_match_column(_clean(gd.get("col", "")), columns)
                claims.append(Claim(
                    id=cid, text=sentence, claim_type=claim_type,
                    columns=[col] if col else [],
                    claimed_value=parse_number(gd.get("val")),
                ))

            elif claim_type == "percentage":
                col = fuzzy_match_column(_clean(gd.get("col", "")), columns) if gd.get("col") else None
                claims.append(Claim(
                    id=cid, text=sentence, claim_type=claim_type,
                    columns=[col] if col else [],
                    claimed_value=parse_number(gd.get("val")),
                    group_value=_clean(gd.get("group", "")),
                ))

            elif claim_type == "ranking":
                col = fuzzy_match_column(_clean(gd.get("col", "")), columns)
                claims.append(Claim(
                    id=cid, text=sentence, claim_type=claim_type,
                    columns=[col] if col else [],
                    group_value=_clean(gd.get("group", "")),
                    direction=gd.get("dir", "").lower(),
                ))

            elif claim_type == "correlation":
                col1 = fuzzy_match_column(_clean(gd.get("col1", "")), columns)
                col2 = fuzzy_match_column(_clean(gd.get("col2", "")), columns)
                claims.append(Claim(
                    id=cid, text=sentence, claim_type=claim_type,
                    columns=[c for c in (col1, col2) if c],
                    direction=gd.get("dir", "").lower(),
                ))

            elif claim_type == "trend":
                col = fuzzy_match_column(_clean(gd.get("col", "")), columns)
                claims.append(Claim(
                    id=cid, text=sentence, claim_type=claim_type,
                    columns=[col] if col else [],
                    claimed_value=parse_number(gd.get("val")),
                    direction=gd.get("dir", "").lower(),
                ))
            break  # stop after first matching pattern for this sentence

        if not matched_any:
            cid += 1
            claims.append(Claim(id=cid, text=sentence, claim_type="qualitative"))

    return claims
