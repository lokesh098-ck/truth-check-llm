"""
Consistency-based verification for 'qualitative' claims (causal statements,
soft interpretations, etc.) that the rule engine cannot recompute directly
from the dataset — e.g. "sales likely rose due to the holiday season".

This mirrors SelfCheckGPT's core idea: ask the model to justify/re-derive
the claim strictly from the dataset multiple times, in different phrasings,
and check whether the answers agree. Disagreement is treated as a signal
of low grounding (i.e. more likely hallucinated), NOT proof of falsity —
this checker is intentionally a weaker, secondary signal compared to the
deterministic rule engine, and the fusion layer weights it accordingly.

Requires an LLM client (see llm_client.py). If none is configured, every
qualitative claim is returned as 'unverifiable' rather than guessed at.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional

import pandas as pd

from .claim_extractor import Claim
from .verifier_rules import SUPPORTED, PARTIAL, HALLUCINATED, UNVERIFIABLE, VerificationResult

REPHRASINGS = [
    "Based strictly on the dataset provided, is the following statement well-supported? Answer in one short sentence, "
    "then say SUPPORTED, PARTIAL, or UNSUPPORTED. Statement: \"{claim}\"",
    "Looking only at the data (not general knowledge), would you agree with this claim? Reply with one short sentence "
    "and then SUPPORTED, PARTIAL, or UNSUPPORTED. Claim: \"{claim}\"",
    "Does the dataset actually justify this statement, or is it an assumption not backed by the numbers? Give a short "
    "justification then end with SUPPORTED, PARTIAL, or UNSUPPORTED. Statement: \"{claim}\"",
]

_LABEL_RE = re.compile(r"\b(SUPPORTED|PARTIAL|UNSUPPORTED)\b", re.I)


def _dataset_summary(df: pd.DataFrame, max_rows: int = 15) -> str:
    """A compact textual snapshot of the dataset to ground the consistency prompts."""
    desc = df.describe(include="all").transpose().round(2).to_string()
    sample = df.head(max_rows).to_string(index=False)
    return f"Columns: {list(df.columns)}\n\nSummary statistics:\n{desc}\n\nSample rows:\n{sample}"


def verify_qualitative(claim: Claim, df: pd.DataFrame, llm_client=None, n_samples: int = 3) -> VerificationResult:
    if llm_client is None:
        return VerificationResult(
            claim, UNVERIFIABLE,
            evidence="No LLM client configured — qualitative claims require the consistency checker to be enabled."
        )

    context = _dataset_summary(df)
    labels: List[str] = []
    raw_responses: List[str] = []

    for template in REPHRASINGS[:n_samples]:
        prompt = f"{context}\n\n{template.format(claim=claim.text)}"
        try:
            response = llm_client.complete(prompt)
        except Exception as exc:
            raw_responses.append(f"[error: {exc}]")
            continue
        raw_responses.append(response)
        m = _LABEL_RE.search(response or "")
        labels.append(m.group(1).upper() if m else "UNSUPPORTED")

    if not labels:
        return VerificationResult(claim, UNVERIFIABLE, evidence="LLM consistency check produced no usable responses.")

    supported_votes = labels.count("SUPPORTED")
    partial_votes = labels.count("PARTIAL")
    unsupported_votes = labels.count("UNSUPPORTED")
    total = len(labels)

    agreement = max(supported_votes, partial_votes, unsupported_votes) / total

    if supported_votes / total >= 0.6:
        status = SUPPORTED
    elif unsupported_votes / total >= 0.6:
        status = HALLUCINATED
    else:
        status = PARTIAL

    evidence = (
        f"Consistency check across {total} independent rephrasings: "
        f"{supported_votes} SUPPORTED / {partial_votes} PARTIAL / {unsupported_votes} UNSUPPORTED "
        f"(agreement={agreement:.0%}). This is a secondary, weaker signal than dataset recomputation."
    )
    return VerificationResult(claim, status, actual_value=f"{agreement:.0%} agreement", evidence=evidence)
