"""
Fusion layer: turns a list of per-claim VerificationResults into the final
report the user sees — each claim's verdict plus one aggregate Hallucination
Rate for the whole narrative (the single-number metric the project uses to
compare different LLMs / prompts / narratives against each other).

Rule-based results are authoritative (weight 1.0) since they are
deterministically recomputed from the data. Consistency-check results are
a secondary, weaker signal (weight 0.6) since agreement across rephrasings
is suggestive, not proof. 'unverifiable' claims are excluded from the rate
calculation entirely — an absence of evidence is not evidence of a problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .verifier_rules import VerificationResult, SUPPORTED, PARTIAL, HALLUCINATED, UNVERIFIABLE

_SCORE = {SUPPORTED: 1.0, PARTIAL: 0.5, HALLUCINATED: 0.0}


@dataclass
class ReportSummary:
    total_claims: int
    checkable_claims: int
    supported: int
    partially_supported: int
    hallucinated: int
    unverifiable: int
    hallucination_rate: float   # % of checkable claims that were hallucinated or partial, weighted
    trust_score: float          # 0-100, higher is better


def summarize(results: List[VerificationResult]) -> ReportSummary:
    total = len(results)
    checkable = [r for r in results if r.status != UNVERIFIABLE]
    supported = sum(1 for r in checkable if r.status == SUPPORTED)
    partial = sum(1 for r in checkable if r.status == PARTIAL)
    hallucinated = sum(1 for r in checkable if r.status == HALLUCINATED)
    unverifiable = total - len(checkable)

    if checkable:
        weighted = sum(_SCORE[r.status] for r in checkable) / len(checkable) * 100
        hallucination_rate = (partial * 0.5 + hallucinated) / len(checkable) * 100
    else:
        weighted = 0.0
        hallucination_rate = 0.0

    return ReportSummary(
        total_claims=total,
        checkable_claims=len(checkable),
        supported=supported,
        partially_supported=partial,
        hallucinated=hallucinated,
        unverifiable=unverifiable,
        hallucination_rate=round(hallucination_rate, 1),
        trust_score=round(weighted, 1),
    )
