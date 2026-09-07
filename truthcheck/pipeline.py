"""
End-to-end pipeline: narrative + dataset -> list of VerificationResults + ReportSummary.
This is the single entry point the Streamlit app (and the evaluation script) call.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .claim_extractor import extract_claims
from .verifier_rules import verify_claim, VerificationResult
from .verifier_consistency import verify_qualitative
from .fusion import summarize, ReportSummary


def run_pipeline(narrative: str, df: pd.DataFrame, llm_client=None, check_qualitative: bool = True):
    """
    Returns (results, summary).
    If check_qualitative is False (or llm_client is None), qualitative claims
    are marked 'unverifiable' instead of consuming API calls — useful for a
    fast, free, offline demo.
    """
    claims = extract_claims(narrative, df)
    results: List[VerificationResult] = []

    for claim in claims:
        if claim.claim_type == "qualitative":
            if check_qualitative and llm_client is not None:
                results.append(verify_qualitative(claim, df, llm_client=llm_client))
            else:
                results.append(VerificationResult(
                    claim, "unverifiable",
                    evidence="Qualitative claim — enable the LLM consistency checker to assess this."
                ))
        else:
            results.append(verify_claim(claim, df))

    summary = summarize(results)
    return results, summary
