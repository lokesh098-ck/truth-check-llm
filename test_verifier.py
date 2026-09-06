import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from truthcheck.pipeline import run_pipeline
from truthcheck.verifier_rules import SUPPORTED, HALLUCINATED


@pytest.fixture
def df():
    return pd.DataFrame({
        "Region": ["North"] * 45 + ["South"] * 16 + ["East"] * 24 + ["West"] * 15,
        "Sales": [700] * 45 + [400] * 16 + [500] * 24 + [450] * 15,
        "MarketingSpend": [210] * 45 + [120] * 16 + [150] * 24 + [135] * 15,
    })


def _status_for(results, keyword):
    for r in results:
        if keyword.lower() in r.claim.text.lower():
            return r.status
    return None


def test_correct_average_is_supported(df):
    narrative = f"The average Sales was {df['Sales'].mean():.2f}."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert _status_for(results, "average") == SUPPORTED


def test_wrong_average_is_hallucinated(df):
    narrative = "The average Sales was 999999."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert _status_for(results, "average") == HALLUCINATED


def test_correct_percentage_is_supported(df):
    actual_pct = (df["Region"] == "North").mean() * 100
    narrative = f"North accounts for {actual_pct:.1f}% of Region."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert _status_for(results, "accounts for") == SUPPORTED


def test_wrong_percentage_is_hallucinated(df):
    narrative = "South accounts for 60% of Region."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert _status_for(results, "accounts for") == HALLUCINATED


def test_wrong_ranking_is_hallucinated(df):
    narrative = "West had the highest Sales."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert _status_for(results, "highest") == HALLUCINATED


def test_correct_ranking_is_supported(df):
    narrative = "North had the highest Sales."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert _status_for(results, "highest") == SUPPORTED


def test_correlation_direction_flip_is_hallucinated(df):
    narrative = "There is a negative correlation between MarketingSpend and Sales."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert _status_for(results, "correlation") == HALLUCINATED


def test_summary_counts_add_up(df):
    narrative = "The average Sales was 999999. North accounts for 45.0% of Region."
    results, summary = run_pipeline(narrative, df, check_qualitative=False)
    assert summary.total_claims == len(results)
    assert summary.supported + summary.partially_supported + summary.hallucinated + summary.unverifiable == summary.total_claims
