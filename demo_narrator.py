"""
Generates a templated narrative WITHOUT calling any LLM API — with a
configurable number of intentionally injected numeric errors. This lets you
demo and grade the verifier end-to-end without needing an API key, and lets
evaluate.py generate labeled test data on demand.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import pandas as pd

from .utils import is_categorical_column


def generate_demo_narrative(df: pd.DataFrame, n_errors: int = 2, seed: int = 0) -> Tuple[str, List[bool]]:
    """
    Returns (narrative_text, ground_truth_is_hallucinated) where the list has
    one bool per generated sentence, in order, for use as an evaluation label set.
    """
    rng = random.Random(seed)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if is_categorical_column(df[c])]

    sentences: List[str] = []
    is_wrong: List[bool] = []

    def maybe_corrupt(value: float, force_wrong: bool) -> float:
        if force_wrong:
            return value * rng.choice([1.5, 1.8, 0.4, 0.3, 2.2])
        return value

    error_slots = set(rng.sample(range(6), min(n_errors, 6))) if numeric_cols or cat_cols else set()
    slot = 0

    if numeric_cols:
        col = rng.choice(numeric_cols)
        actual = df[col].mean()
        stated = maybe_corrupt(actual, slot in error_slots)
        sentences.append(f"The average {col} was {stated:.2f}.")
        is_wrong.append(slot in error_slots)
        slot += 1

    if numeric_cols:
        col = rng.choice(numeric_cols)
        actual = df[col].sum()
        stated = maybe_corrupt(actual, slot in error_slots)
        sentences.append(f"The total {col} amounted to {stated:.2f}.")
        is_wrong.append(slot in error_slots)
        slot += 1

    if cat_cols:
        col = rng.choice(cat_cols)
        vc = df[col].value_counts(normalize=True) * 100
        cat = vc.index[0]
        actual = vc.iloc[0]
        stated = maybe_corrupt(actual, slot in error_slots)
        sentences.append(f"{cat} accounts for {stated:.1f}% of {col}.")
        is_wrong.append(slot in error_slots)
        slot += 1

    if numeric_cols and cat_cols:
        ncol, ccol = rng.choice(numeric_cols), rng.choice(cat_cols)
        agg = df.groupby(ccol)[ncol].mean().sort_values(ascending=False)
        actual_top = agg.index[0]
        wrong = slot in error_slots
        named = actual_top if not wrong else rng.choice([c for c in agg.index if c != actual_top] or [actual_top])
        sentences.append(f"{named} had the highest {ncol} among all {ccol} groups.")
        is_wrong.append(wrong)
        slot += 1

    if len(numeric_cols) >= 2:
        c1, c2 = rng.sample(numeric_cols, 2)
        r = df[c1].corr(df[c2])
        actual_dir = "positive" if r > 0.1 else ("negative" if r < -0.1 else "no")
        wrong = slot in error_slots
        stated_dir = actual_dir if not wrong else {"positive": "negative", "negative": "positive", "no": "strong positive"}[actual_dir]
        sentences.append(f"There is a {stated_dir} correlation between {c1} and {c2}.")
        is_wrong.append(wrong)
        slot += 1

    if numeric_cols:
        col = rng.choice(numeric_cols)
        half = len(df) // 2
        first, second = df[col].iloc[:half].mean(), df[col].iloc[half:].mean()
        pct_change = (second - first) / abs(first) * 100 if first else 0
        actual_dir = "increased" if pct_change > 0 else "decreased"
        wrong = slot in error_slots
        stated_dir = actual_dir if not wrong else ("decreased" if actual_dir == "increased" else "increased")
        stated_pct = abs(pct_change) if not wrong else abs(pct_change) + rng.uniform(10, 20)
        sentences.append(f"{col} {stated_dir} by {stated_pct:.1f}% over the course of the dataset.")
        is_wrong.append(wrong)
        slot += 1

    narrative = " ".join(sentences)
    return narrative, is_wrong
