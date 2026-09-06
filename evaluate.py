"""
Evaluation harness for TruthCheck-LLM's rule-based verifier.

Since there is no existing public benchmark for "claims about a tabular
dataset labeled hallucinated/not", this script builds one on the fly: for
each dataset, it generates a templated narrative with a known number of
injected numeric errors (demo_narrator.py knows the ground truth for every
sentence), runs the verifier, and compares its per-claim verdicts against
that ground truth to compute Precision / Recall / F1 — mirroring the style
of Table 2 in the base paper this project takes methodological inspiration
from, but for a benchmark that is newly constructed here.

Usage:
    python evaluate.py --dataset data/sample_sales.csv --trials 30
    python evaluate.py --all        # runs against every CSV in data/
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import pandas as pd

from truthcheck.demo_narrator import generate_demo_narrative
from truthcheck.pipeline import run_pipeline
from truthcheck.verifier_rules import SUPPORTED, HALLUCINATED, UNVERIFIABLE


@dataclass
class Confusion:
    tp: int = 0  # correctly flagged as hallucinated
    fp: int = 0  # flagged as hallucinated but was actually fine
    fn: int = 0  # was hallucinated but verifier missed it (said supported/partial)
    tn: int = 0  # correctly identified as fine
    skipped: int = 0  # verifier said 'unverifiable' — excluded from P/R/F1


def evaluate_dataset(path: str, trials: int = 30, max_errors: int = 3) -> Confusion:
    df = pd.read_csv(path)
    conf = Confusion()

    for seed in range(trials):
        n_errors = seed % (max_errors + 1)
        narrative, ground_truth = generate_demo_narrative(df, n_errors=n_errors, seed=seed)
        results, _ = run_pipeline(narrative, df, check_qualitative=False)

        # results are produced in the same sentence order generate_demo_narrative used
        for result, truth_is_wrong in zip(results, ground_truth):
            predicted_wrong = result.status == HALLUCINATED
            if result.status == UNVERIFIABLE:
                conf.skipped += 1
                continue
            if truth_is_wrong and predicted_wrong:
                conf.tp += 1
            elif truth_is_wrong and not predicted_wrong:
                conf.fn += 1
            elif not truth_is_wrong and predicted_wrong:
                conf.fp += 1
            else:
                conf.tn += 1
    return conf


def report(name: str, conf: Confusion):
    precision = conf.tp / (conf.tp + conf.fp) if (conf.tp + conf.fp) else float("nan")
    recall = conf.tp / (conf.tp + conf.fn) if (conf.tp + conf.fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (conf.tp + conf.tn) / (conf.tp + conf.tn + conf.fp + conf.fn) if (conf.tp + conf.tn + conf.fp + conf.fn) else float("nan")

    print(f"\n=== {name} ===")
    print(f"  TP={conf.tp}  FP={conf.fp}  FN={conf.fn}  TN={conf.tn}  (skipped/unverifiable: {conf.skipped})")
    print(f"  Accuracy:  {accuracy*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to a single CSV to evaluate.")
    parser.add_argument("--all", action="store_true", help="Evaluate every CSV under data/.")
    parser.add_argument("--trials", type=int, default=30, help="Number of synthetic narratives per dataset.")
    args = parser.parse_args()

    if args.all or args.dataset is None:
        paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.csv")))
    else:
        paths = [args.dataset]

    overall = Confusion()
    for path in paths:
        conf = evaluate_dataset(path, trials=args.trials)
        report(os.path.basename(path), conf)
        overall.tp += conf.tp
        overall.fp += conf.fp
        overall.fn += conf.fn
        overall.tn += conf.tn
        overall.skipped += conf.skipped

    if len(paths) > 1:
        report("OVERALL", overall)


if __name__ == "__main__":
    main()
