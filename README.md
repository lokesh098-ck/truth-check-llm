# TruthCheck-LLM

**A lightweight, dataset-grounded verification layer for LLM-generated data-analytics narratives.**

When you ask an LLM to "summarize this spreadsheet," it can produce fluent,
confident statements — averages, percentages, rankings, correlations — that
sound correct but aren't. TruthCheck-LLM sits between the LLM and the user,
independently recomputes every checkable claim directly from the source
dataset, and labels each one **Supported / Partially Supported /
Hallucinated**, with the actual value and a suggested correction shown
alongside.

No model training. No external knowledge base. Every number is recomputed
with plain pandas/numpy from data you already have.

---

## 1. How it works (architecture)

```
             ┌─────────────┐
  CSV/XLSX → │   Dataset   │
             └──────┬──────┘
                    │
        ┌───────────┴────────────┐
        │  LLM narrates the data │   (or: paste an existing narrative)
        └───────────┬────────────┘
                     │  narrative text
                     ▼
          ┌─────────────────────┐
          │   Claim Extractor    │  regex + fuzzy column matching
          │  (claim_extractor.py)│  -> average / sum / % / ranking /
          └──────────┬──────────┘     correlation / trend / qualitative
                     │
        ┌────────────┴─────────────┐
        ▼                          ▼
┌───────────────┐         ┌─────────────────────┐
│  Rule Verifier │         │ Consistency Verifier │
│ (deterministic,│         │ (LLM self-consistency,│
│  authoritative)│         │  qualitative claims   │
└───────┬───────┘         │  only, weaker signal) │
        │                 └──────────┬───────────┘
        └────────────┬───────────────┘
                     ▼
             ┌───────────────┐
             │ Fusion Layer   │  per-claim verdict + evidence
             │  (fusion.py)   │  + report-level Hallucination Rate
             └───────┬───────┘
                     ▼
             ┌───────────────┐
             │  Streamlit UI  │  color-coded claims, corrected text
             └───────────────┘
```

| Module | Responsibility |
|---|---|
| `truthcheck/claim_extractor.py` | Splits a narrative into sentences and pattern-matches each one into a structured `Claim` (type, matched columns, claimed value). |
| `truthcheck/verifier_rules.py` | Deterministically recomputes each claim type directly from the DataFrame (mean, sum, share, groupby ranking, correlation, first/second-half trend). This is the **authoritative** signal. |
| `truthcheck/verifier_consistency.py` | For claims that can't be recomputed (causal/qualitative statements), re-asks the LLM the same question multiple ways and checks agreement — a weaker, secondary signal, modeled on SelfCheckGPT. |
| `truthcheck/fusion.py` | Combines both signals into a final verdict per claim and one aggregate **Hallucination Rate** / **Trust Score** for the whole report. |
| `truthcheck/llm_client.py` | Thin wrapper around the Anthropic API — generates the initial narrative and answers consistency-check prompts. |
| `truthcheck/demo_narrator.py` | Generates a narrative with a known number of *injected* numeric errors, entirely offline — used for the evaluation harness and for demoing without an API key. |
| `truthcheck/pipeline.py` | Wires extraction → verification → fusion into one call. |
| `app.py` | Streamlit UI. |
| `evaluate.py` | Precision/Recall/F1 evaluation harness against synthetically labeled data. |

---

## 2. Project setup (local)

```bash
git clone <your-repo-url> truthcheck-llm     # or unzip the provided archive
cd truthcheck-llm

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY (optional — the app works
# offline without it, just without live narrative generation)
```

Run the tests to confirm everything works:

```bash
pytest tests/ -v
```

Run the app:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Check
"use bundled sample dataset" in the sidebar for an instant working demo,
or upload your own CSV.

---

## 3. Using it

1. **Upload a dataset** (CSV/XLSX) or use the bundled sample (`data/sample_sales.csv`).
2. **Get a narrative** — either:
   - Enter your Anthropic API key in the sidebar and click "Generate narrative from this data", or
   - Paste a narrative you already have (from ChatGPT, Copilot, another tool, etc.) into the "Paste your own" tab.
3. Click **Verify this narrative**.
4. Read the per-claim breakdown: each claim is labeled ✅ Supported, ⚠️ Partially supported, ❌ Hallucinated, or ❔ Unverifiable, with the recomputed evidence and — where relevant — a corrected sentence.
5. The top metrics show a **Trust Score** (0–100) and **Hallucination Rate** for the whole narrative, so you can compare different prompts, narratives, or LLMs against each other on the same dataset.

---

## 4. Evaluation

There's no existing public benchmark for "claims about a spreadsheet,
labeled hallucinated or not," so `evaluate.py` builds one on the fly using
`demo_narrator.py`, which knows the ground truth for every sentence it
generates (because it deliberately injects a controlled number of numeric
errors).

```bash
python evaluate.py --dataset data/sample_sales.csv --trials 40
python evaluate.py --all --trials 40     # every CSV under data/
```

This prints Accuracy / Precision / Recall / F1 for the rule-based verifier,
in the same style as a typical model-comparison table — except the
ground-truth generation process, not a public dataset, is what makes this
reproducible. Swap in your own labeled claims for a stronger evaluation once
you have some (see "Extending" below).

---

## 5. Extending the claim types

The regex patterns in `claim_extractor.py` cover the most common analytical
claim shapes (average, sum, percentage, ranking, correlation, trend). To add
a new claim type:

1. Add a regex to `_PATTERNS` in `claim_extractor.py` and populate a `Claim`.
2. Add a matching `verify_<type>()` function to `verifier_rules.py` that
   recomputes the claim from the DataFrame and returns a `VerificationResult`.
3. Register it in `_DISPATCH` at the bottom of `verifier_rules.py`.
4. Add a test case to `tests/test_verifier.py`.

---

## 6. Deployment

### Option A — Streamlit Community Cloud (easiest, free)

1. Push this project to a GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick your repo/branch, and set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "your-key-here"
   ```
5. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

### Option B — Docker (any cloud VM / Render / Railway / Fly.io)

```bash
docker build -t truthcheck-llm .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your-key-here truthcheck-llm
```

Then point any platform that runs a Dockerfile (Render, Railway, Fly.io, an
EC2/VM instance behind nginx, etc.) at this image, exposing port `8501`.

### Option C — Local network demo only

`streamlit run app.py --server.address 0.0.0.0` and share your machine's
LAN IP — fine for a classroom/review demo, not for production.

---

## 7. Known limitations (be upfront about these in your report)

- **Trend claims** use row order as a proxy for time when no explicit date
  column is detected — flag this in your writeup as a simplification, and
  extend `verify_trend` to look for a real date/time column if your data has one.
- **Claim extraction is regex-based**, not a trained NLP model — it covers
  common phrasings but will miss unusual sentence structures. This is a
  deliberate simplicity/transparency trade-off (see Research Contributions
  below), not a bug, but it's worth stating as a limitation in your report.
- **Qualitative/causal claims** (e.g. "this is likely due to seasonal
  demand") cannot be deterministically verified at all — the consistency
  checker is a weaker, secondary signal for these, not a factual arbiter.

---

## 8. Why this project is novel (for your write-up)

- **Domain**: existing hallucination-detection work (SelfCheckGPT,
  FActScore) targets open-domain text; VeriFin targets financial XBRL
  filings specifically. This project generalizes the "ground every operand
  in source data" principle to *any* CSV/spreadsheet — a far more common,
  everyday use case (BI dashboards, spreadsheet copilots, survey analysis).
- **Hybrid fusion**: pure rule-based recomputation (fast, deterministic,
  explainable) is combined with LLM self-consistency (for the claims rules
  can't reach), following the fusion-layer idea from hybrid
  ML+LLM+RAG misinformation-detection architectures, but applied to a
  fundamentally different verification problem.
- **No training required**: unlike ML-classifier-based approaches, this
  needs zero labeled training data or model fine-tuning to deploy on a new
  dataset — it works out of the box on any tabular data.
- **New evaluation methodology**: since no public "hallucinated tabular
  claim" benchmark exists, this project also contributes a synthetic,
  reproducible evaluation harness (`evaluate.py` + `demo_narrator.py`) that
  future work in this space can reuse or extend.
