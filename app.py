import os

import pandas as pd
import streamlit as st

from truthcheck.pipeline import run_pipeline
from truthcheck.llm_client import LLMClient
from truthcheck.verifier_rules import SUPPORTED, PARTIAL, HALLUCINATED, UNVERIFIABLE

st.set_page_config(page_title="TruthCheck-LLM", page_icon="✅", layout="wide")

STATUS_STYLE = {
    SUPPORTED: ("✅ Supported", "#1e7e34", "#e6f4ea"),
    PARTIAL: ("⚠️ Partially supported", "#8a6d00", "#fff8e1"),
    HALLUCINATED: ("❌ Hallucinated", "#a11", "#fdecea"),
    UNVERIFIABLE: ("❔ Unverifiable", "#555", "#f1f1f1"),
}


def load_dataset(uploaded):
    """Read an uploaded CSV/Excel file and validate it. Returns a DataFrame or None."""
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            raise ValueError("Unsupported file type. Upload a CSV or Excel file.")

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")
        if len(df.columns) < 1:
            raise ValueError("No columns were detected.")

        df.columns = df.columns.astype(str).str.strip()
        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names found. Rename them to make every column unique.")
        return df
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return None


st.title("✅ TruthCheck-LLM")
st.caption(
    "A lightweight, dataset-grounded verification layer for LLM-generated data narratives. "
    "Every numeric claim is independently recomputed from your own data — no external knowledge base, "
    "no model training required."
)

with st.sidebar:
    st.header("1. Upload your data")
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    use_sample = st.checkbox("Use bundled sample dataset instead", value=uploaded is None)

    st.header("2. LLM settings (optional)")
    api_key_input = st.text_input(
        "Anthropic API key", type="password",
        help="Leave blank to use ANTHROPIC_API_KEY from the environment, or to run in offline demo mode.")
    check_qualitative = st.checkbox(
        "Also check qualitative/causal claims (uses more API calls)", value=False)

df = None
if use_sample:
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_sales.csv")
    try:
        df = pd.read_csv(sample_path)
        st.sidebar.info("Using bundled sample_sales.csv (Region, Sales, MarketingSpend, Quarter).")
    except FileNotFoundError:
        st.error(f"Sample dataset not found at {sample_path}. Upload your own file instead.")
        st.stop()
elif uploaded is not None:
    df = load_dataset(uploaded)

if df is None:
    st.info("Upload a dataset or check 'use bundled sample dataset' in the sidebar to get started.")
    st.stop()

st.subheader("Your data")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"{len(df):,} rows × {len(df.columns)} columns.")

api_key = api_key_input or os.getenv("ANTHROPIC_API_KEY")
llm_client = LLMClient(api_key=api_key) if api_key else None

st.subheader("Narrative to verify")
tab_generate, tab_paste = st.tabs(["🪄 Generate with LLM", "📋 Paste your own"])

with tab_generate:
    if llm_client and llm_client.enabled:
        if st.button("Generate narrative from this data"):
            with st.spinner("Asking the LLM to summarize the dataset..."):
                st.session_state["narrative"] = llm_client.narrate(df)
    else:
        st.warning("Add an API key in the sidebar to generate a narrative live. "
                   "You can still paste one manually in the other tab.")

with tab_paste:
    pasted = st.text_area(
        "Paste an LLM-generated narrative about this dataset",
        value=st.session_state.get("narrative", ""),
        height=160,
        placeholder="e.g. 'The average Sales was 612.4. North accounts for 45% of all regions...'",
    )
    if pasted:
        st.session_state["narrative"] = pasted

narrative = st.session_state.get("narrative", "")

if narrative:
    st.text_area("Narrative being checked", narrative, height=120, disabled=True)

    if st.button("🔍 Verify this narrative", type="primary"):
        with st.spinner("Extracting and verifying claims..."):
            results, summary = run_pipeline(
                narrative, df,
                llm_client=llm_client if check_qualitative else None,
                check_qualitative=check_qualitative,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trust score", f"{summary.trust_score}/100")
        c2.metric("Hallucination rate", f"{summary.hallucination_rate}%")
        c3.metric("Claims checked", summary.checkable_claims)
        c4.metric("Hallucinated", summary.hallucinated)

        st.divider()
        st.subheader("Per-claim breakdown")

        for r in results:
            label, fg, bg = STATUS_STYLE[r.status]
            with st.container(border=True):
                st.markdown(
                    f"<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:6px;"
                    f"font-weight:600;font-size:0.85em'>{label}</span>",
                    unsafe_allow_html=True,
                )
                st.write(r.claim.text)
                st.caption(r.evidence)
                if r.corrected_text:
                    st.success(f"Suggested correction: {r.corrected_text}")
else:
    st.info("Generate or paste a narrative above, then click Verify.")

st.divider()
st.caption(
    "TruthCheck-LLM verifies numeric/statistical claims (averages, sums, percentages, rankings, "
    "correlations, trends) deterministically against your dataset, and — optionally — cross-checks "
    "qualitative/causal claims via LLM self-consistency. Rule-based results are authoritative; "
    "consistency-check results are a secondary, weaker signal."
)
