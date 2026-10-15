"""
TruthCheck-LLM — Streamlit demo app.

Run locally:   streamlit run app.py
Deploy:        see README.md (Streamlit Community Cloud / Docker)
"""
import io
import pandas as pd
import streamlit as st

def load_dataset(uploaded):
    filename = uploaded.name.lower()
    raw = uploaded.getvalue()

    try:
        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))

        elif filename.endswith(".csv"):
            errors = []

            for encoding in ("utf-8-sig", "cp1252", "latin1"):
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw),
                        sep=None,
                        engine="python",
                        encoding=encoding
                    )
                    break
                except (pd.errors.ParserError,
                        UnicodeDecodeError,
                        ValueError) as e:
                    errors.append(str(e))
            else:
                raise ValueError(
                    "Could not parse this CSV. Check its "
                    "delimiter, column headers, and row structure."
                )

        else:
            raise ValueError(
                "Unsupported file type. Upload a CSV or Excel file."
            )

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        if len(df.columns) < 1:
            raise ValueError("No columns were detected.")

        # Clean column names
        df.columns = df.columns.astype(str).str.strip()

        if df.columns.duplicated().any():
            raise ValueError(
                "Duplicate column names found. Rename them "
                "to make every column unique."
            )

        return df

    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return None


uploaded = st.file_uploader(
    "Upload your data",
    type=["csv", "xlsx", "xls"]
)

if uploaded is not None:
    df = load_dataset(uploaded)

    if df is not None:
        st.success(
            f"Dataset loaded: {len(df)} rows, "
            f"{len(df.columns)} columns."
        )
        st.dataframe(df.head(10))
