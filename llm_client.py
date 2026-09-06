"""
Thin wrapper around an LLM API so the rest of the codebase never has to know
which provider is in use. Two responsibilities:

  1. narrate(df)   -> generate the initial analytical narrative to be checked
  2. complete(prompt) -> single-turn completion, used by the consistency checker

Configure via environment variables (see .env.example):
    LLM_PROVIDER=anthropic   (default; 'none' disables live calls entirely)
    ANTHROPIC_API_KEY=...
    LLM_MODEL=claude-sonnet-4-5   (check Anthropic's docs for the current model name)
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

        if self.provider == "anthropic" and self.api_key:
            import anthropic  # pip install anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        if not self.enabled:
            raise RuntimeError("LLM client is not configured — set ANTHROPIC_API_KEY to enable live calls.")
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")

    def narrate(self, df: pd.DataFrame) -> str:
        """Ask the LLM to write a short analytical narrative about the dataset."""
        sample = df.head(20).to_string(index=False)
        stats = df.describe(include="all").round(2).to_string()
        prompt = (
            "You are a data analyst. Write a short (5-8 sentence) narrative summarizing key patterns in this "
            "dataset for a non-technical reader — mention averages, notable percentages, which category is "
            "highest/lowest where relevant, and any correlation or trend you notice.\n\n"
            f"Columns: {list(df.columns)}\n\nSummary statistics:\n{stats}\n\nSample rows:\n{sample}"
        )
        return self.complete(prompt, max_tokens=500)
