"""LangGraph workflows. Nodes reach DB/LLM only through injected ports (AGENTS.md §2)."""

import os

# LangSmith ships with LangGraph and would upload traces incl. prompts (AGENTS.md §7).
for _flag in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
    os.environ[_flag] = "false"
