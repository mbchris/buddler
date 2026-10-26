"""Token budgeting without a tokenizer dependency (AGENTS.md §3: explicit, never silent)."""

import logging
import math
from collections.abc import Sequence

from buddler.llm.ports import ChatMessage, TokenBudgetExceededError

logger = logging.getLogger(__name__)

# Conservative average for English/German text with BPE tokenizers.
CHARS_PER_TOKEN = 4
# Per-message framing overhead in chat formats.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Sequence[ChatMessage]) -> int:
    return sum(estimate_tokens(m.content) + MESSAGE_OVERHEAD_TOKENS for m in messages)


def ensure_within_budget(messages: Sequence[ChatMessage], max_input_tokens: int) -> int:
    estimated = estimate_messages_tokens(messages)
    if estimated > max_input_tokens:
        msg = f"Prompt needs ~{estimated} tokens, budget is {max_input_tokens}."
        raise TokenBudgetExceededError(msg)
    return estimated


def truncate_for_embedding(texts: Sequence[str], max_chars: int) -> tuple[list[str], int]:
    """Returns (texts, number truncated); truncation is logged by count, never content."""
    truncated = 0
    result: list[str] = []
    for text in texts:
        if len(text) > max_chars:
            truncated += 1
            text = text[:max_chars]
        result.append(text)
    if truncated:
        logger.warning("Truncated %d embedding inputs to %d chars", truncated, max_chars)
    return result, truncated
