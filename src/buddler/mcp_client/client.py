"""Postgres MCP Pro client: the agent's only database access (AGENTS.md §5)."""

import ast
import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx2
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult

from buddler.mcp_client.sql_guard import guard_sql

logger = logging.getLogger(__name__)

EXECUTE_SQL_TOOL = "execute_sql"
# Only tool the agent may call; schema info is given in the prompt.
ALLOWED_TOOLS = frozenset({EXECUTE_SQL_TOOL})
# Postgres MCP Pro reports SQL errors as normal text results.
ERROR_PREFIX = "Error:"
MAX_RESULT_CHARS = 6000


class McpUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryResult:
    sql: str
    text: str
    is_error: bool
    row_count: int | None
    latency_ms: int
    truncated: bool


# Row starts in a repr like "[{'a': Decimal('1')}, {'a': ...}]".
_ROW_START = re.compile(r"(?:^\[|\}, )\{")


def _row_count(text: str) -> int | None:
    """Results are Python reprs (not JSON); parse when possible, else count row starts."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # Reprs with Decimal(...)/UUID(...)/datetime(...) are not literals.
        return len(_ROW_START.findall(text)) if text.startswith("[{") else None
    return len(value) if isinstance(value, list) else None


class McpKnowledgeClient:
    def __init__(self, session: ClientSession, *, max_rows: int, timeout_s: float) -> None:
        self._session = session
        self._max_rows = max_rows
        self._timeout_s = timeout_s

    async def execute_sql(self, sql: str) -> QueryResult:
        bounded = guard_sql(sql, self._max_rows)
        started = time.perf_counter()
        result = await self._session.call_tool(
            EXECUTE_SQL_TOOL,
            {"sql": bounded},
            read_timeout_seconds=self._timeout_s,
        )
        if not isinstance(result, CallToolResult):
            raise McpUnavailableError(f"Unexpected MCP response type {type(result).__name__}")
        text = "\n".join(getattr(block, "text", "") for block in result.content).strip()
        is_error = result.is_error or text.startswith(ERROR_PREFIX)
        truncated = len(text) > MAX_RESULT_CHARS
        return QueryResult(
            sql=bounded,
            text=text[:MAX_RESULT_CHARS],
            is_error=is_error,
            row_count=None if is_error else _row_count(text),
            latency_ms=int((time.perf_counter() - started) * 1000),
            truncated=truncated,
        )


def _single_exception(group: BaseExceptionGroup[BaseException]) -> BaseException:
    """anyio task groups wrap errors; unwrap nested single-exception groups."""
    error: BaseException = group
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


@asynccontextmanager
async def connect_knowledge_mcp(
    url: str, *, max_rows: int, timeout_s: float
) -> AsyncIterator[McpKnowledgeClient]:
    """Connect and verify the allowlisted tool exists.

    Connection failures raise McpUnavailableError; errors from the caller's block are
    re-raised unwrapped (the SSE transport's task group would wrap them otherwise).
    """
    connected = False
    try:
        async with AsyncExitStack() as stack:
            async with asyncio.timeout(timeout_s):
                read, write = await stack.enter_async_context(sse_client(url))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
            if not tools >= ALLOWED_TOOLS:
                missing = sorted(ALLOWED_TOOLS - tools)
                raise McpUnavailableError(f"MCP server lacks tools {missing}")
            logger.info("Connected to MCP server %s (%d tools)", url, len(tools))
            connected = True
            yield McpKnowledgeClient(session, max_rows=max_rows, timeout_s=timeout_s)
    except BaseExceptionGroup as group:
        error = _single_exception(group)
        if connected or isinstance(error, McpUnavailableError):
            raise error from None
        raise McpUnavailableError(f"MCP server at {url} unreachable: {error!r}") from error
    except (OSError, TimeoutError, httpx2.HTTPError) as err:
        if connected:
            raise
        raise McpUnavailableError(f"MCP server at {url} unreachable: {err!r}") from err
