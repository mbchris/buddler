from dataclasses import dataclass, field
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

from buddler.mcp_client.client import (
    MAX_RESULT_CHARS,
    McpKnowledgeClient,
    _row_count,
    _single_exception,
)
from buddler.mcp_client.sql_guard import UnsafeQueryError, guard_sql


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT 1", "SELECT * FROM (SELECT 1) AS bounded LIMIT 10"),
        ("select * from knowledge.symptoms limit 5;", "select * from knowledge.symptoms limit 5"),
        (
            "WITH c AS (SELECT 1) SELECT * FROM c",
            "SELECT * FROM (WITH c AS (SELECT 1) SELECT * FROM c) AS bounded LIMIT 10",
        ),
        ("-- note\nSELECT 2", "SELECT * FROM (SELECT 2) AS bounded LIMIT 10"),
    ],
)
def test_guard_bounds_selects(sql: str, expected: str) -> None:
    assert guard_sql(sql, max_rows=10) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DELETE FROM knowledge.symptoms",
        "UPDATE x SET y = 1",
        "SELECT 1; DROP TABLE x",
        "/* SELECT */ INSERT INTO x VALUES (1)",
    ],
)
def test_guard_rejects_non_selects(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        guard_sql(sql, max_rows=10)


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("[{'n': 0}]", 1),
        ("[]", 0),
        ("[{'id': UUID('x')}]", 1),
        ("[{'n': Decimal('1')}, {'n': Decimal('2')}]", 2),
        ("Error: boom", None),
    ],
)
def test_row_count_from_python_repr(text: str, count: int | None) -> None:
    assert _row_count(text) == count


def test_single_exception_unwraps_nested_groups() -> None:
    inner = KeyError("x")

    group = ExceptionGroup("outer", [ExceptionGroup("inner", [inner])])

    assert _single_exception(group) is inner


@dataclass
class FakeSession:
    text: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def call_tool(
        self, name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> CallToolResult:
        self.calls.append({"name": name, **arguments})
        return CallToolResult(content=[TextContent(type="text", text=self.text)], is_error=False)


async def test_error_text_is_detected_and_query_is_bounded() -> None:
    session = FakeSession("Error: permission denied for schema app")
    client = McpKnowledgeClient(session, max_rows=5, timeout_s=1)  # type: ignore[arg-type]  # duck-typed session

    result = await client.execute_sql("SELECT * FROM app.run")

    assert result.is_error and result.row_count is None
    assert session.calls == [
        {"name": "execute_sql", "sql": "SELECT * FROM (SELECT * FROM app.run) AS bounded LIMIT 5"}
    ]


async def test_large_results_are_truncated() -> None:
    session = FakeSession("[" + ", ".join(["{'a': 1}"] * 2000) + "]")
    client = McpKnowledgeClient(session, max_rows=5, timeout_s=1)  # type: ignore[arg-type]  # duck-typed session

    result = await client.execute_sql("SELECT 1")

    assert result.truncated
    assert len(result.text) == MAX_RESULT_CHARS
    assert result.row_count == 2000
