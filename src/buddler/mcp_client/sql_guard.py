"""Client-side SQL guard for agent queries (AGENTS.md §5/§6).

Defense in depth: the MCP server runs in restricted mode and the DB role is read-only;
this layer rejects anything that is not a single SELECT and bounds the result size.
"""

import re

ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+\s*$", re.IGNORECASE)
COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


class UnsafeQueryError(ValueError):
    pass


def guard_sql(sql: str, max_rows: int) -> str:
    """Return a bounded, single-statement SELECT or raise UnsafeQueryError."""
    statement = COMMENT.sub(" ", sql).strip().rstrip(";").strip()
    if not statement:
        raise UnsafeQueryError("Empty query.")
    if ";" in statement:
        raise UnsafeQueryError("Only a single statement is allowed.")
    if not ALLOWED_START.match(statement):
        raise UnsafeQueryError("Only SELECT/WITH queries are allowed.")
    if HAS_LIMIT.search(statement):
        return statement
    return f"SELECT * FROM ({statement}) AS bounded LIMIT {max_rows}"
