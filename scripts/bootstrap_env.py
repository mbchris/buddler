"""Create `.env` from `.env.example` with generated local secrets. Never overwrites.

    uv run python scripts/bootstrap_env.py [--env .env] [--db-port 5432] [--mcp-port 8001]

`LLM_API_KEY` is taken from the environment if set; otherwise the offline fake LLM is
configured (`LLM_PROVIDER=fake`) so the demo still runs end to end.
"""

import argparse
import os
import secrets
import sys
from pathlib import Path

SECRET_BYTES = 24


def _set(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = prefix + value
            return lines
    return [*lines, prefix + value]


def build_env(template: str, *, db_port: int, mcp_port: int, llm_api_key: str) -> str:
    app_pw, mcp_pw = secrets.token_urlsafe(SECRET_BYTES), secrets.token_urlsafe(SECRET_BYTES)
    db = "buddler"
    values = {
        "DB_PORT": str(db_port),
        "MCP_PORT": str(mcp_port),
        "POSTGRES_PASSWORD": secrets.token_urlsafe(SECRET_BYTES),
        "POSTGRES_DB": db,
        "APP_RW_PASSWORD": app_pw,
        "MCP_RO_PASSWORD": mcp_pw,
        "DATABASE_URL": f"postgresql+asyncpg://app_rw:{app_pw}@localhost:{db_port}/{db}",
        "TEST_DATABASE_URL": f"postgresql+asyncpg://app_rw:{app_pw}@localhost:{db_port}/{db}_test",
        "TEST_MCP_RO_DATABASE_URL": (
            f"postgresql+asyncpg://mcp_ro:{mcp_pw}@localhost:{db_port}/{db}_test"
        ),
        "MCP_URL": f"http://localhost:{mcp_port}/sse",
        "API_KEY": secrets.token_urlsafe(SECRET_BYTES),
        "LLM_API_KEY": llm_api_key,
        "LLM_PROVIDER": "openai" if llm_api_key else "fake",
    }
    lines = template.splitlines()
    for key, value in values.items():
        lines = _set(lines, key, value)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--example", type=Path, default=Path(".env.example"))
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--mcp-port", type=int, default=8001)
    args = parser.parse_args()
    if args.env.exists():
        print(f"{args.env} exists, left unchanged.")
        return 0
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    args.env.write_text(
        build_env(
            args.example.read_text(encoding="utf-8"),
            db_port=args.db_port,
            mcp_port=args.mcp_port,
            llm_api_key=llm_api_key,
        ),
        encoding="utf-8",
    )
    args.env.chmod(0o600)
    print(f"Created {args.env} with generated secrets.")
    if not llm_api_key:
        print("LLM_API_KEY not set: LLM_PROVIDER=fake (offline). Set it in .env for real runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
