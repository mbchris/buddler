"""Demonstrator trace UI (design §6): static page; all data comes from the authenticated API."""

import uuid
from importlib import resources
from pathlib import Path

from fastapi import APIRouter, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

STATIC_DIR = Path(str(resources.files("buddler.ui") / "static"))
INDEX = STATIC_DIR / "index.html"

# The page holds no data, so it is public; API calls carry the bearer key.
# 'unsafe-inline' styles: Mermaid writes inline SVG styles. Scripts: self + pinned CDN (SRI).
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
# Revalidate on every load (cheap 304 via ETag): a restarted server must never be
# paired with stale JS from the browser's heuristic cache.
NO_CACHE = {"Cache-Control": "no-cache"}
SECURITY_HEADERS = {
    **NO_CACHE,
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

router = APIRouter(prefix="/ui", include_in_schema=False)


class RevalidatedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


def _index() -> FileResponse:
    return FileResponse(INDEX, media_type="text/html", headers=SECURITY_HEADERS)


@router.get("", response_class=FileResponse, status_code=status.HTTP_200_OK)
async def run_list_page() -> FileResponse:
    return _index()


@router.get("/runs/{run_id}", response_class=FileResponse, status_code=status.HTTP_200_OK)
async def run_page(run_id: uuid.UUID) -> FileResponse:
    return _index()
