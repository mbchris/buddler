import re
import uuid
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from buddler.api.main import create_app
from buddler.api.routers.ui import STATIC_DIR
from buddler.config import Settings


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app(Settings(_env_file=None, llm_provider="fake"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.mark.parametrize("path", ["/ui", f"/ui/runs/{uuid.uuid4()}"])
async def test_pages_are_public_html_with_csp(client: AsyncClient, path: str) -> None:
    response = await client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    csp = response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "connect-src 'self'" in csp


async def test_run_page_rejects_non_uuid(client: AsyncClient) -> None:
    assert (await client.get("/ui/runs/../../etc")).status_code == 404


@pytest.mark.parametrize(
    ("asset", "media"),
    [("app.js", "javascript"), ("styles.css", "text/css")],
)
async def test_static_assets_are_served(client: AsyncClient, asset: str, media: str) -> None:
    response = await client.get(f"/ui/static/{asset}")

    assert response.status_code == 200
    assert media in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache"


def test_cdn_script_is_pinned_with_sri() -> None:
    html = (STATIC_DIR / "index.html").read_text()

    for tag in re.findall(r"<script[^>]+cdn\.jsdelivr\.net[^>]*>", html):
        assert re.search(r"@\d+\.\d+\.\d+/", tag), "CDN script must pin an exact version"
        assert 'integrity="sha384-' in tag


def test_untrusted_data_is_never_written_as_html() -> None:
    script = (STATIC_DIR / "app.js").read_text()

    # Single exception: Mermaid SVG generated from our own graph definition.
    assert re.findall(r"\w+\.innerHTML\s*=.*", script) == ["container.innerHTML = svg;"]
    assert "insertAdjacentHTML" not in script and "outerHTML" not in script
