from httpx import ASGITransport, AsyncClient

from buddler import __version__
from buddler.api.main import create_app


async def test_health_returns_ok() -> None:
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
