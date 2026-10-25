"""Build the Homematic adapter from settings; the only place that knows implementations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2

from buddler.config import Settings
from buddler.home.ccu_jack import CcuJackHome
from buddler.home.fake import FakeHome
from buddler.home.ports import HomePort


def ccu_jack_client(
    settings: Settings, transport: httpx2.AsyncBaseTransport | None = None
) -> httpx2.AsyncClient:
    password = settings.homematic_password.get_secret_value()
    return httpx2.AsyncClient(
        base_url=settings.homematic_endpoint,
        auth=(settings.homematic_username, password) if settings.homematic_username else None,
        verify=settings.homematic_verify_tls,
        timeout=settings.homematic_timeout_s,
        transport=transport,
        trust_env=False,
    )


@asynccontextmanager
async def open_home(
    settings: Settings, transport: httpx2.AsyncBaseTransport | None = None
) -> AsyncIterator[HomePort]:
    if settings.homematic_provider == "fake":
        yield FakeHome()
        return
    async with ccu_jack_client(settings, transport) as client:
        yield CcuJackHome(client, settings.homematic_rooms)
