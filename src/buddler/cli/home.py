"""`buddler home …` / `buddler weather`: inspect the home and the weather source."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from buddler.config import get_settings
from buddler.home.ccu_jack import CcuJackHome, DiscoveredChannel
from buddler.home.factory import ccu_jack_client, open_home
from buddler.home.ports import HomeError
from buddler.services.home_service import RoomStatus, room_overview
from buddler.weather.factory import open_weather
from buddler.weather.ports import WeatherError, WeatherObservation

console = Console()
home_app = typer.Typer(no_args_is_help=True, help="Inspect the Homematic home (CCU-Jack).")


def _value(value: float | bool | None, unit: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "open" if value else "closed"
    return f"{value:g}{unit}"


async def _rooms() -> tuple[str, list[RoomStatus]]:
    settings = get_settings()
    async with open_home(settings) as home:
        return home.name, await room_overview(home, settings.actuation_rooms)


@home_app.command("rooms")
def rooms() -> None:
    """Configured rooms with their current climate."""
    name, statuses = asyncio.run(_rooms())
    if not statuses:
        console.print("No rooms configured: set HOMEMATIC_ROOMS (see `buddler home discover`).")
        return
    table = Table("room", "actual", "setpoint", "humidity", "window", "writable", title=name)
    for status in statuses:
        climate = status.climate
        if climate is None:
            table.add_row(status.room, f"[red]{status.error}[/red]", "", "", "", "")
            continue
        table.add_row(
            status.room + (f" ({', '.join(status.aliases)})" if status.aliases else ""),
            _value(climate.actual_c, " C"),
            _value(climate.setpoint_c, " C"),
            _value(climate.humidity_pct, " %"),
            _value(climate.window_open),
            "yes" if status.writable else "no",
        )
    console.print(table)


async def _discover() -> list[DiscoveredChannel]:
    settings = get_settings()
    async with ccu_jack_client(settings) as client:
        return await CcuJackHome(client, {}).discover()


@home_app.command("discover")
def discover(all_channels: bool = typer.Option(False, "--all", help="Show every channel.")) -> None:
    """List CCU rooms and channels; climate channels go into HOMEMATIC_ROOMS."""
    try:
        channels = asyncio.run(_discover())
    except HomeError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    table = Table("room", "address", "channel", "climate", "parameters")
    for channel in channels:
        if all_channels or channel.is_climate:
            table.add_row(
                channel.room,
                channel.address,
                channel.title,
                "yes" if channel.is_climate else "",
                ", ".join(channel.parameters),
            )
    console.print(table)
    console.print(
        'Example: HOMEMATIC_ROOMS=\'{"living room": {"climate": "<address>", '
        '"aliases": ["wohnzimmer"]}}\'',
        markup=False,
    )


async def _weather() -> WeatherObservation:
    async with open_weather(get_settings()) as weather:
        return await weather.current()


def weather() -> None:
    """Current weather and sunshine forecast for the configured location."""
    try:
        observation = asyncio.run(_weather())
    except WeatherError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    table = Table("field", "value", title=observation.source)
    table.add_row("outdoor", _value(observation.outdoor_temp_c, " C"))
    table.add_row("cloud cover", _value(observation.cloud_cover_pct, " %"))
    table.add_row(
        f"sunshine next {observation.forecast_hours} h",
        _value(observation.sun_hours_forecast, " h"),
    )
    console.print(table)
