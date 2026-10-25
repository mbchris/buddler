"""In-memory home for tests and offline demos (`HOMEMATIC_PROVIDER=fake`)."""

from dataclasses import dataclass, field

from buddler.home.ports import RoomClimate, UnknownRoomError


def demo_rooms() -> dict[str, RoomClimate]:
    return {
        "living room": RoomClimate(
            room="living room", actual_c=19.5, setpoint_c=20.0, humidity_pct=48, window_open=False
        ),
        "bedroom": RoomClimate(
            room="bedroom", actual_c=17.5, setpoint_c=18.0, humidity_pct=55, window_open=False
        ),
        "office": RoomClimate(
            room="office", actual_c=21.0, setpoint_c=21.0, humidity_pct=63, window_open=True
        ),
    }


DEMO_ALIASES: dict[str, list[str]] = {
    "living room": ["wohnzimmer", "lounge"],
    "bedroom": ["schlafzimmer"],
    "office": ["büro", "study"],
}


@dataclass
class FakeHome:
    climates: dict[str, RoomClimate] = field(default_factory=demo_rooms)
    aliases: dict[str, list[str]] = field(default_factory=lambda: dict(DEMO_ALIASES))
    writes: list[tuple[str, float]] = field(default_factory=list)
    name: str = "fake home"

    def rooms(self) -> dict[str, list[str]]:
        return {room: self.aliases.get(room, []) for room in self.climates}

    def _get(self, room: str) -> RoomClimate:
        try:
            return self.climates[room]
        except KeyError as err:
            raise UnknownRoomError(f"Unknown room {room!r}.") from err

    async def read_climate(self, room: str) -> RoomClimate:
        return self._get(room).model_copy()

    async def set_setpoint(self, room: str, value_c: float) -> None:
        climate = self._get(room)
        self.climates[room] = climate.model_copy(update={"setpoint_c": value_c})
        self.writes.append((room, value_c))
