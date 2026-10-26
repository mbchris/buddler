import json
from datetime import datetime
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from buddler.agents.graphs.explain import explanation
from buddler.agents.graphs.resolve import RESOLVE_NODE_ORDER, resolve_graph_mermaid
from buddler.agents.graphs.train import TRAIN_NODE_ORDER
from buddler.cli.ask import parse_at
from buddler.cli.main import app
from buddler.models.enums import ActionKind, RunKind
from buddler.schemas.knowledge import SeedFile
from tests.unit.workflow_fakes import make_settings

SEED = Path(__file__).parents[2] / "data" / "knowledge" / "seed.json"


def test_demo_seed_is_valid_and_covers_the_cold_scenarios() -> None:
    seed = SeedFile.model_validate(json.loads(SEED.read_text(encoding="utf-8")))

    cold = next(s for s in seed.symptoms if s.text == "feeling cold")
    kinds = {a.kind for a in cold.actions}
    assert {ActionKind.ADVICE, ActionKind.HEATING_DELTA} <= kinds


def test_seed_rejects_params_that_do_not_fit_the_kind() -> None:
    bad = {
        "symptoms": [
            {"text": "x", "actions": [{"kind": "heating_delta", "title": "t", "instruction": "i"}]}
        ]
    }

    with pytest.raises(ValueError, match="delta_c"):
        SeedFile.model_validate(bad)


@pytest.mark.parametrize(
    ("kind", "order"), [(RunKind.RESOLVE, RESOLVE_NODE_ORDER), (RunKind.TRAIN, TRAIN_NODE_ORDER)]
)
def test_every_node_is_explained(kind: RunKind, order: tuple[str, ...]) -> None:
    assert set(explanation(kind, make_settings()).steps) == set(order)


def test_mermaid_lists_all_resolve_nodes() -> None:
    source = resolve_graph_mermaid()

    assert all(node in source for node in RESOLVE_NODE_ORDER)


def test_parse_at_accepts_clock_time_and_iso() -> None:
    clock = parse_at("02:00")
    iso = parse_at("2026-11-10T13:00")

    assert clock is not None and (clock.hour, clock.minute) == (2, 0) and clock.tzinfo
    assert iso is not None
    assert iso == datetime.fromisoformat("2026-11-10T13:00").replace(tzinfo=iso.tzinfo)
    assert parse_at(None) is None
    with pytest.raises(typer.BadParameter):
        parse_at("tonight")


def test_cli_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("ask", "train", "knowledge", "home", "weather", "runs", "graph"):
        assert command in result.output
