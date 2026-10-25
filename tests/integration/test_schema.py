import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from buddler.models import EMBEDDING_DIM, Action, Symptom
from buddler.models.enums import ActionKind, KnowledgeStatus, Provenance
from buddler.repositories.knowledge_repository import KnowledgeRepository


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


def _symptom(
    text_: str, status: KnowledgeStatus, index: int, action_status: KnowledgeStatus
) -> Symptom:
    symptom = Symptom(
        text=text_, embedding=_unit_vector(index), status=status, provenance=Provenance.HUMAN
    )
    symptom.actions.append(
        Action(
            kind=ActionKind.ADVICE,
            title="Tea",
            instruction="Drink tea.",
            params={},
            conditions={"time_from": "22:00:00", "time_to": "06:00:00"},
            priority=10,
            status=action_status,
            provenance=Provenance.HUMAN,
        )
    )
    return symptom


async def test_symptom_vector_roundtrip(session: AsyncSession) -> None:
    symptom = _symptom("feeling cold", KnowledgeStatus.CONFIRMED, 0, KnowledgeStatus.CONFIRMED)
    session.add(symptom)
    await session.flush()

    stored = await session.scalar(select(Symptom.embedding).where(Symptom.id == symptom.id))

    assert stored is not None and list(stored) == _unit_vector(0)


async def test_nearest_symptoms_orders_by_distance_and_filters_status(
    session: AsyncSession,
) -> None:
    near = _symptom("feeling cold", KnowledgeStatus.CONFIRMED, 0, KnowledgeStatus.CONFIRMED)
    far = _symptom("feeling warm", KnowledgeStatus.CONFIRMED, 1, KnowledgeStatus.CONFIRMED)
    proposed = _symptom("freezing", KnowledgeStatus.PROPOSED, 0, KnowledgeStatus.PROPOSED)
    session.add_all([far, near, proposed])
    await session.flush()

    matches = await KnowledgeRepository(session).nearest_symptoms(
        _unit_vector(0), top_k=5, max_distance=2
    )

    assert [m.symptom.text for m in matches] == ["feeling cold", "feeling warm"]
    assert matches[0].distance == pytest.approx(0) and len(matches[0].symptom.actions) == 1


async def test_knowledge_views_expose_only_confirmed(session: AsyncSession) -> None:
    confirmed = _symptom("feeling cold", KnowledgeStatus.CONFIRMED, 0, KnowledgeStatus.CONFIRMED)
    pending = _symptom("feeling warm", KnowledgeStatus.CONFIRMED, 1, KnowledgeStatus.PROPOSED)
    session.add_all([confirmed, pending])
    await session.flush()

    symptoms = set((await session.execute(text("SELECT text FROM knowledge.symptoms"))).scalars())
    actions = set((await session.execute(text("SELECT symptom FROM knowledge.actions"))).scalars())

    assert {"feeling cold", "feeling warm"} <= symptoms
    assert "feeling cold" in actions and "feeling warm" not in actions


async def test_knowledge_views_hide_vectors_and_user_text(session: AsyncSession) -> None:
    rows = await session.execute(
        text(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'knowledge'"
        )
    )
    columns = {column for _, column in rows}

    assert columns
    assert {"embedding", "subject", "body", "payload"}.isdisjoint(columns)


async def test_mcp_role_can_read_knowledge(mcp_ro_engine: AsyncEngine) -> None:
    async with mcp_ro_engine.connect() as connection:
        result = await connection.execute(text("SELECT count(*) FROM knowledge.device_commands"))

    assert result.scalar_one() >= 0


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM app.symptom",
        "SELECT subject FROM app.run",
        "INSERT INTO knowledge.symptoms (id) VALUES (gen_random_uuid())",
        "DELETE FROM app.action",
        "CREATE TABLE knowledge.evil (id int)",
    ],
)
async def test_mcp_role_cannot_access_app_or_write(
    mcp_ro_engine: AsyncEngine, statement: str
) -> None:
    async with mcp_ro_engine.connect() as connection:
        with pytest.raises(DBAPIError):
            await connection.execute(text(statement))
