"""Resolve workflow nodes (design §3): situation -> observations -> knowledge -> decision -> act."""

import json
import re
import time
from collections.abc import Sequence

from buddler.agents.ports import CommandRecord, ResolveDeps
from buddler.agents.schemas import AgentStep, DecisionDraft, UnderstandingDraft
from buddler.agents.state import (
    ActionOutcome,
    AdviceItem,
    AgentFindings,
    Candidate,
    ChosenAction,
    Resolution,
    ResolveState,
    RoomSource,
    ToolCallRecord,
    Understanding,
)
from buddler.agents.tracing import NodeUpdate
from buddler.home.ccu_jack import SET_POINT_TEMPERATURE
from buddler.home.ports import HomeError, RoomClimate
from buddler.knowledge.actuation import (
    ActuationBounds,
    GuardDecision,
    check_runtime,
    check_static,
)
from buddler.knowledge.conditions import (
    CheckOutcome,
    ConditionCheck,
    Observations,
    evaluate_conditions,
)
from buddler.llm.ports import ChatMessage, LLMError
from buddler.mcp_client.client import McpUnavailableError
from buddler.mcp_client.sql_guard import UnsafeQueryError
from buddler.models.enums import ActionKind, CommandStatus, RunKind, TraceEventType
from buddler.prompts.loader import load_prompt
from buddler.tracing.tracer import SpanResult
from buddler.weather.ports import WeatherError, WeatherObservation

UNDERSTAND_PROMPT = "understand"
AGENT_PROMPT = "agent"
DECIDE_PROMPT = "decide"
# Tool output is untrusted and token-hungry: bounded before it re-enters the prompt.
MAX_OBSERVATION_CHARS = 3000
MAX_SUMMARY_ARG_CHARS = 120
MAX_STEP_SUMMARY_CHARS = 160
MAX_SYMPTOM_CHARS = 120
REF_PATTERN = re.compile(r"\b([SAO]\d+)\b")
OBSERVATION_REFS = ("O1", "O2", "O3")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def match_room(text: str, rooms: dict[str, list[str]]) -> str | None:
    """Exact name/alias match first, then containment ("the living room" -> "living room")."""
    wanted = _normalize(text)
    names = {_normalize(name): name for name in rooms} | {
        _normalize(alias): name for name, aliases in rooms.items() for alias in aliases
    }
    if wanted in names:
        return names[wanted]
    for key, name in sorted(names.items(), key=lambda item: -len(item[0])):
        if key and (key in wanted or wanted in key):
            return name
    return None


def _refs(texts: Sequence[str]) -> list[str]:
    """Accept "A1", "[A1]", "A1, O2" …; normalized and de-duplicated in order."""
    found = [match.upper() for text in texts for match in REF_PATTERN.findall(text.upper())]
    return list(dict.fromkeys(found))


class ResolveNodes:
    def __init__(self, deps: ResolveDeps) -> None:
        self._deps = deps
        self._bounds = ActuationBounds.from_settings(deps.settings)

    # --- understand / locate / observe -------------------------------------

    async def understand(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        prompt = load_prompt(RunKind.RESOLVE, UNDERSTAND_PROMPT)
        situation = state["situation"]
        try:
            result = await self._deps.chat.complete_structured(
                [
                    ChatMessage(role="system", content=prompt.text),
                    ChatMessage(role="user", content=f"<situation>\n{situation}\n</situation>"),
                ],
                UnderstandingDraft,
            )
        except LLMError as err:
            await self._deps.tracer.emit(
                TraceEventType.ERROR,
                f"Understanding failed, using the sentence as symptom: {type(err).__name__}",
            )
            understanding = Understanding(
                symptom=situation[:MAX_SYMPTOM_CHARS],
                room_mention=None,
                urgency="normal",
                from_llm=False,
            )
        else:
            draft = result.value
            await self._deps.tracer.llm_call(
                result.usage, f"symptom '{draft.symptom}', room {draft.room_mention or '-'}"
            )
            understanding = Understanding(
                symptom=(draft.symptom.strip() or situation)[:MAX_SYMPTOM_CHARS],
                room_mention=draft.room_mention,
                urgency=draft.urgency,
            )
        span.summary = f"symptom '{understanding.symptom}', urgency {understanding.urgency}" + (
            f", mentions '{understanding.room_mention}'" if understanding.room_mention else ""
        )
        span.payload = {"prompt_version": prompt.version, **understanding.model_dump(mode="json")}
        return {"understanding": understanding}

    async def locate(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        rooms = self._deps.home.rooms()
        argument = state.get("room_argument")
        mention = state["understanding"].room_mention
        default = self._deps.settings.default_room
        room: str | None = None
        source = RoomSource.UNKNOWN
        if argument:
            room = match_room(argument, rooms)
            source = RoomSource.ARGUMENT
            if room is None:
                await self._deps.tracer.emit(
                    TraceEventType.ERROR, f"Room {argument!r} is not configured"
                )
        elif mention and (room := match_room(mention, rooms)) is not None:
            source = RoomSource.MENTIONED
        elif default and default in rooms:
            room, source = default, RoomSource.DEFAULT
        span.summary = f"room: {room or 'unknown'} ({source})"
        span.payload = {"rooms": sorted(rooms), "argument": argument, "mention": mention}
        return {"room": room, "room_source": source}

    async def observe(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        settings = self._deps.settings
        at = state.get("at")
        if at is not None and at.tzinfo is None:
            # Simulated wall-clock time in the home's timezone.
            at = at.replace(tzinfo=settings.tz)
        now = (at or self._deps.clock.now()).astimezone(settings.tz)
        weather = await self._read_weather()
        room = state.get("room")
        climate = await self._read_climate(room) if room else None
        observations = Observations(
            local_time=now, simulated_time=at is not None, weather=weather, climate=climate
        )
        parts = [f"{now:%a %H:%M}" + (" (simulated)" if at else "")]
        if weather:
            parts.append(
                f"outside {weather.outdoor_temp_c} C, sun {weather.sun_hours_forecast} h "
                f"in next {weather.forecast_hours} h"
            )
        else:
            parts.append("weather unknown")
        if climate:
            parts.append(
                f"{climate.room} {climate.actual_c} C (setpoint {climate.setpoint_c} C)"
                + (", window open" if climate.window_open else "")
            )
        elif room:
            parts.append(f"{room} climate unknown")
        span.summary = "; ".join(parts)
        span.payload = observations.model_dump(mode="json")
        return {"observations": observations}

    async def _read_weather(self) -> WeatherObservation | None:
        started = time.perf_counter()
        try:
            weather = await self._deps.weather.current()
        except WeatherError as err:
            await self._deps.tracer.emit(TraceEventType.ERROR, f"Weather unavailable: {err}")
            return None
        await self._deps.tracer.emit(
            TraceEventType.WEATHER_CALL,
            f"{weather.source}: {weather.outdoor_temp_c} C, clouds {weather.cloud_cover_pct} %, "
            f"sun {weather.sun_hours_forecast} h",
            payload=weather.model_dump(mode="json"),
            latency_ms=_elapsed_ms(started),
        )
        return weather

    async def _read_climate(self, room: str) -> RoomClimate | None:
        started = time.perf_counter()
        try:
            climate = await self._deps.home.read_climate(room)
        except HomeError as err:
            await self._deps.tracer.emit(TraceEventType.ERROR, f"Room climate unavailable: {err}")
            return None
        await self._deps.tracer.emit(
            TraceEventType.HOME_CALL,
            f"read {room}: {climate.actual_c} C, setpoint {climate.setpoint_c} C",
            payload={"home": self._deps.home.name, **climate.model_dump(mode="json")},
            latency_ms=_elapsed_ms(started),
        )
        return climate

    # --- retrieve / evaluate -------------------------------------------------

    async def retrieve(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        settings = self._deps.settings
        symptom = state["understanding"].symptom
        embedded = await self._deps.embedder.embed([symptom])
        if embedded.usage is not None:
            await self._deps.tracer.llm_call(
                embedded.usage,
                f"Embedded symptom '{symptom}'",
                type_=TraceEventType.EMBEDDING_CALL,
                payload={"cache_hits": embedded.cache_hits},
            )
        found = await self._deps.retrieval.similar_symptoms(
            embedded.vectors[0],
            top_k=settings.resolve_top_k,
            max_distance=settings.resolve_max_distance,
        )
        retrieved = [
            item.model_copy(update={"ref": f"S{index}"})
            for index, item in enumerate(found, start=1)
        ]
        span.summary = (
            f"{len(retrieved)} symptoms, {sum(len(s.actions) for s in retrieved)} actions"
            + (
                f"; closest '{retrieved[0].text}' ({retrieved[0].distance:.2f})"
                if retrieved
                else ""
            )
        )
        span.payload = {
            "top_k": settings.resolve_top_k,
            "max_distance": settings.resolve_max_distance,
            "matches": [
                {"ref": s.ref, "text": s.text, "distance": round(s.distance, 4)} for s in retrieved
            ],
        }
        return {"retrieved": retrieved}

    async def evaluate(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        observations = state["observations"]
        room = state.get("room")
        current = observations.climate.setpoint_c if observations.climate else None
        candidates: list[Candidate] = []
        for symptom in state["retrieved"]:
            for action in sorted(symptom.actions, key=lambda a: a.priority):
                checks = evaluate_conditions(action.conditions, observations)
                if action.kind is ActionKind.ADVICE:
                    decision = _advice_decision(checks)
                else:
                    decision = check_static(
                        action.kind,
                        action.params,
                        room=room,
                        current_c=current,
                        checks=checks,
                        bounds=self._bounds,
                    )
                candidates.append(
                    Candidate(
                        ref=f"A{len(candidates) + 1}",
                        symptom_ref=symptom.ref,
                        action=action,
                        checks=checks,
                        applicable=decision.allowed,
                        reason=decision.reason,
                        target_c=decision.target_c,
                    )
                )
        applicable = [c for c in candidates if c.applicable]
        span.summary = f"{len(applicable)}/{len(candidates)} candidate actions applicable" + (
            f": {', '.join(f'{c.ref} {c.action.title}' for c in applicable)}" if applicable else ""
        )
        span.payload = {
            "candidates": [
                {
                    "ref": c.ref,
                    "title": c.action.title,
                    "kind": c.action.kind,
                    "applicable": c.applicable,
                    "reason": c.reason,
                }
                for c in candidates
            ]
        }
        return {"candidates": candidates}

    # --- agent ----------------------------------------------------------------

    def _context_message(self, state: ResolveState) -> str:
        understanding = state["understanding"]
        observations = state["observations"]
        weather = observations.weather
        climate = observations.climate
        situation = {
            "symptom": understanding.symptom,
            "urgency": understanding.urgency,
            "room": state.get("room"),
            "room_source": state.get("room_source"),
        }
        observed = [
            {
                "ref": "O1",
                "local_time": f"{observations.local_time:%Y-%m-%d %H:%M %A}",
                "simulated": observations.simulated_time,
            },
            {"ref": "O2", "weather": weather.model_dump(mode="json") if weather else "unknown"},
            {
                "ref": "O3",
                "room_climate": climate.model_dump(mode="json") if climate else "unknown",
            },
        ]
        symptoms = [
            {"ref": s.ref, "text": s.text, "distance": round(s.distance, 3)}
            for s in state["retrieved"]
        ]
        candidates = [
            {
                "ref": c.ref,
                "symptom": c.symptom_ref,
                "kind": c.action.kind,
                "title": c.action.title,
                "instruction": c.action.instruction,
                "params": c.action.params.model_dump(exclude_none=True),
                "priority": c.action.priority,
                "checks": [_check_text(check) for check in c.checks],
                "applicable": c.applicable,
                "reason": c.reason,
            }
            for c in state["candidates"]
        ]
        lines = [
            "<situation>",
            json.dumps(situation),
            "</situation>",
            "<observations>",
            *(json.dumps(o) for o in observed),
            "</observations>",
            "<symptoms>",
            *(json.dumps(s) for s in symptoms),
            "</symptoms>",
            "<candidates>",
            *(json.dumps(c, default=str) for c in candidates),
            "</candidates>",
        ]
        return "\n".join(lines)

    async def agent(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        settings = self._deps.settings
        prompt = load_prompt(RunKind.RESOLVE, AGENT_PROMPT)
        tools_note = (
            "Database tool `query`: available."
            if self._deps.knowledge_query is not None
            else "Database tool `query`: UNAVAILABLE in this run. Do not query."
        )
        rooms = "<rooms>\n" + json.dumps(sorted(self._deps.home.rooms())) + "\n</rooms>"
        messages = [
            ChatMessage(role="system", content=prompt.text),
            ChatMessage(
                role="user", content=f"{tools_note}\n{self._context_message(state)}\n{rooms}"
            ),
        ]
        tool_calls: list[ToolCallRecord] = []
        thoughts: list[str] = []
        notes: str | None = None
        stopped = "max_steps"
        steps = 0
        for step in range(1, settings.agent_max_steps + 1):
            steps = step
            try:
                result = await self._deps.chat.complete_structured(messages, AgentStep)
            except LLMError as err:
                stopped = f"llm_error: {type(err).__name__}"
                await self._deps.tracer.emit(
                    TraceEventType.ERROR, f"Agent step {step} failed: {err}"
                )
                break
            decision = result.value
            await self._deps.tracer.llm_call(
                result.usage,
                f"Step {step}: {decision.action} - "
                f"{(decision.thought or decision.notes or '')[:MAX_STEP_SUMMARY_CHARS]}",
            )
            messages.append(ChatMessage(role="assistant", content=decision.model_dump_json()))
            thoughts.append(decision.thought)
            if decision.action == "finish":
                notes = decision.notes or decision.thought
                stopped = "finished"
                break
            observation = await self._run_tool(step, decision, tool_calls)
            messages.append(
                ChatMessage(role="user", content=f"<tool_result>\n{observation}\n</tool_result>")
            )
        findings = AgentFindings(
            notes=notes or " ".join(thoughts),
            steps=steps,
            tool_calls=tool_calls,
            stopped_reason=stopped,
        )
        span.summary = f"{steps} steps, {len(tool_calls)} tool calls, stopped: {stopped}"
        span.payload = {
            "prompt_version": prompt.version,
            "max_steps": settings.agent_max_steps,
            "max_tool_calls": settings.agent_max_tool_calls,
            "notes": findings.notes,
        }
        return {"agent": findings}

    async def _run_tool(
        self, step: int, decision: AgentStep, tool_calls: list[ToolCallRecord]
    ) -> str:
        """Execute one agent tool call within budget; every outcome becomes an observation."""
        if len(tool_calls) >= self._deps.settings.agent_max_tool_calls:
            return "Error: tool call budget exhausted. Finish now."
        if decision.action == "read_room":
            return await self._read_room_tool(step, decision.room, tool_calls)
        return await self._query_tool(step, decision.sql, tool_calls)

    async def _read_room_tool(
        self, step: int, room: str | None, tool_calls: list[ToolCallRecord]
    ) -> str:
        rooms = self._deps.home.rooms()
        target = match_room(room, rooms) if room else None
        if target is None:
            return f"Error: unknown room {room!r}; use one of {sorted(rooms)}."
        started = time.perf_counter()
        try:
            climate = await self._deps.home.read_climate(target)
        except HomeError as err:
            await self._deps.tracer.emit(TraceEventType.ERROR, f"read_room failed: {err}")
            return "Error: room read failed. Continue without it."
        latency = _elapsed_ms(started)
        tool_calls.append(
            ToolCallRecord(
                step=step,
                tool="read_room",
                argument=target,
                is_error=False,
                row_count=None,
                latency_ms=latency,
            )
        )
        await self._deps.tracer.emit(
            TraceEventType.HOME_CALL,
            f"read_room {target}: {climate.actual_c} C, setpoint {climate.setpoint_c} C",
            payload={"tool": "read_room", **climate.model_dump(mode="json")},
            latency_ms=latency,
        )
        return climate.model_dump_json()

    async def _query_tool(
        self, step: int, sql: str | None, tool_calls: list[ToolCallRecord]
    ) -> str:
        client = self._deps.knowledge_query
        if client is None:
            return "Error: the database tool is unavailable in this run."
        if not sql:
            return "Error: action=query requires `sql`."
        try:
            result = await client.execute_sql(sql)
        except UnsafeQueryError as err:
            await self._deps.tracer.emit(
                TraceEventType.MCP_CALL, f"rejected by SQL guard: {err}", payload={"sql": sql}
            )
            return f"Error: {err}"
        except McpUnavailableError as err:
            await self._deps.tracer.emit(TraceEventType.ERROR, f"MCP call failed: {err}")
            return "Error: database tool failed. Continue with what you know."
        tool_calls.append(
            ToolCallRecord(
                step=step,
                tool="query",
                argument=result.sql,
                is_error=result.is_error,
                row_count=result.row_count,
                latency_ms=result.latency_ms,
            )
        )
        outcome = "error" if result.is_error else f"{result.row_count} rows"
        await self._deps.tracer.emit(
            TraceEventType.MCP_CALL,
            f"{result.sql[:MAX_SUMMARY_ARG_CHARS]} -> {outcome}",
            payload={
                "tool": "execute_sql",
                "sql": result.sql,
                "is_error": result.is_error,
                "row_count": result.row_count,
                "truncated": result.truncated,
            },
            latency_ms=result.latency_ms,
        )
        return result.text[:MAX_OBSERVATION_CHARS]

    # --- decide / act / report -----------------------------------------------

    async def decide(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        prompt = load_prompt(RunKind.RESOLVE, DECIDE_PROMPT)
        agent = state["agent"]
        user = (
            f"<original_sentence>\n{state['situation']}\n</original_sentence>\n"
            f"{self._context_message(state)}\n<agent_notes>\n{agent.notes}\n</agent_notes>"
        )
        result = await self._deps.chat.complete_structured(
            [
                ChatMessage(role="system", content=prompt.text),
                ChatMessage(role="user", content=user),
            ],
            DecisionDraft,
        )
        await self._deps.tracer.llm_call(result.usage, "Drafted decision")
        resolution = resolve_decision(
            result.value,
            state,
            dry_run=state.get("dry_run", False) or self._deps.settings.dry_run,
            prompt_versions={
                UNDERSTAND_PROMPT: load_prompt(RunKind.RESOLVE, UNDERSTAND_PROMPT).version,
                AGENT_PROMPT: load_prompt(RunKind.RESOLVE, AGENT_PROMPT).version,
                DECIDE_PROMPT: prompt.version,
            },
        )
        span.summary = (
            f"{len(resolution.actions)} actions chosen "
            f"({', '.join(a.ref for a in resolution.actions) or '-'}), "
            f"{len(resolution.advice)} advice, {resolution.uncited_count} uncited"
        )
        span.payload = {"confidence": resolution.confidence, "reply": resolution.reply}
        return {"resolution": resolution}

    async def act(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        resolution = state["resolution"]
        candidates = {c.ref: c for c in state["candidates"]}
        room = state.get("room")
        observations = state["observations"]
        current = observations.climate.setpoint_c if observations.climate else None
        writes = 0
        acted: list[ChosenAction] = []
        for chosen in resolution.actions:
            candidate = candidates[chosen.ref]
            if candidate.action.kind is ActionKind.ADVICE:
                outcome = ActionOutcome.ADVISED if candidate.applicable else ActionOutcome.REJECTED
                acted.append(
                    chosen.model_copy(update={"outcome": outcome, "detail": candidate.reason})
                )
                continue
            decision = check_runtime(
                GuardDecision(candidate.applicable, candidate.reason, candidate.target_c),
                writes_this_run=writes,
                last_write_at=(await self._deps.commands.last_applied_at(room) if room else None),
                now=self._deps.clock.now(),
                bounds=self._bounds,
            )
            outcome, detail = await self._execute(candidate, decision, room, resolution.dry_run)
            if outcome is ActionOutcome.APPLIED:
                writes += 1
            await self._deps.commands.record(
                CommandRecord(
                    run_id=self._deps.tracer.run_id,
                    action_id=candidate.action.id,
                    room=room,
                    parameter=SET_POINT_TEMPERATURE,
                    old_value=current,
                    new_value=decision.target_c,
                    status=CommandStatus(outcome.value),
                    reason=detail,
                )
            )
            acted.append(chosen.model_copy(update={"outcome": outcome, "detail": detail}))
        updated = resolution.model_copy(update={"actions": acted})
        counts: dict[str, int] = {}
        for item in acted:
            if item.outcome is not None:
                counts[item.outcome] = counts.get(item.outcome, 0) + 1
        span.summary = ", ".join(f"{n} {o}" for o, n in counts.items()) or "nothing to do"
        span.payload = {
            "dry_run": resolution.dry_run,
            "actions": [{"ref": a.ref, "outcome": a.outcome, "detail": a.detail} for a in acted],
        }
        return {"resolution": updated}

    async def _execute(
        self,
        candidate: Candidate,
        decision: GuardDecision,
        room: str | None,
        dry_run: bool,
    ) -> tuple[ActionOutcome, str]:
        if not decision.allowed or room is None or decision.target_c is None:
            await self._deps.tracer.emit(
                TraceEventType.HOME_CALL,
                f"{candidate.ref} rejected by guard: {decision.reason}",
                payload={"ref": candidate.ref, "room": room, "reason": decision.reason},
            )
            return ActionOutcome.REJECTED, decision.reason
        if dry_run:
            await self._deps.tracer.emit(
                TraceEventType.HOME_CALL,
                f"dry run: would set {room} {decision.reason}",
                payload={"ref": candidate.ref, "room": room, "target_c": decision.target_c},
            )
            return ActionOutcome.DRY_RUN, f"dry run: {decision.reason}"
        started = time.perf_counter()
        try:
            await self._deps.home.set_setpoint(room, decision.target_c)
        except HomeError as err:
            await self._deps.tracer.emit(TraceEventType.ERROR, f"Setpoint write failed: {err}")
            return ActionOutcome.FAILED, str(err)
        await self._deps.tracer.emit(
            TraceEventType.HOME_CALL,
            f"set {room} {decision.reason}",
            payload={"ref": candidate.ref, "room": room, "target_c": decision.target_c},
            latency_ms=_elapsed_ms(started),
        )
        return ActionOutcome.APPLIED, decision.reason

    async def report(self, state: ResolveState, span: SpanResult) -> NodeUpdate:
        resolution_id = await self._deps.resolutions.save(
            self._deps.tracer.run_id, state["resolution"]
        )
        span.summary = f"resolution {resolution_id} stored"
        return {"resolution_id": resolution_id}


def _check_text(check: ConditionCheck) -> str:
    return f"{check.condition} {check.expected}: {check.observed or '?'} -> {check.outcome}"


def _advice_decision(checks: Sequence[ConditionCheck]) -> GuardDecision:
    """Advice never touches a device: unknown observations do not block it."""
    violated = [c.condition for c in checks if c.outcome is CheckOutcome.VIOLATED]
    if violated:
        return GuardDecision(False, f"conditions violated: {', '.join(violated)}")
    return GuardDecision(True, "advice; conditions met" if checks else "advice; unconditional")


def resolve_decision(
    draft: DecisionDraft,
    state: ResolveState,
    *,
    dry_run: bool,
    prompt_versions: dict[str, str],
) -> Resolution:
    """Map refs back to knowledge; unknown refs are dropped and listed, uncited claims flagged."""
    candidates = {c.ref: c for c in state["candidates"]}
    valid_refs = {*candidates, *(s.ref for s in state["retrieved"]), *OBSERVATION_REFS}
    invalid: list[str] = []
    chosen: list[ChosenAction] = []
    seen: set[str] = set()
    for item in draft.chosen:
        refs = _refs([item.ref])
        ref = refs[0] if refs else item.ref.strip()
        candidate = candidates.get(ref)
        if candidate is None or ref in seen:
            if candidate is None:
                invalid.append(item.ref)
            continue
        seen.add(ref)
        citations = [r for r in _refs([*item.citations, item.rationale]) if r in valid_refs]
        chosen.append(
            ChosenAction(
                ref=ref,
                action_id=candidate.action.id,
                kind=candidate.action.kind,
                title=candidate.action.title,
                rationale=item.rationale,
                citations=citations,
                uncited=not citations,
            )
        )
    advice: list[AdviceItem] = []
    for tip in draft.advice:
        refs = _refs([*tip.citations, tip.text])
        invalid += [r for r in refs if r not in valid_refs]
        citations = [r for r in refs if r in valid_refs]
        advice.append(AdviceItem(text=tip.text, citations=citations, uncited=not citations))
    return Resolution(
        symptom=state["understanding"].symptom,
        room=state.get("room"),
        reply=draft.reply,
        summary=draft.summary,
        actions=chosen,
        advice=advice,
        observations=state["observations"],
        confidence=draft.confidence,
        prompt_versions=prompt_versions,
        uncited_count=sum(a.uncited for a in chosen) + sum(a.uncited for a in advice),
        invalid_refs=list(dict.fromkeys(invalid)),
        dry_run=dry_run,
    )
