// Buddler trace UI. No framework, no build step.
// All server data is untrusted (it derives from user sentences and LLM output):
// it is only ever rendered via textContent, never innerHTML.
"use strict";

const KEY_STORAGE = "buddler-api-key";
const LIST_REFRESH_MS = 5000;
const RECONNECT_MS = 1500;
const ICONS = {
  node_start: "▶", node_end: "✔", llm_call: "🤖", embedding_call: "🔢",
  mcp_call: "🗄", tool_call: "🛠", home_call: "🏠", weather_call: "🌦", error: "✖",
};
const NODE_STATES = ["st-pending", "st-running", "st-paused", "st-done", "st-warn", "st-error"];
const PAUSED_PREFIX = "paused";

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.hidden = true; }, 4000);
}

// --- API access ---------------------------------------------------------------

function storedKey() {
  try { return sessionStorage.getItem(KEY_STORAGE) || ""; } catch { return ""; }
}

function askForKey() {
  return new Promise((resolve) => {
    const dialog = $("key-dialog");
    const input = $("key-input");
    input.value = "";
    dialog.addEventListener("close", () => {
      const key = input.value.trim();
      try { sessionStorage.setItem(KEY_STORAGE, key); } catch { /* private mode: key lives for this page only */ }
      askForKey.fallback = key;
      resolve(key);
    }, { once: true });
    dialog.showModal();
    input.focus();
  });
}

function currentKey() {
  return storedKey() || askForKey.fallback || "";
}

async function api(path, options = {}, retried = false) {
  const headers = { ...(options.headers || {}), Authorization: `Bearer ${currentKey()}` };
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401 && !retried) {
    await askForKey();
    return api(path, options, true);
  }
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} for ${path}`);
  return response;
}

const getJson = async (path) => (await api(path)).json();

// --- formatting ---------------------------------------------------------------

function formatDuration(ms) {
  if (ms == null || Number.isNaN(ms)) return "–";
  if (ms < 1000) return `${ms} ms`;
  const seconds = ms / 1000;
  return seconds < 60 ? `${seconds.toFixed(1)} s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function runDurationMs(run) {
  const end = run.finished_at ? new Date(run.finished_at) : new Date();
  return end - new Date(run.started_at);
}

function badge(status) {
  return el("span", { class: `badge ${status}` }, status.replace("_", " "));
}

// --- run list -----------------------------------------------------------------

async function showRunList() {
  $("view-list").hidden = false;
  const refresh = async () => {
    try {
      const runs = await getJson("/runs?limit=50");
      const body = $("runs-body");
      body.replaceChildren(...runs.map((run) => el("tr", {},
        el("td", {}, new Date(run.started_at).toLocaleString()),
        el("td", {}, run.kind),
        el("td", { class: "subject" }, run.subject),
        el("td", {}, badge(run.status)),
        el("td", {}, formatDuration(runDurationMs(run))),
        el("td", { class: "mono" }, `${run.tokens_in}/${run.tokens_out}`),
        el("td", {}, el("a", { href: `/ui/runs/${run.id}`, class: "mono" }, run.id.slice(0, 8))),
      )));
      $("runs-empty").hidden = runs.length > 0;
      $("list-updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
    } catch (error) {
      toast(error.message);
    }
  };
  await refresh();
  setInterval(refresh, LIST_REFRESH_MS);
}

// --- run view -----------------------------------------------------------------

class RunView {
  constructor(runId) {
    this.runId = runId;
    this.run = null;
    this.lastSeq = 0;
    this.firstTs = null;
    this.ended = false;
    this.nodeState = new Map();   // name -> {status, errors}
    this.graphNodes = new Map();  // name -> SVG <g>
    this.events = new Map();      // seq -> event
    this.filter = null;
    this.counters = { llm: 0, mcp: 0, home: 0, tokens: 0 };
  }

  async start() {
    $("view-run").hidden = false;
    $("run-id").textContent = this.runId;
    const detail = await getJson(`/runs/${this.runId}`);
    this.applyRun(detail);
    await this.renderGraph(detail.run.kind);
    this.durationTimer = setInterval(() => this.renderDuration(), 1000);
    this.stream();
  }

  applyRun(detail) {
    this.run = detail.run;
    $("run-kind").textContent = `${detail.run.kind} workflow`;
    $("run-title").textContent = detail.run.subject;
    $("stat-status").replaceWith(Object.assign(badge(detail.run.status), { id: "stat-status" }));
    this.renderDuration();
    const errorBox = $("run-error");
    errorBox.hidden = !detail.run.error;
    errorBox.textContent = detail.run.error || "";
    if (detail.resolution) this.renderResolution(detail.resolution);
  }

  renderDuration() {
    if (!this.run) return;
    $("stat-duration").textContent = formatDuration(runDurationMs(this.run));
    if (this.run.finished_at) clearInterval(this.durationTimer);
  }

  async renderGraph(kind) {
    const graph = await getJson(`/graphs/${kind}`);
    const source = graph.mermaid;
    this.steps = new Map(graph.steps.map((step) => [step.node, step]));
    this.renderExplanation(graph);
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "default" });
    // LangGraph's classDefs use !important fills that would mask the live state colors.
    const cleaned = source.split("\n").filter((line) => !line.trim().startsWith("classDef")).join("\n");
    const { svg } = await window.mermaid.render(`wf-${kind}`, cleaned);
    const container = $("graph");
    // Mermaid output is generated from our own graph definition, not from run data.
    container.innerHTML = svg;
    for (const group of container.querySelectorAll("g.node")) {
      const name = group.dataset.id || (group.id.match(/flowchart-(.+?)-\d+$/) || [])[1];
      if (!name || name.startsWith("__")) continue;
      this.graphNodes.set(name, group);
      group.classList.add("st-pending");
      const step = this.steps.get(name);
      if (step) {
        const tooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
        tooltip.textContent = `${step.title}: ${step.summary}`;
        group.prepend(tooltip);
      }
      group.addEventListener("click", () => {
        this.toggleFilter(name);
        if (step) this.showStep(step);
      });
    }
  }

  renderExplanation(graph) {
    if (!graph.steps.length) return;
    $("explain").hidden = false;
    $("explain-overview").textContent = graph.overview || "";
    $("explain-glossary").replaceChildren(...graph.glossary.flatMap((entry) => [
      el("dt", {}, entry.term), el("dd", {}, entry.definition),
    ]));
    $("explain-steps").replaceChildren(...graph.steps.map((step) => el("li", {},
      el("details", {},
        el("summary", {}, el("span", { class: "node mono" }, step.node), el("b", {}, step.title), ` – ${step.summary}`),
        el("p", {}, step.description),
        el("p", { class: "step-example" }, step.example),
      ),
    )));
  }

  showStep(step) {
    this.shownStep = step;
    for (const selected of document.querySelectorAll(".timeline li.selected")) selected.classList.remove("selected");
    const section = (heading, ...content) => [el("h3", {}, heading), ...content];
    $("detail").replaceChildren(
      el("div", { class: "eyebrow" }, `step · ${step.node}`),
      el("h3", { class: "step-title" }, step.title),
      el("p", { class: "step-lead" }, step.description),
      ...section("Why it matters", el("p", {}, step.why)),
      ...section("Example", el("p", { class: "step-example" }, step.example)),
      ...section("In this run", el("p", {}, this.stepOutcome(step.node))),
      el("details", { class: "step-tech" },
        el("summary", {}, "Technical details"),
        el("ul", { class: "step-details" }, ...step.details.map((line) => el("li", {}, line))),
      ),
    );
    $("detail").classList.remove("muted");
  }

  // The step's own trace summary: what it actually did in this run.
  stepOutcome(node) {
    const ends = [...this.events.values()].filter((e) => e.node === node && e.type === "node_end");
    if (ends.length) return ends[ends.length - 1].summary;
    return this.nodeState.get(node)?.status === "running" ? "Running…" : "Not run yet.";
  }

  setNodeState(name, status) {
    const current = this.nodeState.get(name) || { status: "pending", errors: 0 };
    current.status = status;
    this.nodeState.set(name, current);
    const group = this.graphNodes.get(name);
    if (!group) return;
    group.classList.remove(...NODE_STATES);
    group.classList.add(`st-${status}`);
  }

  onEvent(event) {
    if (this.events.has(event.seq)) return;
    this.events.set(event.seq, event);
    this.lastSeq = Math.max(this.lastSeq, event.seq);
    this.firstTs ??= new Date(event.ts);
    const state = this.nodeState.get(event.node) || { status: "pending", errors: 0 };
    switch (event.type) {
      case "node_start":
        this.setNodeState(event.node, "running");
        break;
      case "node_end":
        this.setNodeState(
          event.node,
          event.summary.startsWith(PAUSED_PREFIX) ? "paused" : state.errors ? "warn" : "done",
        );
        break;
      case "error":
        state.errors += 1;
        this.nodeState.set(event.node, state);
        this.setNodeState(event.node, "error");
        break;
      case "llm_call":
        this.counters.llm += 1;
        break;
      case "mcp_call":
        this.counters.mcp += 1;
        break;
      case "home_call":
        this.counters.home += 1;
        break;
      default:
        break;
    }
    this.counters.tokens += event.tokens || 0;
    $("stat-llm").textContent = this.counters.llm;
    $("stat-mcp").textContent = this.counters.mcp;
    $("stat-home").textContent = this.counters.home;
    $("stat-tokens").textContent = this.counters.tokens.toLocaleString();
    this.appendTimeline(event);
    if (this.shownStep?.node === event.node && ["node_start", "node_end"].includes(event.type)) this.showStep(this.shownStep);
  }

  appendTimeline(event) {
    const list = $("timeline");
    const stick = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
    const paused = event.type === "node_end" && event.summary.startsWith(PAUSED_PREFIX);
    const metrics = [
      event.tokens != null ? `${event.tokens} tok` : null,
      event.latency_ms != null ? formatDuration(event.latency_ms) : null,
      `+${((new Date(event.ts) - this.firstTs) / 1000).toFixed(2)}s`,
    ].filter(Boolean).join(" · ");
    const child = !["node_start", "node_end"].includes(event.type);
    const item = el("li", { class: `t-${event.type}${child ? " child" : ""}`, dataset: { seq: event.seq, node: event.node } },
      el("span", { class: "seq" }, event.seq),
      el("span", {}, paused ? "⏸" : ICONS[event.type] || "•"),
      el("span", {}, el("span", { class: "node" }, event.node), el("span", { class: "summary" }, event.summary)),
      el("span", { class: "metrics" }, metrics),
    );
    item.hidden = this.filter !== null && this.filter !== event.node;
    item.addEventListener("click", () => this.showDetail(event, item));
    list.append(item);
    if (stick) list.scrollTop = list.scrollHeight;
  }

  showDetail(event, item) {
    this.shownStep = null;
    for (const selected of document.querySelectorAll(".timeline li.selected")) selected.classList.remove("selected");
    item.classList.add("selected");
    const rows = [
      ["seq", event.seq], ["type", event.type], ["node", event.node],
      ["parent", event.parent_node || "–"], ["time", new Date(event.ts).toLocaleTimeString()],
      ["tokens", event.tokens ?? "–"], ["latency", event.latency_ms != null ? formatDuration(event.latency_ms) : "–"],
    ];
    const list = el("dl", {});
    for (const [key, value] of rows) list.append(el("dt", {}, key), el("dd", {}, value));
    $("detail").replaceChildren(
      el("p", {}, event.summary),
      list,
      el("pre", { class: "json" }, JSON.stringify(event.payload, null, 2)),
    );
    $("detail").classList.remove("muted");
  }

  toggleFilter(name) {
    this.filter = this.filter === name ? null : name;
    for (const [node, group] of this.graphNodes) group.classList.toggle("selected", node === this.filter);
    for (const item of $("timeline").children) item.hidden = this.filter !== null && item.dataset.node !== this.filter;
    $("filter-label").textContent = this.filter ? `filtered: ${this.filter} (click node again to clear)` : "";
  }

  renderResolution(res) {
    $("resolution").hidden = false;
    $("res-reply").textContent = res.reply;
    $("res-meta").textContent = `confidence ${res.confidence.toFixed(2)} · ${res.uncited_count} uncited${res.dry_run ? " · dry run" : ""}`;
    $("res-summary").textContent = `${res.summary} (symptom "${res.symptom}", room ${res.room || "unknown"})`;
    const obs = res.observations;
    const facts = [["O1 time", `${new Date(obs.local_time).toLocaleString()}${obs.simulated_time ? " (simulated)" : ""}`]];
    const w = obs.weather;
    facts.push(["O2 weather", w ? `${w.outdoor_temp_c} °C, clouds ${w.cloud_cover_pct} %, sun ${w.sun_hours_forecast} h in ${w.forecast_hours} h (${w.source})` : "unknown"]);
    const c = obs.climate;
    facts.push(["O3 room", c ? `${c.room}: ${c.actual_c} °C, setpoint ${c.setpoint_c} °C, humidity ${c.humidity_pct ?? "–"} %, window ${c.window_open == null ? "–" : c.window_open ? "open" : "closed"}` : "unknown"]);
    $("res-observations").replaceChildren(...facts.flatMap(([key, value]) => [el("dt", {}, key), el("dd", {}, value)]));
    const refs = (item) => el("span", { class: "refs" }, item.citations.length ? item.citations.join(", ") : "no citations");
    const uncited = (item) => (item.uncited ? el("span", { class: "uncited" }, "uncited") : null);
    $("res-actions-wrap").hidden = res.actions.length === 0;
    $("res-actions").replaceChildren(...res.actions.map((a) => el("li", {},
      el("div", {}, el("span", { class: `sev ${a.outcome || "pending"}` }, a.outcome || "chosen"), el("span", { class: "title" }, `${a.ref} ${a.title}`), uncited(a)),
      el("div", {}, a.rationale),
      a.detail ? el("div", { class: "muted small" }, a.detail) : null,
      refs(a),
    )));
    $("res-advice-wrap").hidden = res.advice.length === 0;
    $("res-advice").replaceChildren(...res.advice.map((a) => el("li", {},
      el("div", {}, el("span", { class: "title" }, a.text), uncited(a)),
      refs(a),
    )));
  }

  async onEnd(run) {
    this.ended = true;
    $("live").hidden = true;
    if (run.status === "failed") {
      for (const [name, state] of this.nodeState) if (state.status === "running") this.setNodeState(name, "error");
    }
    this.applyRun(await getJson(`/runs/${this.runId}`));
  }

  // EventSource cannot send Authorization headers; read the SSE stream via fetch instead.
  async stream() {
    while (!this.ended) {
      try {
        $("live").hidden = false;
        const response = await api(`/runs/${this.runId}/stream`, { headers: { "Last-Event-ID": String(this.lastSeq) } });
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += value;
          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) >= 0) {
            await this.onMessage(buffer.slice(0, boundary));
            buffer = buffer.slice(boundary + 2);
          }
        }
      } catch (error) {
        toast(`stream: ${error.message}`);
      }
      if (!this.ended) await new Promise((resolve) => setTimeout(resolve, RECONNECT_MS));
    }
  }

  async onMessage(block) {
    let type = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // heartbeat comment
      const [field, ...rest] = line.split(": ");
      const value = rest.join(": ");
      if (field === "event") type = value;
      else if (field === "data") data.push(value);
    }
    if (!data.length) return;
    const payload = JSON.parse(data.join("\n"));
    if (type === "trace") this.onEvent(payload);
    else if (type === "end") await this.onEnd(payload);
  }
}

// --- routing ------------------------------------------------------------------

async function main() {
  $("change-key").addEventListener("click", () => askForKey().then(() => location.reload()));
  const match = location.pathname.match(/^\/ui\/runs\/([0-9a-f-]{36})$/);
  try {
    if (match) await new RunView(match[1]).start();
    else await showRunList();
  } catch (error) {
    toast(error.message);
  }
}

document.addEventListener("DOMContentLoaded", main);
