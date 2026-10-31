/* SYNAPSE dashboard. All data comes from this server; no external assets. */
"use strict";

const state = { workflows: [], selected: null, compiled: null, benchMode: "mock",
                bench: null, validation: null, pair: null };
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const ms = value => value == null ? "—" : `${Number(value).toFixed(1)} ms`;
const num = (value, digits = 1) => value == null ? "—" : Number(value).toFixed(digits);
const cost = (value, unit) => value == null ? "—" :
  unit?.startsWith("USD") ? `$${Number(value).toFixed(8)}` : `${num(value, 3)} mock units`;
const tokens = value => value == null ? "—" : num(value, Number.isInteger(value) ? 0 : 1);
function change(before, after, format) {
  if (before == null || after == null) return "—";
  const delta = before - after;
  if (Math.abs(delta) < 1e-12) return "No change";
  const share = before ? ` (${num(Math.abs(delta) / before * 100, 1)}%)` : "";
  return `${format(Math.abs(delta))}${share} ${delta < 0 ? "more" : "saved"}`;
}
const recordUsage = record => record?.usage_totals || {
  tokens_in: (record?.nodes || []).reduce((sum, node) => sum + node.usage.tokens_in, 0),
  tokens_out: (record?.nodes || []).reduce((sum, node) => sum + node.usage.tokens_out, 0),
  tokens_total: (record?.nodes || []).reduce((sum, node) =>
    sum + node.usage.tokens_in + node.usage.tokens_out, 0),
  cost: (record?.nodes || []).reduce((sum, node) => sum + node.usage.cost, 0)
};

async function json(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { "Content-Type": "application/json",
    ...options.headers } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `${response.status} ${response.statusText}`);
  return data;
}
const post = (url, body) => json(url, { method: "POST", body: JSON.stringify(body) });
function error(where, exception) {
  const message = String(exception.message || exception);
  const friendly = /minute model budget|local minute budget/i.test(message) ?
    "This model's local minute budget is full. Wait a minute or select Mock." :
    /daily model budget|daily model quota/i.test(message) ?
    "The provider reported an exhausted quota. You can run this model again to recheck, or choose another model." :
    /quota|RESOURCE_EXHAUSTED|rate.?limit|\b429\b/i.test(message) ?
    "Model quota or rate limit reached. Select another provider or Mock, then try again." :
    message.length > 280 ? `${message.slice(0, 277)}…` : message;
  $(where).innerHTML = `<p class="notice error">${esc(friendly)}</p>`;
}
const savingClass = (before, after) => before > after ? "good" : before < after ? "bad" : "";
const deltaLabel = (before, after, name) => `${name} ${after > before ? "increase" : "reduction"}`;
const timeRatio = (before, after) => before && after ? (before >= after ?
  `${num(before / after, 2)}× faster` : `${num(after / before, 2)}× slower`) : "—";
function metric(label, value, detail = "", klass = "") {
  return `<div class="metric"><div class="label">${esc(label)}</div>` +
    `<div class="value ${klass}">${esc(value)}</div><div class="detail">${esc(detail)}</div></div>`;
}
function selectCases() {
  const workflow = state.workflows.find(w => w.file === state.selected);
  const cases = workflow?.cases || [];
  for (const id of ["timelineCase", "liveCase"]) {
    $(id).innerHTML = cases.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  }
  for (const id of ["timelineScale", "liveScale"]) {
    $(id).innerHTML = [["0.001", "Fast"], ["0.01", "Demo"],
      ["0.1", "Visible"], ["1", "Full"]].map(([v, label]) =>
      `<option value="${v}" ${v === "0.01" ? "selected" : ""}>${label} (×${v})</option>`).join("");
  }
}

function graphSvg(nodes, groups, original = false) {
  const perRow = original ? 1 : Math.max(1, ...groups.map(g => g.length));
  const cellW = 205, cellH = 88, left = 24, top = 34;
  const height = top + groups.length * cellH + 26;
  const width = left * 2 + perRow * cellW;
  const positions = {};
  groups.forEach((group, row) => group.forEach((id, col) => {
    positions[id] = { x: left + col * cellW, y: top + row * cellH };
  }));
  const arrows = nodes.flatMap(node => (node.deps || []).map(dep => {
    const a = positions[dep], b = positions[node.id];
    if (!a || !b) return "";
    return `<path class="edge" marker-end="url(#arrow)" d="M ${a.x + 92} ${a.y + 53} ` +
      `C ${a.x + 92} ${b.y - 9}, ${b.x + 92} ${a.y + 61}, ${b.x + 92} ${b.y - 4}"/>`;
  })).join("");
  const boxes = groups.map((group, row) => {
    if (original || group.length < 2) return "";
    return `<g class="group"><rect x="${left - 10}" y="${top + row * cellH - 16}" ` +
      `width="${perRow * cellW - 5}" height="${cellH - 4}" rx="10"/>` +
      `<text x="${left}" y="${top + row * cellH - 1}">PARALLEL</text></g>`;
  }).join("");
  const cards = nodes.map(node => {
    const p = positions[node.id];
    if (!p) return "";
    const removed = node.removed ? (node.removed.pass === "cse" ?
      `merged → ${node.removed.into}` : "removed") : "";
    const tags = [removed, node.routed ? `${node.routed.from} → ${node.routed.to}` : "",
      node.cached ? "cached" : ""].filter(Boolean).join(" · ");
    const kind = `${node.kind || ""} · ${node.effect || ""}` +
      (node.model ? ` · ${node.model}` : "");
    return `<g class="node ${esc(node.effect)} ${node.removed ? "removed" : ""} ` +
      `${node.routed ? "routed" : ""}"><rect class="box" x="${p.x}" y="${p.y}" ` +
      `width="184" height="58" rx="8"/><text class="id" x="${p.x + 10}" ` +
      `y="${p.y + 21}">${esc(node.id)}</text><text class="sub" x="${p.x + 10}" ` +
      `y="${p.y + 39}">${esc(kind)}</text>` +
      (tags ? `<text class="tag" x="${p.x + 10}" y="${p.y + 55}">${esc(tags)}</text>` : "") +
      `</g>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" ` +
    `aria-label="${original ? "Original" : "Compiled"} workflow graph">` +
    `<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" ` +
    `orient="auto"><path class="arrowhead" d="M0 0L6 3L0 6Z"/></marker></defs>` +
    `${boxes}${arrows}${cards}</svg>`;
}

function renderGraph() {
  const data = state.compiled;
  if (!data) return;
  const before = data.graph.source.nodes, after = data.graph.compiled.nodes;
  const originalGroups = before.map(n => [n.id]);
  $("graphOriginal").innerHTML = graphSvg(before, originalGroups, true);
  $("graphCompiled").innerHTML = graphSvg(after, data.graph.levels);
  const parallel = data.graph.levels.filter(g => g.length > 1).length;
  $("graphSummary").innerHTML = [
    `${before.length} → ${after.length} steps`, `${parallel} parallel groups`,
    `${after.filter(n => n.cached).length} cacheable`].map(x =>
      `<span class="chip">${esc(x)}</span>`).join("");
}

function renderDecisions() {
  const rows = state.compiled?.passes || [];
  $("decisionRows").innerHTML = rows.map(row => {
    const verdict = row.verdict?.verdict || "—";
    const measured = row.measured_ms == null ? "—" :
      `${ms(row.measured_ms)}${row.measured_basis === "warm" ? " (warm)" : ""}`;
    return `<tr class="${esc(row.status)}"><td data-label="Pass">${esc(row.label || row.pass_name)}</td>` +
      `<td data-label="Compiler"><span class="pill ${esc(row.status)}">${esc(row.status)}</span></td>` +
      `<td data-label="Validation"><span class="pill ${esc(verdict)}">${esc(verdict)}</span></td>` +
      `<td data-label="Steps">${esc((row.nodes || []).join(", ") || "—")}</td>` +
      `<td data-label="Reason">${esc(row.reason || "—")}</td>` +
      `<td data-label="Predicted gain" class="num">${esc(ms(row.predicted_ms))}</td>` +
      `<td data-label="Measured gain" class="num">${esc(measured)}</td>` +
      `<td data-label="Cost change" class="num">` +
      `${esc(row.measured_cost == null ? "—" : num(row.measured_cost, 3))}</td></tr>`;
  }).join("") || `<tr><td colspan="8">No compiler decisions yet.</td></tr>`;
  $("decisionNote").textContent = state.compiled?.profile ?
    `Predictions use ${state.compiled.profile.runs} stored mock baseline run(s). ` +
    `Measured gains come from the latest benchmark, when available.` :
    "Run a profile and benchmark to populate predicted and measured gains.";
}

function gantt(records) {
  if (!records?.baseline && !records?.compiled) return `<p class="muted">Run a workflow first.</p>`;
  const max = Math.max(records.baseline?.wall_ms || 0, records.compiled?.wall_ms || 0, 1);
  const kinds = new Map((state.compiled?.source?.nodes || []).map(n => [n.id, n.kind]));
  const sections = ["baseline", "compiled"].filter(mode => records[mode]).map(mode => {
    const record = records[mode];
    const rows = (record.nodes || []).map(node => {
      const left = 100 * node.start_s * 1000 / max;
      const width = Math.max(.4, 100 * (node.end_s - node.start_s) * 1000 / max);
      const label = `${node.node_id}: ${ms((node.end_s - node.start_s) * 1000)}` +
        (node.cache_hit ? " · cache hit" : "") + (node.error ? ` · ${node.error}` : "");
      return `<div class="row"><span class="name" title="${esc(node.node_id)}">` +
        `${esc(node.node_id)}</span><div class="track"><div class="bar ` +
        `k-${esc(kinds.get(node.node_id) || "tool")} ${node.cache_hit ? "hit" : ""} ` +
        `${node.error ? "err" : ""}" style="left:${left}%;width:${width}%" ` +
        `title="${esc(label)}"></div></div></div>`;
    }).join("");
    return `<div class="section ${mode}"><div class="section-head"><strong>` +
      `${mode === "baseline" ? "Original" : "Compiled"}</strong><span class="time">` +
      `${esc(ms(record.wall_ms))}</span></div>${rows}</div>`;
  }).join("");
  const ticks = [0, .25, .5, .75, 1].map(part =>
    `<span style="left:${part * 100}%">${esc(ms(max * part))}</span>`).join("");
  return sections + `<div class="axis">${ticks}</div>`;
}

function watch(runId, mode, onEvent) {
  return new Promise((resolve, reject) => {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${scheme}//${location.host}/ws/runs/${encodeURIComponent(runId)}`);
    let finished = false;
    socket.onmessage = message => {
      const event = JSON.parse(message.data);
      onEvent(mode, event);
      if (event.type === "run_end") { finished = true; socket.close(); resolve(event); }
      if (event.type === "error") { finished = true; socket.close(); reject(new Error(event.message)); }
    };
    socket.onerror = () => { finished = true; reject(new Error(`Event stream failed for ${mode}`)); };
    socket.onclose = () => { if (!finished) reject(new Error(`Event stream closed during ${mode}`)); };
  });
}

async function runOne(caseId, scale, choice, mode, onEvent = () => {}) {
  const { provider, model } = choice;
  onEvent(mode, { type: "run_start" });
  const started = await post("/runs", { workflow: state.selected, mode, case_id: caseId,
    scale, live: provider !== "mock", provider, model,
    cache_mode: mode === "compiled" && choice.warm ? "warm" : "cold" });
  await watch(started.run_id, mode, onEvent);
  const record = await json(`/runs/${encodeURIComponent(started.run_id)}`);
  if (record.status !== "done" || record.nodes?.some(n => n.error)) {
    const failure = new Error(`${record.mode} run failed: ${record.error || record.nodes?.find(n => n.error)?.error}`);
    failure.records = { [mode]: record };
    throw failure;
  }
  return record;
}

async function runPair(caseId, scale, choice, onEvent = () => {}) {
  const one = mode => runOne(caseId, scale, choice, mode, onEvent);
  const records = {};
  const collect = async mode => { records[mode] = await one(mode); };
  try {
    if (choice.provider === "mock") {
      const results = await Promise.allSettled([collect("baseline"), collect("compiled")]);
      const failed = results.find(r => r.status === "rejected");
      if (failed) throw failed.reason;
    } else {
      await collect("baseline");
      await collect("compiled");
    }
    return records;
  } catch (exc) {
    exc.records = { ...records, ...exc.records };
    throw exc;
  }
}

async function timelineRun() {
  $("timelineRun").disabled = true;
  $("timelineGantt").innerHTML = `<p class="muted">Running both versions…</p>`;
  try {
    state.pair = await runPair($("timelineCase").value, Number($("timelineScale").value),
      { provider: "mock", model: null });
    $("timelineGantt").innerHTML = gantt(state.pair);
    const a = state.pair.baseline, b = state.pair.compiled;
    $("timelineLegend").innerHTML = `<span>Decision: ${esc(a.key_value)} → ` +
      `${esc(b.key_value)}</span><span>Speedup: ${esc(num(a.wall_ms / b.wall_ms, 2))}×</span>`;
  } catch (exc) { error("timelineGantt", exc); }
  finally { $("timelineRun").disabled = false; }
}

function renderValidation(data) {
  $("validationStatus").textContent = `${data.passed}/${data.checks.length} checks passed ` +
    `across ${data.cases.length} cases in ${num(data.duration_s, 1)} s.`;
  const verdicts = Object.entries(data.verdicts || {}).map(([name, value]) =>
    `<div class="verdict ${esc(value.verdict)}"><strong>${esc(name)}: ` +
    `${esc(value.verdict)}</strong><div class="why">${esc(value.reason)}</div></div>`).join("");
  const cases = data.cases.map(row => `<div class="case ${row.passed ? "" : "fail"}">` +
    `<strong>${esc(row.case_id)}</strong><small>${row.passed ? "pass" : "fail"} · ` +
    `${esc(row.checks)} checks</small></div>`).join("");
  $("validationContent").innerHTML = `<div class="card"><h3>Pass verdicts</h3>` +
    `<div class="verdicts">${verdicts}</div></div><div class="card"><h3>Cases</h3>` +
    `<div class="case-grid">${cases}</div></div>`;
}

async function validateRun() {
  $("validateRun").disabled = true;
  $("validationStatus").textContent = "Validating every case and pass…";
  try {
    state.validation = await post("/validate", { workflow: state.selected,
      unsafe_route_risk: $("unsafe").checked, scale: .001 });
    renderValidation(state.validation);
    state.compiled = await post("/compile", { workflow: state.selected });
    renderGraph(); renderDecisions();
  } catch (exc) { error("validationContent", exc); }
  finally { $("validateRun").disabled = false; }
}

function renderBench() {
  const all = state.bench;
  const result = Array.isArray(all) ? all.find(x => x.workflow ===
    state.workflows.find(w => w.file === state.selected)?.name) : null;
  if (!result) {
    $("benchContent").innerHTML = `<p class="notice">No ${esc(state.benchMode)} benchmark ` +
      `for this workflow yet. Run <code>make bench</code> to generate one.</p>`;
    return;
  }
  const cold = result.configurations.filter(x => x.cache === "cold");
  const warm = result.configurations.filter(x => x.cache === "warm");
  const first = cold[0], last = cold[cold.length - 1];
  const unit = result.cost_unit || "mock units";
  const getCost = row => row.mean_cost ?? row.mean_cost_mock_units;
  const getTokens = row => row.mean_tokens_in + row.mean_tokens_out;
  const speedup = first.p50_ms / last.p50_ms;
  const headline = [
    metric("Cold median", ms(last.p50_ms), `${ms(first.p50_ms)} original · ${timeRatio(first.p50_ms, last.p50_ms)}`, savingClass(first.p50_ms, last.p50_ms)),
    metric("Cold p95", ms(last.p95_ms), `${ms(first.p95_ms)} original`),
    metric("Original tokens / case", tokens(getTokens(first)),
      `${tokens(first.mean_tokens_in)} in · ${tokens(first.mean_tokens_out)} out`),
    metric("Optimized tokens / case", tokens(getTokens(last)),
      `${tokens(last.mean_tokens_in)} in · ${tokens(last.mean_tokens_out)} out`),
    metric("Token reduction", change(getTokens(first), getTokens(last), tokens)),
    metric("Original cost / case", cost(getCost(first), unit)),
    metric("Optimized cost / case", cost(getCost(last), unit)),
    metric("Cost reduction", change(getCost(first), getCost(last), value => cost(value, unit))),
    metric("Correctness", `${last.correct}/${last.runs}`, `Across ${result.cases} cases × ${result.repeats} repeats`, "good")
  ].join("");
  const table = cold.map((row, i) => `<tr><td>${esc(row.label || row.name)}</td>` +
    `<td class="num">${esc(ms(row.p50_ms))}</td><td class="num">${esc(ms(row.p95_ms))}</td>` +
    `<td class="num">${esc(num(row.mean_llm_calls, 2))}</td>` +
    `<td class="num">${esc(num(row.mean_tokens_in, 0))} / ${esc(num(row.mean_tokens_out, 0))}</td>` +
    `<td class="num">${esc(cost(getCost(row), unit))}</td>` +
    `<td class="num">${esc(row.correct)}/${esc(row.runs)}</td>` +
    `<td class="num">${i ? esc(ms(cold[i - 1].p50_ms - row.p50_ms)) : "—"}</td></tr>`).join("");
  const warmText = warm.length > 1 ? `<p class="note">Repeat customer: ` +
    `${esc(ms(warm[0].p50_ms))} → ${esc(ms(warm[warm.length - 1].p50_ms))} ` +
    `(${esc(num(warm[0].p50_ms / warm[warm.length - 1].p50_ms, 2))}×).</p>` : "";
  $("benchContent").innerHTML = `<div class="metrics">${headline}</div><div class="card scroll">` +
    `<h3>Ablation bridge · cold cache</h3><table><thead><tr><th>Configuration</th>` +
    `<th class="num">Median</th><th class="num">p95</th><th class="num">LLM calls</th>` +
    `<th class="num">Tokens in / out</th><th class="num">Cost</th>` +
    `<th class="num">Correct</th><th class="num">Gain vs prior</th></tr></thead>` +
    `<tbody>${table}</tbody></table>${warmText}</div><p class="note">` +
    `${result.sample ? "Bundled synthetic example" : "Measured on this server"} ` +
    `at mock delay scale ×${esc(result.latency_scale)}; generated ${esc(result.created_at)}. ` +
    `${state.benchMode === "mock" ? "Mock cost units are not dollars." :
      "USD costs estimate published paid-tier prices; free-tier usage may incur no charge."}</p>`;
}

async function loadBench() {
  try { state.bench = await json(`/bench/latest?mode=${state.benchMode}`); }
  catch (_) { state.bench = []; }
  renderBench();
}

function renderTimer(mode, start, events, record) {
  const element = $(mode === "baseline" ? "timerBaseline" : "timerCompiled");
  const elapsed = record?.wall_ms ?? (start == null ? 0 : performance.now() - start);
  const nodes = (state.compiled?.graph?.[mode === "baseline" ? "source" : "compiled"]?.nodes || []);
  const seen = new Map(events.filter(e => e.node_id).map(e => [e.node_id, e]));
  const done = events.filter(e => e.type === "node_end").length;
  const list = nodes.map(node => {
    const event = seen.get(node.id);
    const status = event?.type === "node_end" ? (event.error ? "err" : event.cache_hit ? "hit" : "done") :
      event?.type === "node_wait" ? (event.reason === "quota" ? "quota wait" : "retry wait") :
      event ? "running" : "waiting";
    return `<li class="${status}"><span>${esc(node.id)}</span><span class="st">${status}</span></li>`;
  }).join("");
  element.innerHTML = `<strong>${mode === "baseline" ? "Original" : "Compiled"}</strong>` +
    `<div class="clock">${esc((elapsed / 1000).toFixed(2))} s</div>` +
    `<div class="progress"><div style="width:${nodes.length ? done / nodes.length * 100 : 0}%"></div></div>` +
    `<ul>${list}</ul>`;
}

function chosenModel() {
  const [provider, model] = $("liveModel").value.split("|");
  return { provider, model: model || null };
}

function showModelBudget() {
  const noModel = state.compiled?.source?.nodes &&
    !state.compiled.source.nodes.some(node => node.kind === "llm");
  if (noModel) $("liveModel").value = "mock|";
  $("liveModel").disabled = Boolean(noModel);
  const { provider, model } = chosenModel();
  const option = state.models?.find(row => row.provider === provider && row.model === model);
  const left = option?.remaining;
  $("budgetStatus").textContent = noModel ?
    "This workflow has no model steps. Model choice does not apply and uses no API calls." :
    provider === "mock" ? "Mock uses no model API calls." :
    left?.blocked || left?.rpd_remaining === 0 || left?.tpd_remaining === 0 ?
    `${provider} last reported or locally estimated an exhausted quota. Run will try the provider again; minute pacing still applies.` :
    `SYNAPSE local budget: ${left?.rpd_remaining ?? "—"} requests and ` +
    `${left?.tpd_remaining ?? "—"} tokens today; ${left?.rpm_remaining ?? "—"} requests and ` +
    `${left?.tpm_remaining ?? "—"} tokens this minute. Runs pause if the minute budget fills. ` +
    `Other apps may use the same provider quota.`;
  $("dataBadge").textContent = provider === "mock" ? "Synthetic data · mock bank" :
    `${provider === "groq" ? "Groq" : "Gemini"} models · mock bank`;
  $("dataBadge").classList.toggle("live", provider !== "mock");
  $("costNote").textContent = provider === "mock" ?
    "Mock costs are comparison units, not dollars." :
    "Live cost is an estimate at published paid-tier token rates. Free-tier usage may incur no charge. " +
    "Elapsed time includes quota waits. Wait totals and model time below sum across calls; " +
    "parallel calls can overlap, so these totals must not be subtracted from elapsed time.";
  const compare = provider === "mock" || $("compareLive").checked;
  $("compareLive").disabled = provider === "mock";
  $("liveRun").textContent = compare ? "Run original and compiled" : "Run compiled";
}

async function loadModels() {
  const previous = $("liveModel").value;
  state.models = await json("/models");
  $("liveModel").innerHTML = state.models.map(row =>
    `<option value="${esc(row.provider)}|${esc(row.model || "")}" ` +
    `>${esc(row.label)}` +
    `${row.remaining?.blocked || row.remaining?.rpd_remaining === 0 ?
      " · daily quota exhausted" : ""}</option>`).join("");
  if (state.models.some(row => `${row.provider}|${row.model || ""}` === previous))
    $("liveModel").value = previous;
  showModelBudget();
}

async function liveRun() {
  $("liveRun").disabled = true;
  $("workflow").disabled = true;
  const choice = { ...chosenModel(), warm: $("warmCache").checked };
  const compare = choice.provider === "mock" || $("compareLive").checked;
  $("liveStatus").textContent = compare ?
    (choice.provider === "mock" ? "Starting both mock runs…" :
      "Running original first, then compiled…") : "Running compiled workflow…";
  $("liveResult").innerHTML = "";
  $("timerBaseline").innerHTML = "";
  $("timerCompiled").innerHTML = "";
  const provider = choice.provider;
  const starts = { baseline: null, compiled: null };
  const events = { baseline: [], compiled: [] }, finished = {};
  const tick = setInterval(() => {
    for (const mode of compare ? ["baseline", "compiled"] : ["compiled"])
      renderTimer(mode, starts[mode], events[mode], finished[mode]);
  }, 100);
  try {
    const onEvent = (mode, event) => {
        if (event.type === "run_start") starts[mode] = performance.now();
        events[mode].push(event);
        if (event.type === "run_end") finished[mode] = { wall_ms: event.wall_ms };
      };
    const records = compare ? await runPair($("liveCase").value,
      Number($("liveScale").value), choice, onEvent) :
      { compiled: await runOne($("liveCase").value, Number($("liveScale").value),
                               choice, "compiled", onEvent) };
    for (const mode of ["baseline", "compiled"].filter(mode => records[mode]))
      renderTimer(mode, starts[mode], events[mode], records[mode]);
    const a = records.baseline, b = records.compiled;
    const before = a && recordUsage(a), after = recordUsage(b);
    const unit = b.cost_unit || "mock units";
    $("liveStatus").textContent = b.provider === "mock" ?
      `Finished ${b.case_id} offline with synthetic bank data.` :
      `Finished ${b.case_id} with ${b.provider === "groq" ? "Groq" : "Gemini"} model calls and synthetic bank data.`;
    const waited = [before, after].filter(Boolean).some(u =>
      (u.quota_wait_ms || 0) + (u.retry_wait_ms || 0) > 100);
    const decision = a?.key_field && b.key_field ?
      (a.key_value === b.key_value ? "Decision matches" : "Decision differs") :
      "See outputs and validation for correctness";
    const timing = compare ?
      metric("Original elapsed time", ms(a.wall_ms), a.key_value) +
      metric("Optimized elapsed time", ms(b.wall_ms), b.key_value, savingClass(a.wall_ms, b.wall_ms)) +
      metric("Elapsed time comparison", timeRatio(a.wall_ms, b.wall_ms),
        waited ? "Includes quota / retry waiting; not a compiler speed benchmark" : decision,
        savingClass(a.wall_ms, b.wall_ms)) :
      metric("Optimized elapsed time", ms(b.wall_ms), b.key_value);
    const usage = compare ? [
      metric("Original tokens", tokens(before.tokens_total),
        `${tokens(before.tokens_in)} in · ${tokens(before.tokens_out)} out`),
      metric("Optimized tokens", tokens(after.tokens_total),
        `${tokens(after.tokens_in)} in · ${tokens(after.tokens_out)} out`),
      metric(deltaLabel(before.tokens_total, after.tokens_total, "Total token"),
        change(before.tokens_total, after.tokens_total, tokens), "Includes output variation",
        savingClass(before.tokens_total, after.tokens_total)),
      metric(deltaLabel(before.tokens_in, after.tokens_in, "Input token"),
        change(before.tokens_in, after.tokens_in, tokens), "Prompt and schema usage",
        savingClass(before.tokens_in, after.tokens_in)),
      metric("Original cost", cost(before.cost, unit)),
      metric("Optimized cost", cost(after.cost, unit)),
      metric(deltaLabel(before.cost, after.cost, "Cost"),
        change(before.cost, after.cost, value => cost(value, unit)), "",
        savingClass(before.cost, after.cost))
    ].join("") :
      metric("Optimized tokens", tokens(after.tokens_total),
        `${tokens(after.tokens_in)} in · ${tokens(after.tokens_out)} out`) +
      metric("Optimized cost", cost(after.cost, unit), "Compare both to measure original cost");
    const waits = Object.entries(records).map(([mode, record]) => {
      const u = recordUsage(record), label = mode === "baseline" ? "Original" : "Optimized";
      return metric(`${label} quota wait`, ms(u.quota_wait_ms || 0), "Sum across model calls") +
        metric(`${label} model time`, ms(u.provider_ms || 0),
          `${u.calls ?? "—"} API calls · retry wait ${ms(u.retry_wait_ms || 0)}`);
    }).join("");
    $("liveResult").innerHTML = metric("Optimized cache", b.cold_cache ? "Cold" : "Warm",
      "Warm savings apply to repeated identical inputs") + timing + usage + (provider !== "mock" ? waits : "");
    if (choice.warm) $("liveStatus").textContent += b.cold_cache ?
      " Cache started empty; repeat this exact case to reuse results. Writes still execute." :
      ` Reused ${b.nodes.filter(n => n.cache_hit).length} exact results; writes still executed.`;
    if (waited) $("liveStatus").textContent +=
      " Quota pacing affected this run. The elapsed comparison is not evidence of compiler speed.";
    if ([before, after].filter(Boolean).some(u => u.usage_complete === false))
      $("liveStatus").textContent += " Some model usage was not reported; cost is incomplete.";
    $("liveGantt").innerHTML = gantt(records) + usageTable(records);
  } catch (exc) {
    error("liveGantt", exc);
    if (exc.records) {
      $("liveResult").innerHTML = Object.entries(exc.records).map(([mode, record]) => {
        const u = recordUsage(record);
        renderTimer(mode, starts[mode], events[mode], record);
        return metric(`${mode} reported tokens`, tokens(u.tokens_total),
          record.status === "failed" ? "Partial run; no savings comparison" : "Completed run") +
          metric(`${mode} reported cost`, cost(u.cost, record.cost_unit),
            u.usage_complete === false ? "Provider did not report all usage" : "Includes failed structured replies");
      }).join("");
      $("liveGantt").innerHTML += usageTable(exc.records);
    }
    $("liveStatus").textContent = `Run failed on ${provider}. You can try this model again or select another model.`;
  }
  finally {
    clearInterval(tick); $("liveRun").disabled = false;
    $("workflow").disabled = false;
    await loadModels().catch(() => {});
  }
}

function usageTable(records) {
  const rows = Object.entries(records).flatMap(([mode, record]) => (record.nodes || [])
    .filter(node => node.usage.model || node.cache_hit).map(node => {
      const u = node.usage;
      return `<tr><td>${esc(mode)}</td><td>${esc(node.node_id)}</td>` +
        `<td>${esc(u.model || (node.cache_hit ? "cached" : "—"))}</td>` +
        `<td>${tokens(u.tokens_in)} / ${tokens(u.tokens_out)}</td>` +
        `<td>${esc(cost(u.cost, record.cost_unit))}</td>` +
        `<td>${ms(u.quota_wait_ms || 0)}</td><td>${ms(u.provider_ms || 0)}</td></tr>`;
    })).join("");
  return `<div class="scroll"><table><thead><tr><th>Run</th><th>Step</th><th>Model</th>` +
    `<th>Tokens in / out</th><th>Cost</th><th>Quota wait</th><th>Model time</th>` +
    `</tr></thead><tbody>${rows}</tbody></table></div>`;
}

async function loadWorkflow() {
  state.selected = $("workflow").value;
  state.pair = null; state.validation = null; state.compiled = null;
  $("timerBaseline").innerHTML = "";
  $("timerCompiled").innerHTML = "";
  $("liveResult").innerHTML = "";
  $("liveStatus").textContent = "Choose a model, then run one synthetic case.";
  $("liveGantt").innerHTML = '<p class="muted">Press run to start.</p>';
  const chosen = state.workflows.find(w => w.file === state.selected);
  $("unsafe").checked = false;
  $("unsafe").disabled = chosen?.name !== "loan_prescreen";
  $("unsafeLabel").title = $("unsafe").disabled ? "The unsafe demo pass is only defined for the loan workflow" : "";
  selectCases();
  $("graphOriginal").innerHTML = `<p class="muted">Compiling…</p>`;
  try {
    state.compiled = await post("/compile", { workflow: state.selected });
    renderGraph(); renderDecisions(); showModelBudget();
  } catch (exc) { error("graphOriginal", exc); }
  await loadBench();
}

function setupControls() {
  document.querySelectorAll("[data-screen]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-screen]").forEach(tab =>
      tab.setAttribute("aria-selected", tab === button ? "true" : "false"));
    document.querySelectorAll(".screen").forEach(screen =>
      screen.classList.toggle("active", screen.id === button.dataset.screen));
  }));
  document.querySelectorAll("[data-bench-mode]").forEach(button => button.addEventListener("click", () => {
    state.benchMode = button.dataset.benchMode;
    document.querySelectorAll("[data-bench-mode]").forEach(b =>
      b.setAttribute("aria-pressed", b === button ? "true" : "false"));
    loadBench();
  }));
  const themes = ["auto", "light", "dark"];
  let theme = localStorage.getItem("synapse-theme") || "auto";
  const applyTheme = () => {
    document.documentElement.dataset.theme = theme === "auto" ? "" : theme;
    $("theme").textContent = `Theme: ${theme}`;
    localStorage.setItem("synapse-theme", theme);
  };
  applyTheme();
  $("theme").addEventListener("click", () => { theme = themes[(themes.indexOf(theme) + 1) % themes.length]; applyTheme(); });
  $("workflow").addEventListener("change", loadWorkflow);
  $("timelineRun").addEventListener("click", timelineRun);
  $("validateRun").addEventListener("click", validateRun);
  $("liveRun").addEventListener("click", liveRun);
  $("liveModel").addEventListener("change", showModelBudget);
  $("compareLive").addEventListener("change", showModelBudget);
}

async function init() {
  setupControls();
  try {
    state.workflows = (await json("/workflows")).filter(w => !w.error);
    await loadModels();
    $("workflow").innerHTML = state.workflows.map(w =>
      `<option value="${esc(w.file)}">${esc(w.name)}</option>`).join("");
    if (!state.workflows.length) throw new Error("No runnable workflows were found.");
    await loadWorkflow();
  } catch (exc) { error("graphOriginal", exc); }
}

init();
