/* RAM_Agent — plain JS, no build step, works offline. */
"use strict";

const $  = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const state = {
  models: [], harnesses: [], projects: [], settings: null,
  loaded: null, run: null, clock: null, images: [], busy: false,
  hardware: null,
};

/* ------------------------------------------------------------------ utils */

const fmtBytes = (b) => {
  if (!b) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1000)));
  return (b / Math.pow(1000, i)).toFixed(i ? 1 : 0) + " " + u[i];
};

const fmtDuration = (s) => {
  if (s === null || s === undefined) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m ${String(sec).padStart(2, "0")}s`;
  if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return `${sec}s`;
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res.text();
}

const post = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

/* ------------------------------------------------------------- navigation */

$("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === btn));
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("active", v.id === "view-" + btn.dataset.view));
  if (btn.dataset.view === "history") loadHistory();
  if (btn.dataset.view === "bench") renderBench();
  if (btn.dataset.view === "tools") renderTools();
  if (btn.dataset.view === "games") renderGames();
});

document.querySelector(".out-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".out-tab");
  if (!btn) return;
  document.querySelectorAll(".out-tab").forEach((t) => t.classList.toggle("active", t === btn));
  document.querySelectorAll(".pane").forEach((p) =>
    p.classList.toggle("active", p.id === "pane-" + btn.dataset.pane));
});

/* ----------------------------------------------------------------- models */

function renderModels() {
  const sel = $("model");
  const keep = sel.value;
  sel.innerHTML = "";
  state.models.forEach((m) => {
    const o = el("option");
    o.value = m.id;
    let badge = m.state;
    if (m.state === "absent")      badge = `not downloaded · ${m.size_gb} GB`;
    else if (m.state === "downloading") badge = `downloading ${m.progress.percent}%`;
    else if (m.id === state.loaded)     badge = "loaded";
    o.textContent = `${m.name} — ${badge}`;
    sel.appendChild(o);
  });
  if (keep && state.models.some((m) => m.id === keep)) sel.value = keep;
  renderModelCaps();
}

function currentModel() {
  return state.models.find((m) => m.id === $("model").value) || null;
}

function renderModelCaps() {
  const m = currentModel();
  const box = $("model-caps");
  box.innerHTML = "";
  if (!m) return;

  const tag = (label, ok) => {
    const t = el("span", "tag " + (ok ? "yes" : "no"), `${label} ${ok ? "yes" : "no"}`);
    box.appendChild(t);
  };
  tag("tools", m.tools);
  tag("vision", m.vision);
  tag("GPU", m.gpu);
  box.appendChild(el("span", "tag", `ctx ${m.context_ceiling.toLocaleString()}`));

  if (m.state === "error") {
    box.appendChild(el("div", "caps warn", "Error: " + (m.error || "unknown")));
  }
  if (!m.tools) {
    box.appendChild(el("div", "caps warn",
      "This model cannot call tools — Direct chat only."));
  }
  if (m.notes) box.appendChild(el("div", "caps", m.notes));

  // Download / load buttons reflect the state machine.
  $("download-btn").classList.toggle("hidden",
    !(m.state === "absent" || m.state === "error"));
  $("load-btn").classList.toggle("hidden",
    !(m.state === "ready" && m.id !== state.loaded));
  const dl = m.state === "downloading";
  $("model-progress").classList.toggle("hidden", !dl);
  if (dl) {
    $("model-progress-fill").style.width = m.progress.percent + "%";
    const speed = m.progress.speed_bps ? fmtBytes(m.progress.speed_bps) + "/s" : "…";
    const eta = m.progress.eta_seconds ? " · ETA " + fmtDuration(m.progress.eta_seconds) : "";
    $("model-progress-text").textContent =
      `${m.progress.percent}% · ${fmtBytes(m.progress.downloaded_bytes)} / ` +
      `${fmtBytes(m.progress.total_bytes)} · ${speed}${eta}`;
  }
  renderHarnessCaps();
  updateAttachAvailability();
}


/* -------------------------------------------------------------- catalogue */
/* The model catalogue answers three questions at a glance: is this model on
   this machine, would it fit if it were, and is it any good.
 *
 * The status mark is the interactive part. Green is informational. Red is a
 * button: it opens the HF page for that model and shows exactly which files to
 * fetch and where to put them, because downloading by hand is a supported path
 * and not a fallback -- a 27 GB pull is something people reasonably want to do
 * with their own tooling, or resume, or copy off another machine. */

function renderCatalog() {
  const body = $("catalog-body");
  const hw = state.hardware;
  body.innerHTML = "";

  const present = state.models.filter((m) => m.present).length;
  $("catalog-sub").textContent =
    `${present} of ${state.models.length} on disk`;

  if (hw) {
    $("catalog-hw").textContent =
      `${hw.ram_total_gb} GB RAM + ${(hw.vram_total_mb / 1024).toFixed(0)} GB VRAM ` +
      `= ${hw.fast_total_gb} GB fast memory · ${hw.usable_gb} GB usable for a model`;
  }

  state.models.forEach((m) => {
    const row = el("div", "catalog-row"
      + (m.default ? " is-default" : "")
      + (m.id === state.loaded ? " is-loaded" : ""));

    // 1. status
    const mark = el("button", "mark " + (m.present ? "present" : "absent"),
      m.present ? "\u2713" : "\u2717");
    mark.title = m.present
      ? "On disk at " + m.target_dir
      : "Not downloaded — click for the files and where to put them";
    mark.setAttribute("aria-label",
      (m.present ? "Downloaded: " : "Not downloaded: ") + m.name);
    if (!m.present) mark.addEventListener("click", () => showModelDetail(m.id));
    row.appendChild(mark);

    // 2. name + family/licence
    const name = el("div", "cat-name");
    name.appendChild(el("span", null, m.name + (m.default ? "  (default)" : "")));
    name.appendChild(el("span", "cat-sub",
      [m.family, m.params_total && `${m.params_total}/${m.params_active} active`,
       m.license, m.reasoning && `reasoning: ${m.reasoning}`]
        .filter(Boolean).join(" · ")));
    row.appendChild(name);

    // 3. size, coloured by whether it fits
    const cls = m.fits ? "" : (m.fits_at_all ? " tight" : " toobig");
    const size = el("div", "cat-size" + cls, m.size_gb.toFixed(1) + " GB");
    size.title = m.fit_note || "";
    row.appendChild(size);

    // 4. capability tags
    const caps = el("div", "cat-bench");
    if (m.vision) caps.appendChild(el("span", "b", "vision"));
    if (m.tools)  caps.appendChild(el("span", "b", "tools"));
    if (m.spec)   caps.appendChild(el("span", "b", "MTP"));
    caps.appendChild(el("span", "b",
      (m.context_ceiling / 1024).toFixed(0) + "k ctx"));
    row.appendChild(caps);

    // 5. benchmarks, whatever the registry happens to carry
    const bench = el("div", "cat-bench");
    Object.entries(m.benchmarks || {}).forEach(([k, v]) => {
      const b = el("span", "b", `${shortBench(k)} ${v}`);
      b.title = `${k}: ${v}`;
      bench.appendChild(b);
    });
    row.appendChild(bench);

    // 6. action
    const act = el("div");
    if (m.present && m.id !== state.loaded && m.state === "ready") {
      const load = el("button", "mini", "Load");
      load.addEventListener("click", () => loadModel(m.id));
      act.appendChild(load);
    } else if (!m.present && m.state !== "downloading") {
      const dl = el("button", "mini", "Download");
      dl.addEventListener("click", () => downloadModel(m.id));
      act.appendChild(dl);
    } else if (m.state === "downloading") {
      act.appendChild(el("span", "caps", m.progress.percent + "%"));
    }
    row.appendChild(act);

    body.appendChild(row);
  });
}

/* Benchmark keys are registry data and can be anything; shorten the ones we
   know and pass the rest through rather than dropping them. */
function shortBench(key) {
  const map = {
    "SWE-bench Verified": "SWE-V",
    "SWE-bench Pro": "SWE-Pro",
    "Terminal-Bench 2.0": "TB2",
    "MMLU-Pro": "MMLU",
  };
  return map[key] || key;
}

function showModelDetail(id) {
  const m = state.models.find((x) => x.id === id);
  const box = $("catalog-detail");
  if (!m) return;
  box.classList.remove("hidden");
  box.innerHTML = "";

  box.appendChild(el("h4", null, `${m.name} is not on this machine`));

  if (!m.fits_at_all) {
    box.appendChild(el("div", "caps warn", m.fit_note));
  }

  const ol = el("ol");

  const li1 = el("li");
  li1.appendChild(document.createTextNode("Open the repository "));
  const a = el("a", null, m.repo);
  a.href = m.hf_tree_url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  li1.appendChild(a);
  li1.appendChild(document.createTextNode(
    " — pinned to the exact revision this registry expects, not main."));
  ol.appendChild(li1);

  const li2 = el("li");
  li2.appendChild(document.createTextNode("Download "
    + (m.missing_files.length === 1 ? "this file:" : "these files:")));
  const ul = el("ul");
  m.missing_files.forEach((f) => {
    const fu = (m.file_urls || []).find((x) => x.name === f);
    const item = el("li");
    if (fu) {
      const link = el("a", null, f);
      link.href = fu.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      item.appendChild(link);
    } else {
      item.appendChild(el("code", null, f));
    }
    ul.appendChild(item);
  });
  li2.appendChild(ul);
  ol.appendChild(li2);

  const li3 = el("li");
  li3.appendChild(document.createTextNode("Put them in this directory, "
    + "keeping the filenames exactly as they are:"));
  li3.appendChild(el("code", "drop-path", m.target_dir));
  ol.appendChild(li3);

  const li4 = el("li");
  li4.appendChild(document.createTextNode("Come back and press "));
  li4.appendChild(el("strong", null, "Refresh"));
  li4.appendChild(document.createTextNode("."));
  ol.appendChild(li4);

  box.appendChild(ol);

  const cmd = el("pre", null,
    `mkdir -p ${shq(m.target_dir)}\n` +
    m.missing_files.map((f) => {
      const fu = (m.file_urls || []).find((x) => x.name === f);
      return `curl -fL -o ${shq(m.target_dir + "/" + f)} \\\n  ${shq(fu ? fu.url : "")}`;
    }).join("\n"));
  box.appendChild(el("div", "caps", "Or, equivalently:"));
  box.appendChild(cmd);

  box.appendChild(el("div", "caps",
    "The Download button does all of this for you. This panel is here for "
    + "when you would rather not have the app hold a multi-gigabyte transfer: "
    + "it resumes, it can run elsewhere, and the files can be copied in."));

  const close = el("button", "mini", "Close");
  close.addEventListener("click", () => box.classList.add("hidden"));
  box.appendChild(close);
}

/* Single-quote for a shell, so a path with spaces in it pastes correctly. */
function shq(v) { return "'" + String(v).replace(/'/g, "'\\''") + "'"; }

/* Both the select and the catalogue call these, so the two cannot drift.
   Selecting the model first means the existing progress UI follows along. */
function downloadModel(id) {
  $("model").value = id;
  renderModelCaps();
  return post(`/api/models/${id}/download`);
}

function loadModel(id) {
  $("model").value = id;
  renderModelCaps();
  return post(`/api/models/${id}/load`);
}

async function refreshCatalog() {
  const note = $("catalog-refresh-note");
  note.textContent = "checking…";
  try {
    const d = await api("/api/models/refresh", { method: "POST" });
    state.models = d.models;
    state.hardware = d.hardware;
    renderCatalog();
    renderModels();
    note.textContent = d.changed.length
      ? `found ${d.changed.length} change(s): ${d.changed.join(", ")}`
      : `no change · ${d.present}/${d.total} on disk`;
  } catch (e) {
    note.textContent = "refresh failed: " + e.message;
  }
  setTimeout(() => (note.textContent = ""), 6000);
}

/* --------------------------------------------------------------- harnesses */

function renderHarnesses() {
  const sel = $("harness");
  sel.innerHTML = "";
  state.harnesses.forEach((h) => {
    const o = el("option");
    o.value = h.id;
    o.textContent = `${h.name}${h.version ? " · " + h.version : ""}`;
    sel.appendChild(o);
  });
  renderHarnessCaps();
}

function currentHarness() {
  return state.harnesses.find((h) => h.id === $("harness").value) || null;
}

function renderHarnessCaps() {
  const h = currentHarness(), m = currentModel();
  const box = $("harness-caps");
  box.innerHTML = "";
  if (!h) return;

  // A model with no tool calling cannot drive a harness at all.
  const sel = $("harness");
  Array.from(sel.options).forEach((o) => {
    const hh = state.harnesses.find((x) => x.id === o.value);
    const blocked = m && !m.tools && hh && hh.id !== "direct";
    o.disabled = blocked;
    o.textContent = `${hh.name}${hh.version ? " · " + hh.version : ""}` +
                    (blocked ? " — needs tool calling" : "");
  });
  if (m && !m.tools && h.id !== "direct") sel.value = "direct";

  if (h.url) {
    const a = el("a", "", h.url);
    a.href = h.url; a.target = "_blank"; a.rel = "noreferrer";
    const wrap = el("div", "caps"); wrap.append("Project: ", a);
    box.appendChild(wrap);
  }
  if (h.mcp === "adapter") {
    box.appendChild(el("div", "caps warn",
      "MCP via pi-mcp-adapter — Pi ships no MCP support itself."));
  }
  if (h.baseline_tokens && m && h.baseline_tokens > m.context_ceiling) {
    box.appendChild(el("div", "caps warn",
      `Baseline prompt (${h.baseline_tokens} tokens) exceeds this model's ` +
      `context ceiling (${m.context_ceiling}).`));
  } else if (h.baseline_tokens) {
    box.appendChild(el("div", "caps",
      `Baseline prompt: ${h.baseline_tokens.toLocaleString()} tokens.`));
  }
  if (h.notes) box.appendChild(el("div", "caps", h.notes));
}

/* ---------------------------------------------------------------- projects */

function renderProjects() {
  const sel = $("project");
  const keep = sel.value;
  sel.innerHTML = "";
  if (!state.projects.length) {
    sel.appendChild(el("option", "", "— no projects yet —"));
  }
  state.projects.forEach((p) => {
    const o = el("option");
    o.value = p.name;
    o.textContent = p.name + (p.mcp_installed ? "" : "  (no MCP addon)");
    sel.appendChild(o);
  });
  if (keep) sel.value = keep;
}

/* ------------------------------------------------------------------ images */

function updateAttachAvailability() {
  const m = currentModel();
  const can = !!(m && m.vision);
  const label = $("attach-label");
  label.classList.toggle("disabled", !can);
  $("attach").disabled = !can;
  label.title = can
    ? "Attach a screenshot (sent over the OpenAI protocol)"
    : m
      ? `${m.name} is text-only, so images cannot be sent.`
      : "";
  if (!can && state.images.length) { state.images = []; renderAttachments(); }
}

function renderAttachments() {
  const box = $("attachments");
  box.innerHTML = "";
  state.images.forEach((src, i) => {
    const wrap = el("div", "thumb");
    const img = el("img"); img.src = src;
    const x = el("button", "x", "×");
    x.onclick = () => { state.images.splice(i, 1); renderAttachments(); };
    wrap.append(img, x);
    box.appendChild(wrap);
  });
}

function addImageFile(file) {
  const m = currentModel();
  if (!m || !m.vision) return;
  const r = new FileReader();
  r.onload = () => { state.images.push(r.result); renderAttachments(); };
  r.readAsDataURL(file);
}

$("attach").addEventListener("change", (e) =>
  Array.from(e.target.files).forEach(addImageFile));

const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => {
  e.preventDefault(); dz.classList.add("drag");
}));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
  e.preventDefault(); dz.classList.remove("drag");
}));
dz.addEventListener("drop", (e) =>
  Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"))
    .forEach(addImageFile));
$("prompt").addEventListener("paste", (e) => {
  Array.from(e.clipboardData.items)
    .filter((it) => it.type.startsWith("image/"))
    .forEach((it) => addImageFile(it.getAsFile()));
});

/* --------------------------------------------------------------- time limit */

const limitSeconds = () =>
  Math.max(1, parseInt($("limit-value").value || "1", 10)) *
  parseInt($("limit-unit").value, 10);

async function pushLimit() {
  await post("/api/settings", {
    time_limit_enabled: $("limit-enabled").checked,
    time_limit_seconds: limitSeconds(),
  });
}
["limit-enabled", "limit-value", "limit-unit"].forEach((id) =>
  $(id).addEventListener("change", pushLimit));

/* ---------------------------------------------------------------- run flow */

function setBusy(busy) {
  state.busy = busy;
  $("run-btn").disabled = busy;
  $("stop-btn").disabled = !busy;
}

$("run-btn").addEventListener("click", async () => {
  const m = currentModel(), h = currentHarness();
  if (!m || !h) return;
  clearOutput();
  try {
    setBusy(true);
    await post("/api/run", {
      model_id: m.id,
      harness_id: h.id,
      project: $("project").value,
      prompt: $("prompt").value,
      images: state.images,
    });
  } catch (err) {
    setBusy(false);
    appendTranscript({ type: "error", error: err.message });
  }
});

$("stop-btn").addEventListener("click", () => post("/api/run/stop"));

$("continue-btn").addEventListener("click", async () => {
  const extra = Math.max(1, parseInt($("extend-value").value || "1", 10)) *
                parseInt($("extend-unit").value, 10);
  $("decision").classList.add("hidden");
  await post("/api/run/continue", { extra_seconds: extra });
});

$("finish-btn").addEventListener("click", async () => {
  $("decision").classList.add("hidden");
  await post("/api/run/stop");
});

$("download-btn").addEventListener("click", () => downloadModel($("model").value));
$("cancel-download").addEventListener("click", () =>
  post(`/api/models/${$("model").value}/cancel`));
$("load-btn").addEventListener("click", () => loadModel($("model").value));
$("catalog-refresh").addEventListener("click", refreshCatalog);
$("model").addEventListener("change", renderModelCaps);
$("harness").addEventListener("change", renderHarnessCaps);

$("new-project").addEventListener("click", async () => {
  const name = prompt("New Godot project name:");
  if (!name) return;
  try {
    const res = await post("/api/projects", { name });
    state.projects = res.projects;
    renderProjects();
    $("project").value = name;
  } catch (err) { alert(err.message); }
});

/* ------------------------------------------------------------------ output */

function clearOutput() {
  $("pane-transcript").innerHTML = "";
  $("pane-stderr").textContent = "";
  $("decision").classList.add("hidden");
}

function appendTranscript(ev) {
  const pane = $("pane-transcript");
  const atBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 60;
  let node;
  if (ev.type === "text") {
    const last = pane.lastElementChild;
    if (last && last.classList.contains("text")) {
      last.textContent += ev.text;               // stream into one block
      if (atBottom) pane.scrollTop = pane.scrollHeight;
      return;
    }
    node = el("div", "msg text", ev.text);
  } else if (ev.type === "tool_call") {
    node = el("details", "tool");
    node.appendChild(el("summary", "", `▸ ${ev.name}`));
    const pre = el("pre", "", typeof ev.args === "string"
      ? ev.args : JSON.stringify(ev.args, null, 2));
    node.appendChild(pre);
  } else if (ev.type === "tool_result") {
    node = el("details", "tool" + (ev.is_error ? " err" : ""));
    node.appendChild(el("summary", "",
      (ev.is_error ? "✕ result" : "✓ result") + (ev.id ? ` · ${ev.id}` : "")));
    node.appendChild(el("pre", "", typeof ev.result === "string"
      ? ev.result : JSON.stringify(ev.result, null, 2)));
  } else if (ev.type === "error") {
    node = el("div", "msg err", "Error: " + (ev.error || ""));
  } else {
    // Unparseable harness output is shown, never silently dropped.
    node = el("div", "msg raw", ev.line || JSON.stringify(ev));
  }
  pane.appendChild(node);
  if (atBottom) pane.scrollTop = pane.scrollHeight;
}

function renderMetrics(m, clock) {
  if (m) {
    $("m-phase").textContent = m.phase;
    $("m-ttft").textContent = m.ttft_seconds !== null ? fmtDuration(m.ttft_seconds) : "–";
    $("m-ttft-tool").textContent = m.time_to_first_tool_call !== null
      ? fmtDuration(m.time_to_first_tool_call) : "–";
    $("m-prefill").textContent = m.prefill_tps ?? "–";
    $("m-decode").textContent = m.decode_tps ?? "–";
    $("m-tokens").textContent = `${m.tokens_in} / ${m.tokens_out}`;
    $("m-tools").textContent = m.tool_calls;
  }
  if (clock) {
    $("m-elapsed").textContent = fmtDuration(clock.elapsed);
    $("countdown").textContent = clock.remaining !== null
      ? "limit in " + fmtDuration(clock.remaining) : "no time limit";
  }
}

/* --------------------------------------------------------------------- SSE */

function connect() {
  const es = new EventSource("/api/stream");

  es.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch (_) { return; }

    switch (ev.type) {
      case "hello":
        setBusy(!!ev.busy);
        if (ev.run) state.run = ev.run;
        if (ev.awaiting_decision) $("decision").classList.remove("hidden");
        break;
      case "run_start":
        setBusy(true); state.run = ev.run; break;
      case "run_end":
        setBusy(false);
        $("decision").classList.add("hidden");
        renderMetrics(ev.run.metrics, null);
        appendTranscript({ type: "raw",
          line: `— run ${ev.run.outcome} after ${fmtDuration(ev.run.wall_seconds)} —` });
        refreshModels();
        break;
      case "event":       appendTranscript(ev.event); break;
      case "metrics":     renderMetrics(ev.metrics, ev.clock); break;
      case "phase":       $("m-phase").textContent = ev.phase; break;
      case "time_limit_reached":
        // A benchmark cell finalises itself rather than waiting for an answer,
        // so there is nothing to decide -- just say what happened.
        if (ev.unattended) {
          appendTranscript({ type: "raw",
            line: `— time limit reached after ${fmtDuration(ev.elapsed)}; ` +
                  `output saved, cell recorded as time_limit —` });
          break;
        }
        $("decision-detail").textContent =
          `Stopped after ${fmtDuration(ev.elapsed)}. Everything produced so far has been saved.`;
        $("decision").classList.remove("hidden");
        break;
      case "stderr":
        $("pane-stderr").textContent += ev.line + "\n"; break;
      case "engine_log":
        $("pane-engine").textContent += ev.line + "\n"; break;
      case "model_progress":
      case "model_state":
      case "model_error":
        refreshModels(); break;
      case "bench_cell_start": {
        const c = ev.cell;
        const line = `[${ev.index}/${ev.total}] ${c.task_id} · ${c.model_id} × ${c.harness_id}`;
        $("bench-results").prepend(el("div", "caps", line));
        appendTranscript({ type: "raw", line: "— " + line + " —" });
        break;
      }
      case "bench_cell_end": {
        const passed = Object.values(ev.checks || {}).filter(Boolean).length;
        const total = Object.keys(ev.checks || {}).length;
        $("bench-results").prepend(
          el("div", "caps", `    → ${ev.outcome} (${passed}/${total} checks)`));
        // A finished cell may have produced a playable game.
        renderGames().catch(() => {});
        break;
      }
      case "bench_cell_error":
        $("bench-results").prepend(
          el("div", "caps warn", `    → error: ${ev.error}`));
        break;
      case "bench_done":
        $("bench-results").prepend(el("div", "caps", "Matrix complete."));
        loadHistory();
        break;
      case "error":
        appendTranscript({ type: "error", error: ev.error }); break;
    }
  };

  // EventSource reconnects on its own, but a long prefill plus a sleeping
  // laptop can still drop it; this makes the retry explicit.
  es.onerror = () => { es.close(); setTimeout(connect, 3000); };
}

/* ------------------------------------------------------------------- games */

async function renderGames() {
  const d = await api("/api/games");
  const wrap = $("games-list");
  wrap.innerHTML = "";

  $("games-watcher").textContent = d.watcher_running
    ? "Play launches games directly."
    : "Play needs scripts/play-watcher.sh running on the host — " +
      "until then, use the command shown on each card.";

  if (!d.games.length) {
    wrap.appendChild(el("div", "caps",
      "No games yet. Run the G1–G3 tasks from the Benchmark tab."));
    return;
  }

  d.games.forEach((g) => {
    const card = el("div", "game-card " + (g.playable ? "playable" : "broken"));

    const head = el("div", "game-head");
    head.appendChild(el("span", "game-title", `${g.task_id} · ${g.task_name}`));
    head.appendChild(el("span", "game-meta",
      `${g.model_id} × ${g.harness_id} · ${fmtDuration(g.wall_seconds)}`));
    head.appendChild(el("span", "outcome " + g.outcome, g.outcome));

    const actions = el("div", "game-actions");
    const play = el("button", "play", "▶ Play");
    play.disabled = !g.playable;
    play.title = g.playable ? "Open this game in a window"
                            : "This build does not load cleanly";
    play.addEventListener("click", async () => {
      play.disabled = true;
      play.textContent = "launching…";
      try {
        const r = await post(`/api/games/${encodeURIComponent(g.project)}/play`);
        play.textContent = r.watcher_running ? "▶ launched" : "▶ queued";
        if (!r.watcher_running) {
          alert("No play watcher is running on the host.\n\nStart it with:\n" +
                "  scripts/play-watcher.sh &\n\nor run the game directly:\n  " +
                r.command);
        }
      } catch (err) {
        alert(err.message);
        play.textContent = "▶ Play";
      }
      setTimeout(() => { play.disabled = !g.playable; play.textContent = "▶ Play"; }, 4000);
    });
    actions.appendChild(play);

    const recheck = el("button", "mini", "Re-check");
    recheck.addEventListener("click", async () => {
      recheck.disabled = true; recheck.textContent = "checking…";
      try {
        const v = await post(`/api/games/${encodeURIComponent(g.project)}/validate`);
        alert(`loads: ${v.loads}\nno script errors: ${v.no_script_errors}\n` +
              `ran clean: ${v.ran_clean}\n\n` +
              (v.errors.length ? v.errors.slice(0, 6).join("\n") : "no errors"));
      } catch (err) { alert(err.message); }
      recheck.disabled = false; recheck.textContent = "Re-check";
    });
    actions.appendChild(recheck);
    head.appendChild(actions);
    card.appendChild(head);

    // Per-check results, so a failure says which requirement is missing.
    const checks = el("div", "game-checks");
    Object.entries(g.checks || {}).forEach(([name, ok]) => {
      checks.appendChild(el("span", "check " + (ok ? "ok" : "bad"),
        (ok ? "✓ " : "✕ ") + name));
    });
    card.appendChild(checks);

    card.appendChild(el("div", "game-stats",
      `${g.scripts} script${g.scripts === 1 ? "" : "s"}, ${g.script_lines} lines · ` +
      `${g.scenes} scene${g.scenes === 1 ? "" : "s"} · ` +
      `${g.checks_passed}/${g.checks_total} checks · ` +
      `${(g.metrics && g.metrics.tool_calls) || 0} tool calls`));

    card.appendChild(el("div", "game-cmd", g.command));
    wrap.appendChild(card);
  });
}

$("refresh-games").addEventListener("click", renderGames);

/* ------------------------------------------------------------------- tools */

const DEFAULT_TOOLSETS = ["scene_edit", "scripts", "runtime"];
const MINIMAL_TOOLSETS = ["scene_edit", "scripts"];

function fmtTokens(n) { return n.toLocaleString(); }

async function renderTools() {
  const c = await api("/api/tools");
  state.tools = c;
  const t = c.totals;

  // Summary: the cost of what is on, and of what is being left out.
  const sum = $("tools-summary");
  sum.innerHTML = "";
  const tile = (label, value, cls) => {
    const d = el("div", "metric" + (cls ? " " + cls : ""));
    d.appendChild(el("span", "", label));
    d.appendChild(el("b", "", value));
    sum.appendChild(d);
  };
  tile("Enabled tools", String(t.enabled_tool_count), "good");
  tile("Enabled tokens", fmtTokens(t.enabled_tokens), "good");
  tile("Excluded tools", String(t.excluded_tool_count));
  tile("Excluded tokens", fmtTokens(t.excluded_tokens));
  tile("If all on", `${t.all_tool_count} / ${fmtTokens(t.all_tokens)}`, "warn");
  tile("Prefill @2 tok/s",
       fmtDuration(c.prefill_estimate_seconds.cpu_2_tok_s), "warn");

  // Always-on toolsets cannot be switched off, so they are stated, not offered.
  $("tools-always").innerHTML = "";
  const always = el("div", "caps", "");
  c.always_on.forEach((a) => {
    const tag = el("span", "tag yes", a.name);
    tag.title = a.description || "";
    $("tools-always").appendChild(tag);
  });
  $("tools-always").appendChild(el("div", "caps",
    `${c.baseline.tool_count} tools, ${fmtTokens(c.baseline.tokens)} tokens. ` +
    `Cannot be disabled — this is the floor for using MCP at all.`));

  // Every gated toolset, enabled or not, with its tools listed underneath.
  const list = $("tools-list");
  list.innerHTML = "";
  c.toolsets.forEach((ts) => {
    const box = el("div", "toolset " + (ts.enabled ? "on" : "off"));
    const head = el("div", "toolset-head");
    const cb = el("input");
    cb.type = "checkbox"; cb.checked = ts.enabled; cb.value = ts.name;
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.addEventListener("change", () => {
      box.classList.toggle("on", cb.checked);
      box.classList.toggle("off", !cb.checked);
      updateToolsPreview();
    });
    head.appendChild(cb);
    head.appendChild(el("span", "toolset-name", ts.name));
    if (ts.min_godot) {
      head.appendChild(el("span", "badge", "Godot " + ts.min_godot + "+"));
    }
    head.appendChild(el("span", "toolset-cost",
      `${ts.tool_count} tools · ${fmtTokens(ts.tokens)} tok`));
    head.appendChild(el("span", "toolset-desc", ts.description || ""));
    head.appendChild(el("span", "expand-hint", "click to list tools"));
    head.addEventListener("click", () => box.classList.toggle("expanded"));
    box.appendChild(head);

    const tools = el("div", "toolset-tools");
    ts.tools.forEach((tool) => {
      const row = el("div", "tool-row");
      row.appendChild(el("span", "tname", tool.name));
      row.appendChild(el("span", "tdesc", tool.description || ""));
      row.appendChild(el("span", "ttok", `${tool.tokens} tok`));
      tools.appendChild(row);
    });
    box.appendChild(tools);
    list.appendChild(box);
  });
  updateToolsPreview();
}

function checkedToolsets() {
  return Array.from($("tools-list").querySelectorAll("input:checked"))
    .map((i) => i.value);
}

function updateToolsPreview() {
  const c = state.tools;
  if (!c) return;
  const on = new Set(checkedToolsets());
  let tokens = c.baseline.tokens, count = c.baseline.tool_count;
  c.toolsets.forEach((ts) => {
    if (on.has(ts.name)) { tokens += ts.tokens; count += ts.tool_count; }
  });
  $("tools-note").textContent =
    `selection: ${count} tools · ${fmtTokens(tokens)} tokens · ` +
    `~${fmtDuration(tokens / 2)} of prefill at 2 tok/s`;
}

function applyToolsetPreset(names) {
  $("tools-list").querySelectorAll("input").forEach((cb) => {
    cb.checked = names.includes(cb.value);
    const box = cb.closest(".toolset");
    box.classList.toggle("on", cb.checked);
    box.classList.toggle("off", !cb.checked);
  });
  updateToolsPreview();
}

$("tools-default").addEventListener("click", () => applyToolsetPreset(DEFAULT_TOOLSETS));
$("tools-minimal").addEventListener("click", () => applyToolsetPreset(MINIMAL_TOOLSETS));
$("save-tools").addEventListener("click", async () => {
  try {
    state.tools = await post("/api/tools", { toolsets: checkedToolsets() });
    if (state.settings) state.settings.godot_toolsets = state.tools.enabled_toolsets;
    await renderTools();
    renderSettings();
    $("tools-note").textContent = "Saved. Applies to the next run.";
  } catch (err) { alert(err.message); }
});

/* ----------------------------------------------------------------- lineup */

async function renderLineup() {
  let d;
  try { d = await api("/api/lineup"); } catch (_) { return; }
  const excluded = d.excluded || [];
  $("lineup-sub").textContent =
    `${excluded.length} ruled out, with reasons`;
  const body = $("lineup-body");
  body.innerHTML = "";
  body.appendChild(el("div", "caps",
    "Every model here was considered and does not fit in 62 GB of RAM. " +
    "The interesting cases are not the obviously huge ones — they are the " +
    "models that 'fit' only by paging weights off the SSD, which is the " +
    "thing this project exists to stop doing."));
  excluded.forEach((m) => {
    const row = el("div", "lineup-row");
    row.appendChild(el("span", "lname", m.name));
    row.appendChild(el("span", "badge disk", `~${m.size_gb} GB`));
    row.appendChild(el("span", "lreason", (m.reason || "").trim()));
    body.appendChild(row);
  });
}

/* --------------------------------------------------------------- benchmark */

function checkboxList(container, items, checkedIds) {
  container.innerHTML = "";
  items.forEach((it) => {
    const lab = el("label");
    const cb = el("input");
    cb.type = "checkbox";
    cb.value = it.id;
    cb.checked = checkedIds ? checkedIds.includes(it.id) : true;
    lab.append(cb, document.createTextNode(it.label));
    container.appendChild(lab);
  });
}

const checkedValues = (container) =>
  Array.from(container.querySelectorAll("input:checked")).map((i) => i.value);

async function renderBench() {
  const { tasks } = await api("/api/bench/tasks");
  state.tasks = tasks;
  checkboxList($("bench-models"),
    state.models.map((m) => ({ id: m.id, label: `${m.name} (${m.size_gb} GB)` })));
  checkboxList($("bench-harnesses"),
    state.harnesses.map((h) => ({ id: h.id, label: h.name })),
    // Direct chat cannot build a game, so it is off by default in a matrix.
    state.harnesses.filter((h) => h.id !== "direct").map((h) => h.id));
  checkboxList($("bench-tasks"),
    tasks.map((t) => ({
      id: t.id,
      label: `${t.id} ${t.name}` + (t.requires_vision ? " (vision only)" : ""),
    })));
  previewMatrix();
}

async function previewMatrix() {
  const body = {
    model_ids: checkedValues($("bench-models")),
    harness_ids: checkedValues($("bench-harnesses")),
    task_ids: checkedValues($("bench-tasks")),
  };
  if (!body.model_ids.length || !body.task_ids.length) return;
  let res;
  try { res = await post("/api/bench/matrix", body); } catch (_) { return; }

  const wrap = $("bench-results");
  wrap.innerHTML = "";
  const summary = el("div", "caps",
    `${res.runnable} cell(s) will run, ${res.skipped} skipped.`);
  wrap.appendChild(summary);
  const skipped = res.cells.filter((c) => c.skipped);
  if (skipped.length) {
    const table = el("table");
    const head = el("tr");
    ["Task", "Model", "Harness", "Skipped because"].forEach((h) =>
      head.appendChild(el("th", "", h)));
    table.appendChild(head);
    skipped.forEach((c) => {
      const tr = el("tr");
      [c.task_id, c.model_id, c.harness_id, c.skipped].forEach((v) =>
        tr.appendChild(el("td", "", v)));
      table.appendChild(tr);
    });
    wrap.appendChild(table);
  }
}

["bench-models", "bench-harnesses", "bench-tasks"].forEach((id) =>
  $(id).addEventListener("change", previewMatrix));

$("bench-run").addEventListener("click", async () => {
  const body = {
    model_ids: checkedValues($("bench-models")),
    harness_ids: checkedValues($("bench-harnesses")),
    task_ids: checkedValues($("bench-tasks")),
  };
  try {
    const res = await post("/api/bench/run", body);
    $("bench-results").prepend(
      el("div", "caps", `Queued ${res.queued} cell(s). Progress appears here ` +
                        `and in the Run tab's transcript.`));
  } catch (err) { alert(err.message); }
});

/* ----------------------------------------------------------------- history */

async function loadHistory() {
  const { runs } = await api("/api/runs");
  const wrap = $("history-table");
  wrap.innerHTML = "";
  if (!runs.length) {
    wrap.appendChild(el("div", "caps", "No runs yet."));
    return;
  }
  const table = el("table");
  const head = el("tr");
  ["When", "Model", "Harness", "Task", "Outcome", "Wall", "TTFT",
   "1st tool", "Tools", "Dec tok/s", "Ext"].forEach((h) =>
    head.appendChild(el("th", "", h)));
  table.appendChild(head);
  runs.forEach((r) => {
    const m = r.metrics || {};
    const tr = el("tr");
    const cells = [
      new Date(r.started_at * 1000).toLocaleString(),
      r.model_id, r.harness_id, r.task_id || "—",
      null,
      fmtDuration(r.wall_seconds),
      m.ttft_seconds !== null && m.ttft_seconds !== undefined ? fmtDuration(m.ttft_seconds) : "–",
      m.time_to_first_tool_call ? fmtDuration(m.time_to_first_tool_call) : "–",
      m.tool_calls ?? 0,
      m.decode_tps ?? "–",
      r.time_limit_extensions || 0,
    ];
    cells.forEach((c, i) => {
      const td = el("td", i >= 5 ? "num-cell" : "");
      if (i === 4) td.appendChild(el("span", "outcome " + r.outcome, r.outcome));
      else td.textContent = c;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  wrap.appendChild(table);
}
$("refresh-history").addEventListener("click", loadHistory);

/* ---------------------------------------------------------------- settings */

function renderSettings() {
  const s = state.settings;
  if (!s) return;
  $("s-limit-enabled").checked = s.time_limit_enabled;
  const useHours = s.time_limit_seconds % 3600 === 0;
  $("s-limit-unit").value = useHours ? "3600" : "60";
  $("s-limit-value").value = useHours
    ? s.time_limit_seconds / 3600 : Math.round(s.time_limit_seconds / 60);
  $("s-inhibit").checked = s.inhibit_sleep;
  $("s-token").textContent = s.hf_token_set
    ? "A token is set." : "No token set (public repos still download).";

  $("s-toolsets-summary").textContent =
    s.godot_toolsets.length ? s.godot_toolsets.join(", ") : "none beyond the always-on pair";

  // Per-model engine flag overrides. Shown as editable KEY=VALUE lines so a
  // flag that is not in the registry can still be set without a code change.
  const flagsBox = $("s-flags");
  flagsBox.innerHTML = "";
  state.models.forEach((m) => {
    const wrap = el("div", "caps");
    wrap.appendChild(el("div", "", m.name));
    const ta = el("textarea");
    ta.rows = 3;
    ta.dataset.model = m.id;
    const overrides = (s.model_flags || {})[m.id] || {};
    ta.value = Object.entries(overrides).map(([k, v]) => `${k}=${v}`).join("\n");
    ta.placeholder = "DSV4_CUDA=1";
    wrap.appendChild(ta);
    flagsBox.appendChild(wrap);
  });

  // Mirror onto the Run tab controls.
  $("limit-enabled").checked = s.time_limit_enabled;
  $("limit-unit").value = useHours ? "3600" : "60";
  $("limit-value").value = $("s-limit-value").value;
}

$("goto-tools").addEventListener("click", (e) => {
  e.preventDefault();
  document.querySelector('.tab[data-view="tools"]').click();
});

$("save-settings").addEventListener("click", async () => {
  const modelFlags = {};
  $("s-flags").querySelectorAll("textarea").forEach((ta) => {
    const entries = ta.value.split("\n")
      .map((l) => l.trim()).filter(Boolean)
      .map((l) => { const i = l.indexOf("="); return i === -1 ? null
        : [l.slice(0, i).trim(), l.slice(i + 1).trim()]; })
      .filter(Boolean);
    if (entries.length) modelFlags[ta.dataset.model] = Object.fromEntries(entries);
  });
  state.settings = await post("/api/settings", {
    time_limit_enabled: $("s-limit-enabled").checked,
    time_limit_seconds: Math.max(1, parseInt($("s-limit-value").value, 10)) *
                        parseInt($("s-limit-unit").value, 10),
    inhibit_sleep: $("s-inhibit").checked,
    model_flags: modelFlags,
  });
  renderSettings();
  $("save-note").textContent = "Saved.";
  setTimeout(() => ($("save-note").textContent = ""), 2500);
});

/* -------------------------------------------------------------------- boot */

async function refreshModels() {
  const data = await api("/api/models");
  state.models = data.models;
  state.loaded = data.loaded;
  state.hardware = data.hardware || state.hardware;
  renderModels();
  renderCatalog();
  $("engine-pill").textContent = "Engine: " + (data.loaded || "idle");
  $("engine-pill").className = "pill" + (data.loaded ? " ok" : "");
}

async function refreshGodot() {
  try {
    const g = await api("/api/godot?project=" + encodeURIComponent($("project").value || ""));
    const pill = $("godot-pill");
    pill.textContent = "Godot: " + (g.editor_connected ? "connected" : "not connected");
    pill.className = "pill " + (g.editor_connected ? "ok" : "bad");
    pill.title = g.message;
    $("godot-hint").textContent = g.editor_connected ? "" : g.message;
  } catch (_) { /* backend restarting */ }
}

async function boot() {
  const [models, harnesses, projects, settings] = await Promise.all([
    api("/api/models"), api("/api/harnesses"), api("/api/projects"),
    api("/api/settings"),
  ]);
  state.models = models.models;
  state.loaded = models.loaded;
  state.hardware = models.hardware;
  state.harnesses = harnesses.harnesses;
  state.projects = projects.projects;
  state.settings = settings;
  renderModels(); renderCatalog(); renderHarnesses(); renderProjects();
  renderSettings();
  $("engine-pill").textContent = "Engine: " + (models.loaded || "idle");
  connect();
  renderLineup();
  refreshGodot();
  setInterval(refreshGodot, 10000);
  setInterval(() => { if (!state.busy) refreshModels(); }, 5000);
}

boot().catch((err) => {
  document.body.prepend(el("div", "msg err", "Failed to start: " + err.message));
});
