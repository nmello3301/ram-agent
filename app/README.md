# RAM_Agent

A local lab for measuring how fast a given **model × harness** combination can
build a 3D game and its assets. You pick a model and a harness in a page in
Firefox, type a prompt, and an agent drives Godot and Blender through MCP.
Everything runs on this laptop; nothing leaves it.

It is the successor to Colibri Local UI, and the difference is one decision:
**the model fits in RAM.** See [`../RAM_AGENT.md`](../RAM_AGENT.md) for why, and
for what the previous approach measured.

```
Firefox ── http://127.0.0.1:8088 ──▶ app container (host network, binds 127.0.0.1)
                                      ├─ FastAPI backend: UI, SSE, run history
                                      ├─ model manager: registry, GGUF downloads
                                      ├─ llama-server (CUDA, --n-cpu-moe)
                                      │    └─ OpenAI API on 127.0.0.1:8077
                                      ├─ harnesses (OpenCode, Goose, Pi)
                                      ├─ godot-editor-mcp ── ws://127.0.0.1:9080 ──┐
                                      ├─ blender-mcp-server ──────────────────┐    │
                                      └─ headless Godot 4.7.2 (validation)    │    │
Host: Godot 4.7.2 editor (Wayland, native) with the MCP addon ────────────────┼────┘
Host: Blender 5.x with the MCP extension ────────────────────────────────────┘
```

---

## Read this before you run anything

Three things shape this project, and none of them is inference speed.

1. **The 6 GB card is contested.** llama.cpp wants ~4–5.5 GB for attention, the
   KV cache and the vision projector. Blender Cycles wants the same card. So
   does the ComfyUI instance already on this machine. **You cannot GPU-render
   in Blender while the model is loaded.** Run Blender's viewport on EEVEE
   against the AMD iGPU, render Cycles on the CPU, or unload the model first.

2. **The budget is RAM + VRAM.** 62 GB + 6 GB = 68 GB of fast memory, ~55 GB
   of it usable for a model. Weights on the GPU come out of the VRAM column and
   the rest sit in RAM, so a model is sized against the sum. The disk is not in
   the budget — that is the whole point. The model catalogue in the UI shows
   each model against this number.

3. **`--jinja` is not optional.** Without it llama-server bypasses the model's
   own chat template and tool calls come back as literal `<tools>` text. Every
   "my local model can't call tools" report is this. The engine adapter always
   passes it and refuses to start a model whose GGUF carries no template.

---

## Quick start

```bash
cd ~/Desktop/RAM_Agent/app

bash scripts/host-setup.sh        # once; the only step that needs sudo
bash scripts/preflight.sh         # measures the machine, writes state/preflight.txt
bash scripts/install-godot.sh     # Godot 4.7.2 + the MCP addon, no root needed
bash scripts/install-blender.sh   # Blender 5.2.2 + the MCP bridge, no root needed
bash scripts/download-models.sh   # 28 GB. --all adds the sweep rows (~166 GB)
bash scripts/up.sh                # builds the image, starts the app, opens Firefox
```

Then open a project in the editor so the agent has something to drive:

```bash
bash scripts/godot-editor.sh "My Game"
```

In the editor: **Project → Project Settings → Plugins → Godot MCP → Enable.**
The indicator in the web UI turns green once the addon connects.

For the asset phase, Blender needs its bridge enabled once:
**Edit → Preferences → Add-ons → Install from Disk →**
`state/blender_mcp_addon.zip`, enable **Blender MCP Bridge**, then **N → MCP**
in the viewport. It must read `Listening on 127.0.0.1:9876`. As with Godot, the
add-on listens and the MCP server dials out to it.

Stop with `scripts/down.sh`. Models, projects, assets and run history live on
bind mounts and survive.

To run without Docker, build the engine natively and point the app at it:

```bash
bash scripts/build-llama.sh
export LLAMA_SERVER_BIN=~/Desktop/RAM_Agent/state/llama/bin/llama-server
```

---

## The models

| Model | Quant | Size | Active | Tools | Vision |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B** *(default)* | `UD-Q5_K_XL` | **27.5 GB** | 3B of 35B | yes | **yes** |
| Qwen3.6-35B-A3B | `UD-Q5_K_XL` + MTP | 28.1 GB | 3B | yes | yes |
| Qwen3.6-35B-A3B | `UD-Q4_K_XL` | 23.3 GB | 3B | yes | yes |
| Qwen3.6-35B-A3B | `Q8_0` | 37.8 GB | 3B | yes | yes |
| Qwen3-Coder-Next | `UD-Q4_K_S` | 46.1 GB | 3B of 80B | yes | no |

Filenames, byte sizes and revisions in `models.yaml` were read from the Hugging
Face API on 2026-09-22, not copied from prose.

**The model catalogue** in the UI lists every registry row with its stats and a
status mark: a green tick when the files are on disk, a red cross when they are
not. The cross is a button — it opens the repository at the pinned revision and
shows exactly which files to fetch, the directory to drop them in, and the
equivalent `curl` commands. **Refresh** re-reads the models directory, so files
copied in by hand or downloaded elsewhere are picked up without a restart. The
Download button does the same job in-app; the manual path exists because a
27 GB transfer is something people reasonably want to resume, script, or copy
off another machine.

**Nothing in the backend knows about any of these models.** Family, prompt
conventions, sampling and template arguments are all registry data. A test
fails the build if a backend module branches on a model family.

**Why Q5 and not Q8, on a machine with 62 GB.** Decode is RAM-bandwidth-bound,
so bytes read per token ≈ quant size × (active/total). Halving the quant roughly
doubles decode speed. Spare RAM is headroom for the KV cache, Godot and
Blender — not free quality. Q8 is a registry row so the question can be
*measured*, not a default.

**Why an 80B model decodes as fast as a 35B one.** Active bytes, not total
bytes, set the speed. Qwen3-Coder-Next activates 3B of 80B, so per token it
reads about what the 35B does — it just costs 46 GB of a ~55 GB budget to hold,
which is why it is the one row that cannot share the machine with a large
Blender scene, and why it is not the default.

The UI's lineup panel lists what was ruled out and why. The interesting entry is
Qwen3.8-Flash-Next: it "fits" in 46 GB only by paging a 51B n-gram table off the
SSD, which is exactly the dependency this project exists to escape.

---

## The harnesses

| Harness | Version | MCP | Role |
|---|---|---|---|
| Direct chat | — | — | speed baseline; cannot build a game |
| **OpenCode** | 1.18.31 | native | **the default** |
| Goose | 1.50.1 | native | comparison: largest preamble |
| Pi | 0.85.1 | via adapter | control |

All of them talk to llama-server over the **OpenAI** protocol.

**Colibri's harness argument has inverted.** It chose Pi because its preamble is
under 1,000 tokens against 10–20k for an MCP harness, and at under 2 tok/s of
prefill that difference was about an hour per turn. At 100–400 tok/s it is
under a minute. Preamble size is no longer the design constraint, so the
harness with the best agent loop wins. Pi stays as a control; its original
justification is gone.

---

## Phases

The agent runs in one of two phases:

```
  asset phase            handoff              integration phase
  ───────────            ───────              ─────────────────
  Blender MCP    ──▶   .glb / .gltf   ──▶     Godot MCP
   (27 tools)          in res://assets         (54 tools)
```

All numbers below were **probed from the running servers** with
`scripts/probe-mcp.sh`, not taken from release notes:

| Phase | Config | Tools | Tokens | of 64k |
|---|---|---|---|---|
| `godot` | scene_edit + scripts + runtime | 54 | 7,312 | 11% |
| `blender` | the whole catalogue | 27 | 3,183 | 5% |
| `both` | both | 81 | 10,495 | 16% |

**The split is a default, not a hard constraint.** It was originally justified
by an estimate that Blender MCP was 404–556 tools and would blow the window on
its own; the server that actually installs is 27 tools and 5% of it. What holds
up is the weaker claim: an agent given 81 tools across two unrelated
applications has a harder selection problem than one given 54 or 27, and the
asset phase genuinely finishes before integration starts. `both` is supported.

The UI still warns when tool schemas pass a third of the window, and every run
record carries its tool budget — the guard is for the configuration someone
reaches for next, such as a smaller context or every Godot toolset switched on.

---

## The time limit

Parametric, with no upper bound, defaulting to **1 hour** (Colibri defaulted to
ten, because a single turn could take three). When the limit is reached the run
is **paused, not discarded**: everything is flushed first, the run is marked
`time_limit_paused`, and the UI asks **Continue or Stop**. Continue grants
another full window and resumes the same harness session, so llama.cpp's prefix
cache and the harness's own context are reused.

Benchmark cells finalise themselves instead of asking, so one slow cell cannot
block a matrix running overnight.

---

## The benchmark

`bench/tasks/` holds fixed, versioned tasks, checked by headless Godot plus
cheap text inspection.

| Task | What it measures |
|---|---|
| **T1 Pong** | build from scratch |
| **T2 Top-down arena** | movement, chasing enemy, health, score, game over |
| **T3 Pause menu** | *editing* existing code — starts from the T2 result |
| **T4 Visual fix** | vision only: fix an overlapping HUD from a screenshot |

T4 is worth calling out: under Colibri it was nearly unrunnable, because the
only engine fast enough to reach it was text-only. The default model here has
vision, so the look-and-fix loop is a real thing to measure rather than a
footnote.

Impossible cells are **skipped with a reason** rather than left blank.

---

## Adding things

**A model.** Add a row to `models.yaml`. **No code changes are needed, ever** —
that is enforced by a test which fails if any backend module branches on a
model family. Give it a repo, a pinned revision, the exact GGUF filename and
byte size (verified against the Hugging Face API, not prose), its capability
flags, its stats for the UI, and — if it needs them — `chat_template_kwargs`
and `sampling`, which the engine passes to llama-server without interpreting.
The test suite also enforces that it fits in fast memory, that a vision claim
comes with an `mmproj`, and that the download is restricted to named files.

**A harness.** Add a row to `harnesses.yaml`, install it in the `Dockerfile` at
a pinned version, write a config generator and an output parser in
`backend/harness.py`, and register the parser in `PARSERS`. Parsers must return
`{"type": "raw", ...}` for anything they cannot read.

**An MCP server.** Add a row to `mcp.yaml` with its command, env mapping, tool
counts and token costs. Get the counts from `scripts/probe-mcp.sh`, which
speaks MCP to the server and prices its catalogue the way a harness actually
sends it. Mark `verified: false` until you have done that — the app surfaces
the difference rather than implying a measurement it does not have. The Blender
entry was originally sized from a blog post about a different server and was
wrong by an order of magnitude; the probe is the correction mechanism.

**Updating pins.** `scripts/update.sh` prints pinned versus latest for every
component. Move a pin in *both* places it appears (the YAML and the matching
`ARG` in the `Dockerfile`), rebuild, and re-run. Numbers from different pins do
not belong in the same table.

---

## Troubleshooting

**The model loads but never calls a tool.** `--jinja` is missing, or the GGUF
has no chat template. The engine adapter refuses the second case at load; for
the first, check the launch line in the Engine log tab.

**`llama-server` exits during load with an unknown-argument error.** llama.cpp's
CLI moves. The adapter checks `--spec-type` before using MTP and fails with a
clear message, but a pin bump can still introduce others — the full argv is
logged at the top of every load.

**Gibberish output.** Check the CUDA version. **13.2 is known-bad with Qwen3.6.**
13.3 (what this machine runs) and anything below 13.2 are fine.

**Out of VRAM partway into a run.** The offload split is sized from *free* VRAM
at load time, so something took the card afterwards — ComfyUI, Blender, a game.
Raise `vram_reserve_mb` in Settings, or pin `n_cpu_moe` higher for that model.

**It is slower than expected.** Check the placement line in the run record: it
says how many expert layers actually landed on the GPU and why. `mode: fallback`
means the GGUF's layer count could not be read and everything went to RAM.

**The UI says Godot is not connected.** The MCP server binds
`ws://127.0.0.1:9080` and the *addon dials out to it*. Start the editor with
`scripts/godot-editor.sh <project>` and enable the plugin.

**Blender tools are missing or the server will not start.** `blender-mcp-server`
is installed best-effort in the image and is marked unverified in `mcp.yaml`.
Confirm the package name and invocation, then set `verified: true`.

---

## Layout

```
app/
├── Dockerfile  compose.yaml  models.yaml  harnesses.yaml  mcp.yaml
├── backend/    FastAPI, model manager, engine lifecycle, harnesses, bench
│   ├── engine.py       llama-server adapter and the offload calculator
│   ├── gguf.py         GGUF header reader (layer counts, chat template)
│   ├── mcp.py          phase-aware server selection and context budgeting
│   └── tests/          state machine, offload math, tool budgets, parsers
├── ui/         plain HTML/CSS/JS, no build step
├── scripts/    preflight, host-setup, up, down, build-llama, download-models,
│               godot-editor, install-godot, bench-games, update
├── bench/tasks/ T1–T4 with their checks and the T4 fixture
├── templates/  minimal Godot 4.7 project, MCP addon, desktop entry
└── BENCHMARKS.md
```

Everything binds to `127.0.0.1`. There is no telemetry. `HF_TOKEN` lives only in
`app/.env`, which is gitignored, and is never printed or logged.

**A safety note that is not boilerplate:** both MCP servers execute
model-generated code — GDScript in the Godot editor, Python inside Blender. The
official Blender server's own documentation says so and advises an isolated
system. Treat any directory the agent can reach as untrusted output.
