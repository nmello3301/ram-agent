# RAM_Agent

A local lab for measuring how fast a given **model × harness** pair can build a
3D game and its assets. Pick a model and a harness in a browser page, type a
prompt, and an agent drives **Godot** and **Blender** over MCP. Everything runs
on one laptop; nothing leaves it.

> **Status: nothing has been benchmarked yet.** The engine, the harnesses and
> both MCP bridges are wired and tested, and the first measurements are
> specified in [`app/BENCHMARKS.md`](app/BENCHMARKS.md). Performance figures in
> the docs are clearly marked as extrapolations until this machine produces its
> own.

## Why it exists

It replaces [colibri-local-ui](https://github.com/nmello3301/colibri-local-ui),
which asked the same question with 167–510 GB models streamed off an SSD. That
produced real numbers and they were damning:

| | Colibri, measured | RAM_Agent, target |
|---|---|---|
| Decode | 0.10–0.99 tok/s | ~15–21 tok/s |
| Prefill | < 2 tok/s | ~100–250 tok/s |
| Godot tool catalogue | ~57 min of prefill | ~45 s |
| 6 h autonomous run | 2 turns, nothing built | — |
| Weights on disk | 818 GB | 27.5 GB |

The bottleneck was never the engine or the prompt. It was that a 510 GB model
cannot be read faster than the drive reads it. **One decision changes
everything here: the model fits in memory.**

The reasoning, the alternatives rejected, and the corrections made along the way
are in **[RAM_AGENT.md](RAM_AGENT.md)**.

## The stack

| Layer | Choice |
|---|---|
| Engine | `llama-server` (llama.cpp, CUDA) with `--n-cpu-moe` |
| Default model | Qwen3.6-35B-A3B, `UD-Q5_K_XL`, 27.5 GB, vision, Apache-2.0 |
| Harness | OpenCode (Goose and Pi kept as comparisons) |
| Tools | Godot MCP (54 tools) + Blender MCP (27 tools) |

**The budget is RAM + VRAM.** 62 GB + 6 GB = 68 GB of fast memory, ~55 GB of it
usable for a model. Weights placed on the GPU come out of the VRAM column and
the rest sit in RAM, so a model is sized against the sum. The disk is not in the
budget — that is the entire point.

**The model layer is model-agnostic.** No backend module knows any model's name;
family, prompt conventions, sampling and chat-template arguments are all
registry data in `app/models.yaml`. A test fails the build if that stops being
true.

## Quick start

```bash
cd app

bash scripts/host-setup.sh        # once; the only step needing sudo
bash scripts/preflight.sh         # measures the machine
bash scripts/install-godot.sh     # Godot 4.7.2 + MCP addon, no root
bash scripts/install-blender.sh   # Blender 5.2.2 + MCP bridge, no root
bash scripts/download-models.sh   # 28 GB (--all adds the sweep rows)
bash scripts/up.sh                # builds the image, opens the UI
```

Both editors need their bridge enabled once, by hand — each addon *listens* and
the MCP server dials out to it:

- **Godot** — Project → Project Settings → Plugins → Godot MCP → Enable
- **Blender** — Edit → Preferences → Add-ons → Install from Disk →
  `state/blender_mcp_addon.zip`, enable *Blender MCP Bridge*, then `N` → MCP.
  It must read `Listening on 127.0.0.1:9876`.

The UI is at <http://127.0.0.1:8088>. Full operating instructions, the model
catalogue, the phase split and troubleshooting are in
**[`app/README.md`](app/README.md)**.

## Layout

```
RAM_AGENT.md        why this exists, and what was measured vs assumed
app/                the application
├── README.md       operating instructions
├── BENCHMARKS.md   the measurement plan (nothing filled in yet)
├── models.yaml     the model registry — the only place a model is described
├── mcp.yaml        MCP servers, with probed tool counts and token costs
├── backend/        FastAPI, engine lifecycle, harnesses, MCP selection
├── ui/             plain HTML/CSS/JS, no build step
├── scripts/        install, build, download, benchmark
└── docs/           vram-budget.md
models/  projects/  assets/  state/     runtime data, all gitignored
```

## A word about what this runs

Both MCP servers execute model-generated code — GDScript in the Godot editor,
Python inside Blender, with no sandbox. The Blender server's own documentation
says as much and advises an isolated machine. Treat any directory the agent can
reach as untrusted output.

Nothing binds beyond `127.0.0.1`. There is no telemetry. A Hugging Face token,
if you set one, lives only in `app/.env`, which is gitignored and never logged.

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT),
[Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0),
[OpenCode](https://github.com/anomalyco/opencode),
[Goose](https://github.com/block/goose),
[Pi](https://pi.dev),
[godot-mcp](https://github.com/hybridindie/godot-mcp) and
[blender-mcp-server](https://github.com/djeada/blender-mcp-server) (MIT).
