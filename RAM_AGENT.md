# RAM_Agent

The successor to Colibri Local UI. Same question — how fast can a
**model × harness** pair build a game — with the one variable that decided
every previous answer removed: **the weights now live in RAM.**

---

## Why this project exists

Colibri Local UI measured three models of 167 GB, 195 GB and 510 GB, streamed
expert-by-expert off a DRAM-less QLC NVMe. The numbers it produced were real
and they were damning:

| | Colibri, measured |
|---|---|
| Decode, best case (DeepSeek V4 Flash + CUDA) | **0.993 tok/s** |
| Decode, GLM-5.3-Flash | **~0.1 tok/s** |
| Prefill | **< 2 tok/s** |
| Godot tool catalogue (6,804 tok) | **~57 min of prefill, every cold turn** |
| G1–G3: 6 h each, MCP on | **2 turns, zero tool calls, nothing built** |
| G4: 24 h, MCP off | 12 turns, 80 lines of GDScript, one real bug |

The bottleneck was never the engine, the harness or the prompt. It was that a
510 GB model cannot be read faster than the drive reads it. Colibri is a good
answer to "how do I run a model far larger than my RAM." It is the wrong
question for this machine.

**The right question:** what is the best agent I can build where the model
fits in 62 GB of RAM and never touches the disk after load?

Answer: roughly **20–100× faster**, with vision, for a 27 GB download.

---

## The machine

Unchanged from Colibri, and it is the whole constraint:

| | |
|---|---|
| CPU | AMD Ryzen 7 4800H, 8C/16T, **AVX2 — no AVX-512** |
| RAM | 62 GB usable (+62 GB swapfile, 4 GB zram) |
| GPU | RTX 3060 **Laptop, 6 GB**, sm_86, driver 610.43.02, CUDA UMD 13.3 |
| iGPU | AMD Renoir (Vega) — drives the desktop and the Godot editor |
| Storage | Kingston SNV2S2000G, 2 TB, DRAM-less QLC |
| OS | Arch (Omarchy), kernel 7.0.11 |

Two facts drive every decision below:

1. **RAM bandwidth is the decode ceiling.** Dual-channel DDR4 delivers roughly
   27–35 GB/s in practice. For an MoE, decode speed ≈ bandwidth ÷ *active*
   bytes per token.
2. **The capacity budget is RAM + VRAM.** 62 GB + 6 GB = **68 GB of fast
   memory**, of which ~55 GB is usable once headroom for the OS, the editors
   and the CUDA context is deducted. Weights placed on the GPU come out of the
   VRAM column and the rest sit in RAM, so a model is sized against the sum.
   What is *not* in the budget is the disk: the moment weights page off the
   NVMe, this is Colibri again.
3. **6 GB of VRAM is now contested.** Under Colibri the 3060 was the engine's
   alone. It no longer is — see [The VRAM problem](#the-vram-problem).

---

## The stack

| Layer | Choice |
|---|---|
| Inference engine | **llama.cpp `llama-server`**, CUDA build, `--n-cpu-moe` |
| Model | **Qwen3.6-35B-A3B** — 35B total / **3B active**, Apache 2.0, vision |
| Quant | **`UD-Q5_K_XL`, 26.6 GB** (default) |
| Harness | **OpenCode** (Goose and Pi retained as comparison rows) |
| Tools | Godot MCP + Blender MCP, **gated and phase-split** |

### Why llama.cpp

The core flag is `--n-cpu-moe N` (alias `-ncmoe`): keep attention and the
shared/dense trunk on the GPU, leave the routed expert FFNs resident in system
RAM. For a 3B-active MoE that means ~1.5 GB read per token from RAM instead of
Colibri's 4.8 GB per token from NVMe.

Alternatives considered:

| | Verdict |
|---|---|
| **Ollama** | Wraps llama.cpp, but its default templates are documented to break Qwen tool calls — the model emits raw `<tools>` markers as text. No `-ot`/`-ncmoe` control. **Rejected.** |
| **LM Studio** | Same engine, GUI-first, poor fit for a scripted benchmark rig. **Rejected.** |
| **ik_llama.cpp** | The CPU-tuned fork. An April 2026 regression report has it ~2× slower on prefill and ~1.5× on decode for *exactly* our case (Qwen3.6 MoE, CPU-MoE, AVX2 Ryzen). **Revisit later**, do not build on it. |
| **KTransformers** | Purpose-built CPU/GPU MoE hybrid; 0.5.3 (Mar 2026) added AVX2-only kernels so the 4800H finally qualifies. Tuned for AMX/AVX-512 and large VRAM, and much heavier to operate. **Phase 2 experiment.** |
| **vLLM / SGLang** | GPU-resident. 6 GB. **No.** |

### Why Qwen3.6-35B-A3B

Released April 2026, Apache 2.0.

| | |
|---|---|
| SWE-bench Verified | **73.4** |
| SWE-bench Pro | 49.5 |
| **Terminal-Bench 2.0** | **51.5** — the agentic-loop number, and the one that matters here |
| Total / active | 35B / **3B** |
| Context | 262K native (→1M with YaRN) |
| Architecture | Hybrid Gated DeltaNet + Gated Attention |
| **Vision** | **Yes** — unified vision-language foundation |

The vision encoder is the unlock. Colibri's only fast engine (DeepSeek V4
Flash) was text-only; its two vision engines ran at 0.1–1 tok/s. A
RAM-resident VLM can actually look at a Godot viewport or a Blender render and
fix what it sees. T4 becomes a real task, and 3D asset work becomes possible
at all.

**What does not fit, and why it is not a near miss:** GLM-5.3-Flash (320B,
105–115 GB at *2-bit*), MiniMax M3 (428B), Hy4 (770B), Kimi K3 (1.6 TB).
Qwen3.8-Flash-Next is the interesting one — 125B + a 51B n-gram table, ~46 GB
of RAM at 3.84bpw *because the n-gram table stays paged on SSD via mmap*. That
reintroduces the exact dependency this project exists to escape, on a
DRAM-less QLC drive, at 6B active. Deliberately excluded.

### Why this quant

Decode is bandwidth-bound, so **bytes per token ≈ quant size × (3/35)**.
Halving the quant roughly doubles decode speed. 62 GB of RAM is *not* a reason
to run Q8 — spare RAM should go to KV cache, Blender scenes and the Godot
editor, not to precision that cannot be measured.

| Quant | File | Est. decode | Fast memory left (of ~55 GB usable) |
|---|---|---|---|
| `UD-Q4_K_XL` | 22.36 GB | ~18–26 tok/s | ~32 GB |
| **`UD-Q5_K_XL`** | **26.59 GB** | **~15–21 tok/s** | **~28 GB** |
| `UD-Q6_K_XL` | 31.84 GB | ~13–17 tok/s | ~23 GB |
| `Q8_0` | 36.90 GB | ~10–14 tok/s | ~18 GB |

Fitting is necessary but not sufficient. Every row here fits; the reason Q5 is
the default is bandwidth, not capacity. The registry carries Q4_K_XL and Q8_0
as rows so the sweep is one click, not an edit.

**MTP.** `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` ships multi-token-prediction
heads — self-speculative decoding, claimed 1.5–2× at no accuracy loss. This is
precisely the right shape of win when bandwidth-bound: several drafted tokens
verified per weight read. Costs ~0.6 GB over the plain build. Enabled with
`--spec-type draft-mtp --spec-draft-n-max N`; sweep N over 1–6, do not assume 2.

Caveat worth stating: MoE routing differs per token, so verifying a draft can
touch more experts than a single token would. The GPU-measured 1.5–2× is an
upper bound here. **Measure it; do not assume it.**

### Why OpenCode — and why the old reasoning inverts

Colibri's README argued that Pi mattered because its preamble is under 1,000
tokens against 10–20k for an MCP harness. **That stops being true.**

| | Colibri | RAM_Agent |
|---|---|---|
| Godot catalogue (6,804 tok) | ~57 min | **~30–60 s**, cached after |
| G1–G3 preamble (15,074 tok) | ~2 h | **~1–2 min** |

Once prefill runs at 100–400 tok/s, preamble size stops being the design
constraint and the quality of the agent loop takes over. That is OpenCode:
native MCP, provider-agnostic OpenAI-compatible provider, already wired into
this codebase. Goose stays as the big-preamble comparison row. Pi stays as the
control, but its entire reason for being in the lineup is gone.

**Non-negotiable:** `llama-server --jinja`. Without it the model's own chat
template is bypassed and tool calling silently degrades into text.

---

## Tool schemas: a correction

This section originally argued that tool catalogues were the new scarce
resource, on the strength of Blender MCP shipping 404 tools (v3) or 556 (v4)
with profiles and progressive disclosure. **That was wrong, and the way it was
wrong is worth recording.**

Those numbers came from a community server's release notes. The server that
`pip install blender-mcp-server` actually installs — djeada/blender-mcp-server,
MIT, the one now on this machine — is **27 tools in 8 namespaces**. Probed from
the running server with `scripts/probe-mcp.sh`:

| | tools | schema bytes | tokens | of 64k |
|---|---|---|---|---|
| Godot MCP (default 3 toolsets) | 54 | 28,145 | **7,312** | 11% |
| **Blender MCP (entire catalogue)** | **27** | **12,733** | **3,183** | **5%** |
| both | 81 | 40,878 | 10,495 | 16% |

So the two servers together cost 16% of the window, not the 36% estimated. The
profile machinery in `mcp.yaml` was fiction and has been deleted rather than
left to mislead; Blender MCP offers no gating and needs none.

The **phase split survives**, demoted from a hard constraint to a default. The
justification is no longer token cost, it is that an agent given 81 tools
across two unrelated applications has a harder selection problem than one given
54 or 27 — and the asset phase genuinely completes before the integration phase
starts.

```
  asset phase            handoff              integration phase
  ───────────            ───────              ─────────────────
  Blender MCP    ──▶   .glb / .gltf   ──▶     Godot MCP
   (27 tools)          in res://assets         (54 tools)
```

`both` is now a supported configuration rather than a trap.

Godot MCP remains genuinely large — 181 tools across 27 gated toolsets — so
selecting down there is real work. The two servers are asymmetric, which is why
`backend/mcp.py` has a real builder for one and a pass-through for the other.

## The VRAM problem

Under Colibri the 3060 was the engine's alone, and the Godot editor was
already pushed onto the AMD Renoir iGPU. That arrangement no longer holds,
because **four things now want 6 GB**:

1. llama.cpp attention + shared trunk + KV cache — ~4–5.5 GB
2. the `mmproj` vision encoder — ~0.9 GB
3. **Blender Cycles**
4. the ComfyUI instance already on this machine (observed holding 110 MiB)

**You cannot GPU-render in Blender while the model is loaded.** Plan for it:
Blender viewport on EEVEE against the iGPU, Cycles on CPU, or unload the model
to render. This is the single biggest operational constraint in the project
and it has no clever fix — it is a 6 GB card.

One mitigation that fell out of actually installing the thing: the Blender MCP
server's render and python-exec tools take a `transport` argument, and
`transport="headless"` spawns a separate `blender -b` process against a .blend
file instead of driving the live GUI session. That path does not need the
interactive Blender at all, which makes it the natural way to render while the
model holds the card.

Two smaller ones:

- **Do not "fix" CUDA by downgrading.** The driver here reports CUDA 13.3,
  which is correct. **CUDA 13.2 produces gibberish output** with Qwen3.6.
- **Consider `--no-mmap --mlock`** so a ballooning Blender scene cannot evict
  model pages out of page cache and quietly return you to disk-streaming.

---

## What this should buy

Extrapolated from two published measurements on comparable rigs:

| Reference rig | Config | Prefill | Decode |
|---|---|---|---|
| GTX 1650 **4 GB** + 32 GB DDR4-2133 | `-ncmoe 40` (all experts in RAM) | 66 tok/s | 20.2 tok/s |
| RTX 3060 **12 GB** + 32 GB DDR4-2133 | `-ncmoe 24` (16/40 expert layers on GPU) | **413 tok/s** | **38.9 tok/s** |

This machine's 6 GB card sits between them, with a materially stronger CPU
(8C/16T vs 4C/8T) and probably faster RAM:

| | Colibri, measured | RAM_Agent, estimated |
|---|---|---|
| Decode | 0.10–0.99 tok/s | **15–21 tok/s** |
| Prefill | < 2 tok/s | **100–250 tok/s** |
| Godot tool catalogue | ~57 min | **~45 s** |
| Pong-sized output (~2,000 tok) | 25–42 min | **~2 min** |
| 30-turn agent loop | days | **under an hour** |
| Vision | unusable | **works** |
| Model on disk | 818 GB | **27.5 GB** |

> **These are extrapolations, not measurements.** The anchors are community
> benchmarks on different hardware and the 6 GB laptop case is interpolated
> between them. Colibri's own history is the argument for saying so plainly:
> its first GPU write-up claimed 2.84× against a contaminated baseline and the
> honest figure turned out to be 1.66×. Nothing in this table belongs in
> `BENCHMARKS.md` until this machine produces it.

---

## What carries over unchanged

The Colibri app was built around a clean seam: `EngineManager.start()` launches
*something* that serves an OpenAI-compatible API on `127.0.0.1:8077`, and every
layer downstream talks to that URL. Swapping the engine is a new adapter, not a
rewrite.

| Reused as-is | Rewritten |
|---|---|
| FastAPI backend, SSE, run history | `backend/engine.py` — llama-server adapter |
| UI (plain HTML/CSS/JS) | `models.yaml` — GGUF registry |
| Harness layer + output parsers | `backend/models_mgr.py` — per-file GGUF downloads |
| `backend/godot.py`, headless validation | `Dockerfile` — builds llama.cpp, not colibri |
| `bench/tasks/` T1–T4, G1–G4 | `scripts/download-models.sh` |
| Time-limit state machine | **new** `backend/blender.py` |
| Sleep inhibition | |

Three pieces of hard-won Colibri machinery are now **obsolete**, and deleting
them is part of the point:

- the 10-hour default time limit,
- the `COLI_QUEUE_TIMEOUT` / `headerTimeout` / `chunkTimeout` timeout war,
- the `nodatacow` subvolume and `O_DIRECT` tuning for expert streaming.

All three existed to survive disk-bound inference. The time limit stays
parametric because a long autonomous run is still a long run — but it defaults
to **1 hour**, not ten.

---

## First measurements

The rig already exists. Let the rig decide, and fill in `BENCHMARKS.md` from
this machine only.

1. **Quant sweep** — Q4_K_XL / Q5_K_XL / Q8_0 over T1–T3, scored on
   *tool-call validity*, not only pass/fail.
2. **`-ncmoe` sweep** — highest expert-layer count that fits 6 GB alongside KV
   without OOM *under load*. The 12 GB reference rig was faster at 20 but only
   stable at 24; expect the same shape here.
3. **MTP on/off**, `--spec-draft-n-max` 1–6.
4. **Thinking vs non-thinking** — reasoning tokens are pure decode cost inside
   a tool loop. Measure whether they pay for themselves.
5. **Tool-budget curve** — Blender's 22-tool default plus progressive
   disclosure vs `core` (131), on a real asset task.

One input still unknown: actual DIMM speed. `sudo dmidecode -t memory | grep -i speed`.
The 4800H supports DDR4-3200 dual-channel; if these sticks are 3200 rather
than 2666, every estimate above shifts up.

---

## Sources

Model and quant data verified against the Hugging Face API on 2026-09-22;
filenames and byte sizes in `models.yaml` come from that query, not from prose.

- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) · [MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) · [Unsloth run guide](https://unsloth.ai/docs/models/qwen3.6)
- [Qwen3-Coder-Next](https://unsloth.ai/docs/models/qwen3-coder-next)
- [Low-VRAM sweeps for 35B-A3B](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/) — the two reference rigs
- [CPU+GPU MoE offload guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)
- [llama.cpp function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) · [llama-swap](https://github.com/mostlygeek/llama-swap)
- [ik_llama.cpp regression #1699](https://github.com/ikawrakow/ik_llama.cpp/issues/1699) · [KTransformers AVX2](https://www.phoronix.com/news/KTransformers-0.5.3)
- [Blender MCP Server v3](https://www.strayspark.studio/blog/blender-mcp-server-v3-blender-5x-rewrite-404-tools) · [Official Blender MCP](https://www.blender.org/lab/mcp-server/)
- [OpenCode + llama.cpp config](https://njannasch.dev/snippets/opencode-llamacpp-config/) · [Qwen3.6 on 24GB: every mistake](https://aminrj.com/posts/llamacpp-qwen36-35b/)
