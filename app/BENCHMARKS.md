# Benchmarks

**Nothing in this file has been measured on RAM_Agent yet.**

That is deliberate and it is the first thing to fix. The estimates in
`../RAM_AGENT.md` are extrapolations from community benchmarks on other
hardware, and they do not belong here until this machine produces them.

Colibri's own history is the argument for that discipline: its first GPU
write-up claimed a 2.84× speedup against a baseline contaminated by a
concurrent model download, and the honest figure turned out to be 1.66×. A
number in this file should mean "this machine, this pin, this configuration."

---

## The machine

| | |
|---|---|
| CPU | AMD Ryzen 7 4800H, 8 cores / 16 threads |
| ISA | AVX2, FMA, F16C, BMI2 — **no AVX-512** |
| RAM | 62 GB + 62 GB swapfile + 4 GB zram |
| **Fast memory** | **68 GB (62 RAM + 6 VRAM), ~55 GB usable for a model** |
| RAM speed | **unknown — run `sudo dmidecode -t memory \| grep -i speed`** |
| GPU | RTX 3060 Laptop, **6 GB**, compute capability 8.6, driver 610.43.02, CUDA UMD 13.3 |
| iGPU | AMD Renoir (Vega) — drives the desktop and the Godot editor |
| Storage | Kingston SNV2S2000G, 2 TB NVMe, DRAM-less QLC |
| Filesystem | btrfs on LUKS, `compress=zstd:3` |
| OS | Arch (Omarchy), kernel 7.0.11 |

The storage line no longer matters, which is the entire point of this project.
It is read once at load and never touched again. The RAM-speed line matters
enormously and is currently **unmeasured** — the 4800H supports DDR4-3200
dual-channel, and whether these sticks are 3200 or 2666 moves every decode
estimate by roughly 20%.

## Pins

| Component | Version |
|---|---|
| llama.cpp | b11115 |
| Godot | 4.7.2-stable |
| godot-editor-mcp | 2026.9.10 |
| Blender | 5.2.2 LTS |
| blender-mcp-server | 0.1.3 (djeada, MIT, rev 7eed33ed) |
| OpenCode | 1.18.31 |
| Goose | 1.50.1 |
| Pi | 0.85.1 (+ pi-mcp-adapter) |

---

## To measure, in this order

Each row is a decision the project currently makes on an estimate.

### 1. The baseline

`llama-bench` on the default model, then Direct chat for an end-to-end figure.

| | target | actual |
|---|---|---|
| Prefill (pp512) | 100–250 tok/s | _pending_ |
| Decode (tg128) | 15–21 tok/s | _pending_ |
| Load time, `--mlock` 27 GB | < 2 min | _pending_ |
| Resident RSS | ~28 GB | _pending_ |

### 2. The `-ncmoe` sweep

The offload calculator will almost certainly resolve to "all experts in RAM" on
a 6 GB card — that is the GTX-1650-class configuration that measured 20.2 tok/s
elsewhere. The question is whether anything can be clawed back.

| `n_cpu_moe` | VRAM used | prefill | decode | stable under load? |
|---|---|---|---|---|
| auto (resolved: ?) | | _pending_ | _pending_ | |
| all-on-CPU | | _pending_ | _pending_ | |
| −4 | | _pending_ | _pending_ | |
| −8 | | _pending_ | _pending_ | |

> Test **under load**, not empty. The 12 GB reference rig was faster at
> `-ncmoe 20` and only *stable* at 24; the faster setting OOM'd once context
> filled. An empty-context number is not a result.

### 3. The quant sweep

Q4_K_XL / Q5_K_XL / Q8_0 over T1–T3.

| Quant | Size | decode | T1 | T2 | T3 | **tool-call validity** |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 22.4 GB | _pending_ | | | | |
| UD-Q5_K_XL | 26.6 GB | _pending_ | | | | |
| Q8_0 | 36.9 GB | _pending_ | | | | |

> Score tool-call validity, not just pass/fail. Quantisation damage in agent
> work shows up as malformed tool arguments long before it shows up as worse
> prose, and a pass/fail column hides it entirely.

### 4. MTP

| `--spec-draft-n-max` | decode | vs. non-MTP |
|---|---|---|
| off | _pending_ | — |
| 1 … 6 | _pending_ | |

> Upstream measures 1.5–2× on GPUs. Expect less here: MoE routing differs per
> token, so verifying a draft can touch more experts than a single token would.
> Do not assume 2 is optimal.

### 5. Thinking vs non-thinking

Reasoning tokens are pure decode cost inside a tool loop. Measure whether they
pay for themselves.

| mode | tokens/turn | turns to finish T1 | wall |
|---|---|---|---|
| thinking, `preserve_thinking` | _pending_ | | |
| thinking, no preserve | _pending_ | | |
| `enable_thinking: false` | _pending_ | | |

### 6. Tool budget

| config | tools | tokens | turns to finish | notes |
|---|---|---|---|---|
| godot default | 54 | 7,312 | _pending_ | measured |
| godot, MCP off | 0 | 0 | _pending_ | agent writes .tscn/.gd as text |
| blender, whole catalogue | 27 | 3,183 | _pending_ | measured |
| both | 81 | 10,495 | _pending_ | measured |

> Token costs are now **measured**, via `scripts/probe-mcp.sh`. An earlier
> version of this table carried estimates for Blender that were wrong by an
> order of magnitude, because they were sized from the release notes of a
> different and much larger community server.

> The MCP-off row is not a formality. Colibri's only successful run — 24 h,
> a working 3D game — had MCP **off**, because halving the preamble doubled the
> turn rate. That trade is no longer forced, but "does the live editor bridge
> actually beat writing files and validating with headless Godot" is a real
> open question, and it is cheap to answer now.

---

## Prior art: what Colibri measured

Kept because it is the control this project is judged against, and because the
failures are more instructive than the successes. Different engine, different
models, **not comparable as numbers** — only as a baseline for the shape of the
problem.

| | Colibri, measured |
|---|---|
| Decode, DeepSeek V4 Flash + CUDA (best case) | **0.993 tok/s** |
| Decode, DeepSeek V4 Flash, CPU only | 0.600 tok/s |
| Decode, GLM-5.3-Flash | **~0.1 tok/s** |
| Prefill | **< 2 tok/s** |
| Godot tool catalogue (6,804 tok) | **~57 min per cold turn** |
| Warm/cold TTFT ratio (prefix cache) | 2.8× |
| Models on disk | **818 GB** |

Three results worth carrying forward:

**G1–G3 — 6 h each, MCP on, zero tool calls, nothing built.** The first turn
took 10,778 s against a pre-run prediction of 10,769 s. The model was never the
wrong shape for the task; it was two orders of magnitude too slow for it. Six
hours bought two turns.

**G4 — 24 h, MCP off, a working 3D game.** 12 completions, 80 lines of GDScript
and a 172-line scene; imports clean, runs 300 frames clean. One real defect:
pitch applied to `$Collision` instead of `$Camera`, so looking up and down did
nothing. The model never caught it **because with MCP off it had no way to see
the screen.** That is the single strongest argument for the vision-capable
default in this project.

**Three runs died at ~31 minutes before anyone found the timeout.**
OpenCode's `headerTimeout` and `chunkTimeout` both default to 300 s and abort a
request that goes quiet — which was every request. The fix is still in
`harness.py`, and there is still a test asserting it, because the failure mode
cost three 6 h runs to diagnose.

### What Colibri predicted for this project

Its own analysis of what would change the outcome:

| change | per-turn floor |
|---|---|
| GLM + 7,000-token catalogue (what it ran) | ~3 h |
| DeepSeek V4 Flash + ~800-token surface | ~2–3 min |

RAM_Agent's premise is that a resident 3B-active model gets below that floor
with the *full* 7,000-token catalogue rather than a hand-built 800-token one.
**That is a prediction, and this file is where it gets checked.**
