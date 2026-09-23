# The VRAM budget

Colibri's hard constraint was RAM: it sized its own expert cache from what was
free, took 53 GB of 62, filled zram, and started failing requests. The fix was
to cap it (`--ram`), and `docs/long-run-requirements.md` existed to explain the
timeout discipline that disk-streaming forced on everything else.

**Both problems are gone.** The RAM bill is now simply the size of the GGUF, and
a turn takes seconds rather than hours. That document has been deleted rather
than updated, because a stale requirements doc is worse than none.

What replaced it is a budget with two columns and one hard exclusion:

    fast memory = RAM (62 GB) + VRAM (6 GB) = 68 GB,  ~55 GB usable
    disk        = not in the budget at all

A model may use both kinds of fast memory freely — weights placed on the GPU
come out of the VRAM column and the rest sit in RAM. What it may not do is
touch the disk, because that is Colibri.

Within that budget, the scarce column is **6 GB of VRAM, wanted by four things
at once.**

## Who wants the card

| | typical |
|---|---|
| llama.cpp attention + shared trunk | 2–3 GB |
| KV cache, `q8_0`, 64k context | 1–1.5 GB |
| `mmproj` vision projector | ~0.9 GB |
| CUDA context and scratch | ~0.5 GB |
| **Blender Cycles** | all of it |
| ComfyUI, if running | observed holding 110 MiB |

Add the first four and a 6 GB card is full before a single expert layer is
considered. That is the expected outcome, it is not a failure, and
`auto_n_cpu_moe()` resolving to "all experts in system RAM" is the correct
answer — it is the configuration a GTX 1650 measured 20.2 tok/s with.

## How the split is decided

`backend/engine.py::auto_n_cpu_moe()`:

```
budget      = free_vram − reserve − kv_cache − mmproj − non_expert_trunk
layers_gpu  = budget ÷ per_layer_expert_bytes
n_cpu_moe   = total_layers − layers_gpu
```

Three deliberate choices in that arithmetic:

1. **`free` VRAM, not `total`.** Colibri's log filled with
   `weight allocation: out of memory` because a ComfyUI instance held 110 MiB
   it had not accounted for.
2. **The KV estimate errs high.** Over-estimating costs a little throughput;
   under-estimating costs an OOM four hours into a run.
3. **Everything that does not fit goes to RAM.** RAM is merely slower. VRAM
   exhaustion is a crash.

The resolved split, and the reason for it, is written into every run record as
`placement` — because "why was this run slow" is otherwise unanswerable after
the fact.

## The headless escape hatch

Found by installing the thing rather than reading about it: the Blender MCP
server's `blender_render_still`, `blender_render_animation` and
`blender_python_exec` take a `transport` argument.

- `bridge` (the default) drives the live GUI session through the add-on.
- `headless` spawns a separate `blender -b` process against a `.blend` file.

The headless path never touches the interactive session, so it is the natural
way to render while llama.cpp holds the card. It is also how to run a heavy
script without risking the add-on's process.

## The operational rule

**You cannot GPU-render in Blender while the model is loaded.** Options, in
order of preference:

1. Blender viewport on EEVEE against the AMD Renoir iGPU; the NVIDIA card stays
   with llama.cpp. This is already how the Godot editor runs here — its
   `glx: failed to create dri3 screen` messages are expected, not a fault.
2. Cycles on CPU. Slow, but it does not contend.
3. Unload the model, render, reload. Reloading 27 GB with `--mlock` is under a
   couple of minutes, not the hours a Colibri load took.

If VRAM is taken *after* the model loads, the split was sized against numbers
that are no longer true. Raise `vram_reserve_mb` in Settings, or pin
`n_cpu_moe` for that model.
