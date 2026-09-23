# Host changes

Everything this project changed on the machine, in order. Anything needing
`sudo` is collected in `scripts/host-setup.sh` and marked "pending" until you
run it.

Host: `omarchy`, Arch Linux, kernel 7.0.11-arch1-1.

---

## 2026-09-22 — Colibri decommissioned

RAM_Agent replaces Colibri Local UI. The old tree at
`~/Desktop/Colibri Local UI/` was left in place; only its bulk was removed.

### Deleted: 818 GB of model weights

```
models/deepseek-v41-flash    476 GB
models/deepseek-v4-flash     162 GB
models/glm53-flash           182 GB
```

Checked before deleting: no live process, no open file handles (`lsof` clean),
and a stale `state/watcher.pid` pointing at a dead PID. `rm -rf` returned in
0.064 s because btrfs defers the work; the cleaner thread took a few minutes to
actually release the extents, during which `df` was unchanged. **That is normal
and not a failed delete** — confirm with `btrfs filesystem usage /`, not `df`.

Disk before: 1.1 TB used, 761 GB free.
Disk after: **281 GB used, 1.6 TB free.**

### Deleted: the Colibri container image

`colibri-local-ui:latest` (7.16 GB) and its exited container, plus 14.93 GB of
now-dangling build cache (`docker builder prune`). Reclaimed ~22 GB.

**Left alone deliberately:** the ComfyUI models under `~/Desktop/ComfyUI`
(~47 GB), the `bd2-rag-app` and `mmartial/comfyui-nvidia-docker` images, the
running `comfyui` and `pgvector-db` containers, and the Ollama blob under
`~/Desktop/NAgents`. None of these are Colibri's.

### Kept

`~/Desktop/Colibri Local UI/projects/` (25 MB of benchmark Godot output,
including the G4 "The Keep" run), `state/runs.jsonl` and the engine logs. They
are the control this project is measured against — see `BENCHMARKS.md`.

---

## 2026-09-22 — RAM_Agent tree

Created `~/Desktop/RAM_Agent/` with `models/`, `projects/`, `assets/`,
`state/`, `app/`. Nothing outside this tree was touched.

`app/` is a copy of the Colibri app with the engine layer replaced. It is not
yet a git repo; `.gitignore` at the tree root is written and excludes
`models/`, `state/`, `projects/`, `assets/` and `.env`.

### `models/` is a plain directory this time

Colibri needed it to be a **nodatacow btrfs subvolume** so that `O_DIRECT`
expert streaming worked — on the default `compress=zstd:3` home subvolume it
silently fell back to buffered reads and lost most of its throughput.

**That no longer applies.** llama.cpp reads the file once at load and holds it
resident, so filesystem compression affects load time and nothing else. No
subvolume, no `chattr +C`.

The mount is also **read-only** in `compose.yaml` now. colibri wrote `.coli_kv`
and `.coli_ssd` sidecars next to the weights and needed write access;
llama.cpp writes nothing there, so an agent with shell access cannot corrupt a
27 GB download.

---

## 2026-09-22 — Blender 5.2.2 and the MCP bridge

Installed by `scripts/install-blender.sh`, without root, mirroring how Godot
was installed here:

| | |
|---|---|
| `~/.local/share/blender-5.2.2/` | portable tarball, 366 MB download, **sha256 verified against upstream** |
| `~/.local/bin/blender` | symlink → the above |
| `state/blender-mcp-server/` | djeada/blender-mcp-server @ `7eed33ed`, MIT |
| `state/venv/` | the server installed `-e`, plus its deps |
| `state/blender_mcp_addon.zip` | the add-on, built from the same revision |

**A pin that is not optional.** The server targets MCP SDK v1 and imports
`mcp.server.fastmcp`. pip resolves `mcp` to 2.x, where FastMCP was renamed to
MCPServer, and the server dies on import with a `ModuleNotFoundError`. The venv
pins `mcp<2` (1.30.0 installed) and the Dockerfile does the same. This is why
the server and the add-on are installed from the same git revision rather than
from PyPI: the two halves speak a private protocol to each other.

**Still manual, once.** Blender has no headless add-on install, so the bridge
has to be enabled by hand: Edit → Preferences → Add-ons → Install from Disk →
`state/blender_mcp_addon.zip`, enable "Blender MCP Bridge", then N → MCP in the
viewport. It must read `Listening on 127.0.0.1:9876`.

Nothing was installed system-wide and no package manager was used.

### Probed, not assumed

`scripts/probe-mcp.sh blender` speaks MCP to the server and reports what it
actually exposes: **27 tools in 8 namespaces, 12,733 bytes of schema,
~3,183 tokens.** `mcp.yaml` had previously carried estimates an order of
magnitude larger, taken from the release notes of a different community
server. The registry now carries the probed numbers.

---

## Pending (needs sudo — `scripts/host-setup.sh`)

- `nvidia-ctk runtime configure` for the NVIDIA container runtime, if not
  already done for ComfyUI.
- Nothing else. The `memlock` ulimit is set per-container in `compose.yaml`
  rather than system-wide.

---

## Still true from the Colibri era

- **Hybrid graphics split.** The Godot editor renders on the AMD Renoir iGPU
  (`glx: failed to create dri3 screen` and `failed to load driver: nvidia-drm`
  are expected, not faults) while the NVIDIA card is left for compute. Do the
  same for Blender's viewport — see `docs/vram-budget.md`.
- **Sleep inhibition during runs.** `systemd-inhibit --what=idle:sleep:handle-lid-switch`
  plus pausing `hypridle`, which does not consult logind inhibitors for its own
  timers. Per-run, released on exit including after a crash.
- **`~/.local/bin/godot`** — Godot 4.7.2, installed by `scripts/install-godot.sh`.
- **`~/.local/bin/blender`** — Blender 5.2.2, installed by `scripts/install-blender.sh`.

## No longer true

- The 10-hour default time limit (now 1 hour).
- `COLI_QUEUE_TIMEOUT`, `DIRECT=1`, and the rest of the disk-streaming tuning.
- `models/` as a nodatacow subvolume.
