"""llama.cpp engine lifecycle.

One `llama-server` process at a time, serving one model over an
OpenAI-compatible API on 127.0.0.1:8077.

This replaces Colibri's `coli serve` adapter and keeps the same contract, which
is the whole reason the rest of the app carries over untouched: start a process,
block until it serves *our* model id, stream its log, stop it cleanly.

What changed underneath is the premise. colibri streamed experts off the SSD and
had to be told how much RAM it was allowed to cache in. llama.cpp holds the
whole quantised model resident and the only real question is how it is split
between the two kinds of fast memory -- which is what `--n-cpu-moe` decides and
what `auto_n_cpu_moe()` below sizes.

THE BUDGET IS RAM + VRAM. 62 GB and 6 GB here, 68 GB of fast memory. Weights on
the GPU come out of the VRAM column and the rest sit in RAM; a model is sized
against the sum, not against either alone. What is NOT in the budget is the
disk: the moment weights page off an NVMe, this is Colibri again.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import MODELS_DIR, STATE_DIR
from .gguf import GGUFError, GGUFInfo, read_info
from .models_mgr import ModelEntry

SERVE_PORT = int(os.environ.get("LLAMA_PORT", "8077"))
SERVE_HOST = "127.0.0.1"

# Where the CUDA build lands. The Dockerfile builds llama.cpp at a pinned ref;
# a host install is picked up from PATH so the app can run natively too.
LLAMA_SERVER = os.environ.get("LLAMA_SERVER_BIN") or shutil.which("llama-server") \
    or "/opt/llama.cpp/bin/llama-server"

# Fraction of an MoE checkpoint that is routed-expert weight. Used only to size
# `--n-cpu-moe` when it is set to `auto`; an explicit integer bypasses it.
# 0.90 is the right ballpark for an A3B-class model (3B of 35B parameters are
# outside the experts). It is a registry default so it can be corrected without
# touching code.
DEFAULT_EXPERT_FRACTION = 0.90


@dataclass
class EngineStatus:
    model_id: str | None = None
    pid: int | None = None
    healthy: bool = False
    loading: bool = False
    message: str = ""
    started_at: float | None = None
    # What the offload decision actually resolved to, so the UI and the run
    # record show the real configuration rather than the requested one.
    placement: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "pid": self.pid,
            "healthy": self.healthy,
            "loading": self.loading,
            "message": self.message,
            "uptime": round(time.time() - self.started_at, 1) if self.started_at else None,
            "base_url": base_url(),
            "placement": self.placement,
        }


def base_url() -> str:
    return f"http://{SERVE_HOST}:{SERVE_PORT}"


class EngineError(RuntimeError):
    pass


# -- hardware introspection -------------------------------------------------

def total_ram_gb() -> int:
    """Physical RAM in GB, or 0 when it cannot be determined."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(int(line.split()[1]) / 1_048_576)
    except OSError:
        pass
    return 0


def available_ram_gb() -> int:
    """RAM actually available right now, not merely unused.

    MemAvailable accounts for reclaimable page cache, which is the number that
    matters when deciding whether a 46 GB model will fit alongside a browser.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(int(line.split()[1]) / 1_048_576)
    except OSError:
        pass
    return total_ram_gb()


def fast_memory(ram_headroom_gb: int = 12, vram_reserve_mb: int = 1024
                ) -> dict[str, Any]:
    """RAM and VRAM together, because the model may use both.

    This is the budget a model is sized against. Weights placed on the GPU come
    out of the VRAM column and the rest sit in RAM, so the capacity that
    matters is the sum -- 62 GB + 6 GB here -- not either one alone.

    `usable_gb` subtracts what must be left for everything that is not the
    model: the OS, the container, Firefox, the Godot editor, and on the GPU
    side the CUDA context, activations and whatever else has the card.
    """
    ram_total = total_ram_gb()
    ram_avail = available_ram_gb()
    vram_total = total_vram_mb()
    vram_free = free_vram_mb()

    usable_ram = max(0, ram_total - ram_headroom_gb)
    usable_vram = max(0, (vram_free - vram_reserve_mb) / 1024)
    return {
        "ram_total_gb": ram_total,
        "ram_available_gb": ram_avail,
        "vram_total_mb": vram_total,
        "vram_free_mb": vram_free,
        "ram_headroom_gb": ram_headroom_gb,
        "vram_reserve_mb": vram_reserve_mb,
        "fast_total_gb": round(ram_total + vram_total / 1024, 1),
        "usable_gb": round(usable_ram + usable_vram, 1),
    }


def total_vram_mb() -> int:
    """Total VRAM on GPU 0, or 0 when there is no usable NVIDIA card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return 0


def free_vram_mb() -> int:
    """Free VRAM on GPU 0, or 0 when there is no usable NVIDIA card.

    Deliberately reads *free* rather than *total*. On this machine the card is
    shared: a ComfyUI instance sitting on it was enough to make colibri's tail
    allocations fail, and the same trap applies here.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return 0


def _kv_cache_mb(info: GGUFInfo, ctx: int, quantised: bool) -> int:
    """Rough KV-cache footprint. Deliberately an over-estimate.

    There is no cheap exact answer -- it depends on the head geometry and on
    whether the hybrid layers keep a recurrent state instead of a KV block --
    so this errs high. Over-estimating costs a little GPU throughput;
    under-estimating costs an OOM four hours into a run.
    """
    if not info.n_layers:
        return 0
    bytes_per_token_per_layer = 2 * 1024 * (1 if quantised else 2)
    total = info.n_layers * ctx * bytes_per_token_per_layer
    return int(total / 1_048_576)


def auto_n_cpu_moe(
    info: GGUFInfo,
    model_bytes: int,
    ctx: int,
    *,
    free_mb: int | None = None,
    reserve_mb: int = 1024,
    mmproj_mb: int = 0,
    kv_quantised: bool = True,
    expert_fraction: float = DEFAULT_EXPERT_FRACTION,
) -> tuple[int, dict[str, Any]]:
    """Decide how many layers' experts go to CPU RAM.

    Returns `(n_cpu_moe, explanation)`. The explanation is surfaced in the UI
    and written into the run record, because "why did it pick 40" is the first
    question anyone asks about a slow run.

    The model is: attention and the shared trunk want to be on the GPU (that is
    what makes prefill fast), the KV cache and the vision projector must fit
    beside them, and whatever VRAM is left over buys expert layers at
    `per_layer` bytes each. Everything that does not fit goes to RAM, which is
    the safe direction -- RAM is merely slower, VRAM exhaustion is a crash.
    """
    layers = info.n_layers
    if not layers:
        # Unknown geometry: put every expert in RAM. Slower, always boots.
        return 0, {"mode": "fallback", "reason": "layer count unknown",
                   "n_cpu_moe": 0, "note": "experts left on CPU by default"}

    if free_mb is None:
        free_mb = free_vram_mb()

    if free_mb <= 0:
        return layers, {"mode": "cpu-only", "reason": "no usable GPU detected",
                        "n_cpu_moe": layers, "layers": layers}

    total_mb = model_bytes / 1_048_576
    expert_mb = total_mb * expert_fraction
    non_expert_mb = total_mb - expert_mb
    per_layer_mb = expert_mb / layers if layers else 0.0
    kv_mb = _kv_cache_mb(info, ctx, kv_quantised)

    budget = free_mb - reserve_mb - kv_mb - mmproj_mb - non_expert_mb
    detail: dict[str, Any] = {
        "mode": "auto",
        "layers": layers,
        "free_vram_mb": free_mb,
        "reserve_mb": reserve_mb,
        "kv_cache_mb": kv_mb,
        "mmproj_mb": mmproj_mb,
        "non_expert_mb": round(non_expert_mb),
        "per_expert_layer_mb": round(per_layer_mb, 1),
        "expert_budget_mb": round(budget),
    }

    if budget <= 0 or per_layer_mb <= 0:
        detail["n_cpu_moe"] = layers
        detail["reason"] = (
            "no VRAM left for experts after attention, KV cache and reserve; "
            "all experts in system RAM"
        )
        return layers, detail

    on_gpu = min(layers, int(budget // per_layer_mb))
    n_cpu_moe = layers - on_gpu
    detail["expert_layers_on_gpu"] = on_gpu
    detail["n_cpu_moe"] = n_cpu_moe
    detail["reason"] = f"{on_gpu}/{layers} expert layers fit in VRAM"
    return n_cpu_moe, detail


# -- command construction ---------------------------------------------------

def _flag_supported(flag: str) -> bool:
    """Ask the binary whether it knows a flag.

    llama.cpp's CLI moves. MTP in particular is documented two different ways
    upstream (`--spec-type draft-mtp` vs `--mtp-nb-heads`), and a rejected flag
    makes llama-server exit during load with a message nobody reads. Checking
    turns that into a clear error at configuration time.
    """
    try:
        out = subprocess.run([LLAMA_SERVER, "--help"], capture_output=True,
                             text=True, timeout=30)
        return flag in (out.stdout + out.stderr)
    except (OSError, subprocess.SubprocessError):
        return False


def build_command(
    entry: ModelEntry,
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    models_dir: Path = MODELS_DIR,
) -> tuple[list[str], dict[str, Any]]:
    """Assemble the llama-server argv. Returns `(cmd, placement)`."""
    cfg: dict[str, Any] = {**defaults, **(entry.engine_opts or {}),
                           **(overrides or {})}

    model_path = entry.gguf_path(models_dir)
    if not model_path.exists():
        raise EngineError(f"{entry.id}: {model_path.name} is not on disk")

    try:
        info = read_info(model_path)
    except GGUFError as exc:
        raise EngineError(f"{entry.id}: {exc}") from exc

    if not info.chat_template:
        # --jinja with no embedded template silently falls back to a generic
        # one, and generic templates do not emit Qwen tool calls.
        raise EngineError(
            f"{entry.id}: {model_path.name} carries no chat template, so tool "
            "calling cannot work. Re-download from a repo that ships one."
        )

    ctx = int(cfg.get("ctx", 65536))
    if entry.context_ceiling and ctx > entry.context_ceiling:
        ctx = entry.context_ceiling

    mmproj_path = entry.mmproj_path(models_dir)
    mmproj_mb = int(mmproj_path.stat().st_size / 1_048_576) if (
        mmproj_path and mmproj_path.exists()) else 0

    requested = cfg.get("n_cpu_moe", "auto")
    if isinstance(requested, int) or (isinstance(requested, str)
                                      and requested.isdigit()):
        n_cpu_moe = int(requested)
        placement = {"mode": "pinned", "n_cpu_moe": n_cpu_moe,
                     "layers": info.n_layers,
                     "reason": "set explicitly in the registry or by the sweep"}
    else:
        n_cpu_moe, placement = auto_n_cpu_moe(
            info, model_path.stat().st_size, ctx,
            reserve_mb=int(cfg.get("vram_reserve_mb", 1024)),
            mmproj_mb=mmproj_mb,
            kv_quantised=str(cfg.get("cache_type_k", "q8_0")) != "f16",
            expert_fraction=float(cfg.get("expert_fraction",
                                          DEFAULT_EXPERT_FRACTION)),
        )
    placement["model"] = info.as_dict()

    cmd = [
        LLAMA_SERVER,
        "--model", str(model_path),
        # The alias is what /v1/models reports, which is what _serves() matches
        # and what the harness puts in its `model` field.
        "--alias", entry.id,
        "--host", SERVE_HOST,
        "--port", str(SERVE_PORT),
        "--ctx-size", str(ctx),
        "--threads", str(cfg.get("threads", 8)),
        "--batch-size", str(cfg.get("batch", 2048)),
        "--ubatch-size", str(cfg.get("ubatch", 512)),
        "--cache-type-k", str(cfg.get("cache_type_k", "q8_0")),
        "--cache-type-v", str(cfg.get("cache_type_v", "q8_0")),
        "--flash-attn", str(cfg.get("flash_attn", "on")),
        # One conversation at a time. The benchmark runs one agent; extra slots
        # would split the KV cache for nothing.
        "--parallel", "1",
        # THE flag. Without --jinja the model's own chat template is bypassed
        # and tool calls come back as literal <tools> text instead of
        # tool_calls. Every local-agent setup that "cannot call tools" is this.
        "--jinja",
        # Truncating mid-run would silently drop the earlier half of an agent
        # transcript. Fail loudly instead; the orchestrator handles the error.
        "--no-context-shift",
        "--metrics",
    ]

    if entry.gpu and free_vram_mb() > 0:
        cmd += ["--n-gpu-layers", "99"]
        if n_cpu_moe > 0:
            cmd += ["--n-cpu-moe", str(n_cpu_moe)]
    else:
        cmd += ["--n-gpu-layers", "0"]
        placement["mode"] = "cpu-only"

    if mmproj_path and mmproj_path.exists():
        cmd += ["--mmproj", str(mmproj_path)]
        placement["mmproj"] = mmproj_path.name

    if cfg.get("mlock"):
        cmd.append("--mlock")
    if cfg.get("no_mmap"):
        cmd.append("--no-mmap")

    spec = entry.spec or {}
    if spec.get("type"):
        if not _flag_supported("--spec-type"):
            raise EngineError(
                f"{entry.id}: this llama-server build does not support "
                "--spec-type, so the MTP draft heads cannot be used. Either "
                "rebuild at a newer ref or select the non-MTP model row."
            )
        cmd += ["--spec-type", str(spec["type"]),
                "--spec-draft-n-max", str(spec.get("draft_n_max", 2))]
        placement["speculative"] = dict(spec)

    # Whatever the registry says, passed through without interpretation. This
    # is the seam that keeps the engine model-agnostic: nothing here knows what
    # `preserve_thinking` or `enable_thinking` mean, only that the model asked
    # for them.
    if entry.chat_template_kwargs:
        try:
            encoded = json.dumps(entry.chat_template_kwargs)
        except (TypeError, ValueError) as exc:
            raise EngineError(
                f"{entry.id}: chat_template_kwargs is not JSON-serialisable: "
                f"{exc}"
            ) from exc
        cmd += ["--chat-template-kwargs", encoded]

    # Per-model sampling defaults. A harness that sends its own values in the
    # request overrides these; they exist so a model behaves sanely in Direct
    # chat and under a harness that sends none.
    for key, flag in (("temp", "--temp"), ("top_p", "--top-p"),
                      ("top_k", "--top-k"), ("min_p", "--min-p"),
                      ("repeat_penalty", "--repeat-penalty")):
        if key in entry.sampling:
            cmd += [flag, str(entry.sampling[key])]

    return cmd, placement


def _port_in_use(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


class EngineManager:
    """Starts, stops and health-checks the single llama-server process."""

    def __init__(self) -> None:
        self.status = EngineStatus()
        self.proc: subprocess.Popen | None = None
        self.entry: ModelEntry | None = None
        self._lock = threading.Lock()
        self._log_lines: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        entry: ModelEntry,
        defaults: dict[str, Any],
        flag_overrides: dict[str, Any] | None = None,
        on_log: Callable[[str], None] | None = None,
        **_ignored: Any,
    ) -> None:
        """Start llama-server for one model and block until it is healthy.

        `**_ignored` absorbs arguments the colibri adapter took and this one has
        no use for (`queue_timeout_s`, `ram_headroom_gb`). They were artefacts
        of disk-streaming: there is no request queue to time out here, and the
        RAM budget is simply the size of the file.
        """
        with self._lock:
            if self.proc is not None:
                self.stop()

            if not Path(LLAMA_SERVER).exists() and not shutil.which("llama-server"):
                raise EngineError(
                    f"llama-server not found at {LLAMA_SERVER}. "
                    "Run scripts/build-llama.sh, or set LLAMA_SERVER_BIN."
                )

            cmd, placement = build_command(entry, defaults, flag_overrides)

            if self._port_busy():
                raise EngineError(
                    f"port {SERVE_PORT} is already in use by another process. "
                    "Stop it first: with host networking a stray llama-server "
                    "on the host occupies the same port as the container's."
                )

            env = os.environ.copy()
            env.update({k: str(v) for k, v in entry.env.items()
                        if k.startswith("LLAMA_ARG_")})

            self.status = EngineStatus(
                model_id=entry.id, loading=True, message="starting engine",
                started_at=time.time(), placement=placement,
            )
            self.entry = entry
            self._log_lines = []

            if on_log:
                on_log(f"[engine] {' '.join(cmd)}")
                on_log(f"[engine] offload: {placement.get('reason', 'n/a')}")

            self.proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
            self.status.pid = self.proc.pid
            threading.Thread(
                target=self._drain_logs, args=(on_log,), daemon=True
            ).start()

        self._await_health(on_log, entry.id)

    def _drain_logs(self, on_log: Callable[[str], None] | None) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip("\n")
            self._log_lines.append(line)
            del self._log_lines[:-500]   # keep the tail bounded
            if on_log:
                on_log(line)

    def _serves(self, model_id: str) -> bool:
        """True only if the server on our port advertises *our* model.

        With host networking anything else bound to this port answers /health
        too. Without this check a stale server left over from a manual run is
        mistaken for a successful load and every request afterwards fails.
        """
        try:
            r = httpx.get(f"{base_url()}/v1/models", timeout=5.0)
            if r.status_code != 200:
                return False
            return any(m.get("id") == model_id for m in r.json().get("data", []))
        except (httpx.HTTPError, ValueError):
            return False

    def _await_health(self, on_log: Callable[[str], None] | None,
                      expect_model_id: str) -> None:
        """Poll /health until our engine answers.

        No deadline by design, but unlike colibri this is now a matter of
        seconds to a couple of minutes: it is a 27 GB sequential read into page
        cache, not a cold expert-by-expert warm-up.
        """
        while True:
            if self.proc is None or self.proc.poll() is not None:
                tail = "\n".join(self._log_lines[-25:])
                failed_id = self.status.model_id
                self.status = EngineStatus(
                    message=f"{failed_id}: engine exited during load")
                self.proc = None
                self.entry = None
                raise EngineError(f"engine exited during load:\n{tail}")
            try:
                r = httpx.get(f"{base_url()}/health", timeout=5.0)
                if r.status_code == 200 and self._serves(expect_model_id):
                    self.status.loading = False
                    self.status.healthy = True
                    self.status.message = "ready"
                    if on_log:
                        on_log("[engine] healthy")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)

    def stop(self) -> None:
        """Stop the engine and its process group, escalating if it ignores us."""
        proc, self.proc = self.proc, None
        if proc is None:
            self.status = EngineStatus()
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=10)
        self.status = EngineStatus()
        self.entry = None

    def _port_busy(self) -> bool:
        return _port_in_use(SERVE_HOST, SERVE_PORT)

    # -- introspection -----------------------------------------------------

    # llama-server reports throughput in its Prometheus endpoint. These are the
    # engine's own numbers, which is what BENCHMARKS.md should quote: Colibri's
    # metrics were parsed out of harness stdout and came back empty for
    # OpenCode, leaving three runs with no ttft at all.
    _METRIC_RE = re.compile(r"^(?P<name>llamacpp:[a-z_]+)\s+(?P<value>[0-9.eE+-]+)$",
                            re.MULTILINE)

    def health(self) -> dict[str, Any]:
        """Engine-side counters, when it is up."""
        out: dict[str, Any] = {}
        try:
            r = httpx.get(f"{base_url()}/health", timeout=5.0)
            if r.status_code == 200:
                out["health"] = r.json()
        except (httpx.HTTPError, ValueError):
            return out
        try:
            m = httpx.get(f"{base_url()}/metrics", timeout=5.0)
            if m.status_code == 200:
                out["metrics"] = {
                    match.group("name"): float(match.group("value"))
                    for match in self._METRIC_RE.finditer(m.text)
                }
        except (httpx.HTTPError, ValueError):
            pass
        return out

    def logs(self, n: int = 200) -> list[str]:
        return self._log_lines[-n:]

    @property
    def loaded_model_id(self) -> str | None:
        return self.status.model_id if self.status.healthy else None
