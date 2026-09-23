# Shared by the scripts in this directory. Sourced, not executed.
RAM_AGENT_ROOT="${RAM_AGENT_ROOT:-$HOME/Desktop/RAM_Agent}"
APP_DIR="$RAM_AGENT_ROOT/app"
STATE_DIR="$RAM_AGENT_ROOT/state"
MODELS_DIR="$RAM_AGENT_ROOT/models"
UI_URL="http://127.0.0.1:8088"
COMPOSE=(docker compose -f "$APP_DIR/compose.yaml")

export RAM_AGENT_ROOT HOST_UID HOST_GID
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# Build args come from what the machine actually is, not from constants.
# llama.cpp takes a bare compute-capability number, not colibri's sm_ prefix.
detect_cuda_arch() {
  local cc
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
  [ -n "$cc" ] && echo "$cc" || echo "86"
}

# Free VRAM in MiB. The offload split is sized from this, and it is `free`
# rather than `total` on purpose: ComfyUI or Blender holding the card changes
# the answer.
free_vram_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 || echo 0
}

# Refuse to run against a CUDA version known to produce garbage with Qwen3.6.
check_cuda_version() {
  local v
  v="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA UMD Version: *[0-9.]+' | grep -oE '[0-9.]+' | head -1)"
  if [ "$v" = "13.2" ]; then
    printf 'WARNING: CUDA 13.2 is known to produce gibberish output with Qwen3.6.\n' >&2
    printf '         Use 13.3 or newer, or something below 13.2.\n' >&2
  fi
}
