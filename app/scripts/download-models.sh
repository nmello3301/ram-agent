#!/usr/bin/env bash
# Download the pinned model files. Resumable: re-running skips what is already
# complete, so it is safe to interrupt and restart. The backend's model manager
# uses the same layout, so anything pulled here is picked up by the UI.
#
# The critical difference from Colibri's version: these repos publish EVERY
# quantisation side by side. Pulling a repo wholesale is 500-900 GB for a 27 GB
# model, so each entry names its exact files and nothing else is fetched.
set -euo pipefail

ROOT="${RAM_AGENT_ROOT:-$HOME/Desktop/RAM_Agent}"
MODELS="$ROOT/models"
LOG_DIR="$ROOT/state/downloads"
VENV="$ROOT/state/venv"
mkdir -p "$MODELS" "$LOG_DIR"

if [ -f "$ROOT/app/.env" ]; then
  # shellcheck disable=SC1091
  set -a; . "$ROOT/app/.env"; set +a
fi
export HF_XET_HIGH_PERFORMANCE=1

# id|repo|revision|size_gb|file[,file...]
# Only the default is pulled by default. The sweep rows are one flag away:
#   scripts/download-models.sh --all
DEFAULT_LIST=(
  "qwen36-35b-a3b-q5|unsloth/Qwen3.6-35B-A3B-GGUF|a483e9e6cbd595906af30beda3187c2663a1118c|28|Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf,mmproj-F16.gguf"
)
SWEEP_LIST=(
  "qwen36-35b-a3b-q5-mtp|unsloth/Qwen3.6-35B-A3B-MTP-GGUF|5bc3e238d916f48a861bac2f8a1990a0e9b7e98d|29|Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf,mmproj-F16.gguf"
  "qwen36-35b-a3b-q4|unsloth/Qwen3.6-35B-A3B-GGUF|a483e9e6cbd595906af30beda3187c2663a1118c|24|Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf,mmproj-F16.gguf"
  "qwen36-35b-a3b-q8|unsloth/Qwen3.6-35B-A3B-GGUF|a483e9e6cbd595906af30beda3187c2663a1118c|38|Qwen3.6-35B-A3B-Q8_0.gguf,mmproj-F16.gguf"
  "qwen3-coder-next-q4|unsloth/Qwen3-Coder-Next-GGUF|ce09c67b53bc8739eef83fe67b2f5d293c270632|47|Qwen3-Coder-Next-UD-Q4_K_S.gguf"
)

MODELS_LIST=("${DEFAULT_LIST[@]}")
if [ "${1:-}" = "--all" ]; then
  MODELS_LIST+=("${SWEEP_LIST[@]}")
  printf 'Downloading the default model and every sweep row (~166 GB total).\n'
else
  printf 'Downloading the default model only (~28 GB). Use --all for the sweep rows.\n'
fi

free_gb() { df -PBG "$MODELS" | awk 'NR==2{gsub("G","",$4); print $4}'; }

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "huggingface_hub==1.31.0"
fi

for entry in "${MODELS_LIST[@]}"; do
  IFS='|' read -r id repo rev size files <<<"$entry"
  dest="$MODELS/$id"
  marker="$dest/.complete"

  if [ -f "$marker" ]; then
    printf '[%s] already complete, skipping\n' "$id"; continue
  fi

  avail=$(free_gb)
  if [ "$avail" -lt "$size" ]; then
    printf '[%s] ERROR: not enough disk space: need %s GB, have %s GB\n' \
      "$id" "$size" "$avail" >&2
    exit 1
  fi

  printf '[%s] %s @ %s -> %s GB\n' "$id" "$repo" "${rev:0:8}" "$size"
  printf '[%s]   files: %s\n' "$id" "$files"
  mkdir -p "$dest"

  RA_REPO="$repo" RA_REV="$rev" RA_DEST="$dest" RA_FILES="$files" \
  "$VENV/bin/python" - <<'PY' 2>&1 | tee -a "$LOG_DIR/$id.log"
import os
from huggingface_hub import snapshot_download

patterns = [f.strip() for f in os.environ["RA_FILES"].split(",") if f.strip()]
snapshot_download(
    repo_id=os.environ["RA_REPO"],
    revision=os.environ["RA_REV"],
    local_dir=os.environ["RA_DEST"],
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
    # Without this, a 27 GB model is a 500 GB pull.
    allow_patterns=patterns,
)
print("ok")
PY

  # Verify before marking complete: a truncated GGUF otherwise surfaces much
  # later as an unreadable engine crash.
  ok=1
  IFS=',' read -ra want <<<"$files"
  for f in "${want[@]}"; do
    if [ ! -f "$dest/$f" ]; then
      printf '[%s] ERROR: %s missing after download\n' "$id" "$f" >&2; ok=0; continue
    fi
    if [ "$(head -c 4 "$dest/$f")" != "GGUF" ]; then
      printf '[%s] ERROR: %s is not a GGUF file (truncated?)\n' "$id" "$f" >&2; ok=0
    fi
  done
  [ "$ok" -eq 1 ] || exit 1

  date -Is > "$marker"
  printf '[%s] complete\n' "$id"
done

printf '\nDone. On disk:\n'
du -sh "$MODELS"/* 2>/dev/null || true
