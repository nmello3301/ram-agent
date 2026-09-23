#!/usr/bin/env bash
# Build if needed, start the app, and open the UI in Firefox.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

CUDA_ARCH="$(detect_cuda_arch)"
export CUDA_ARCH
check_cuda_version
echo "Building for CUDA compute capability $CUDA_ARCH"
echo "Free VRAM: $(free_vram_mb) MiB (the offload split is sized from this)"

"${COMPOSE[@]}" up -d --build

printf 'Waiting for the backend'
for _ in $(seq 1 120); do
  if curl -fsS -m 3 "$UI_URL/api/health" >/dev/null 2>&1; then
    printf ' ready\n'
    if command -v firefox >/dev/null; then
      firefox "$UI_URL" >/dev/null 2>&1 &
    fi
    echo "UI: $UI_URL"
    exit 0
  fi
  printf '.'; sleep 2
done
printf '\nBackend did not come up. Logs:\n'
"${COMPOSE[@]}" logs --tail 60
exit 1
