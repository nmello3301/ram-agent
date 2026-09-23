#!/usr/bin/env bash
# Show what is pinned and what upstream has now. Never changes a pin by itself:
# a benchmark whose toolchain moved under it is not a benchmark.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

latest() { curl -fsS -m 20 "https://api.github.com/repos/$1/releases/latest" \
  | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p' | head -1; }

pinned() { grep -oE "$2" "$1" | head -1; }

printf '%-22s %-16s %s\n' COMPONENT PINNED LATEST
printf '%-22s %-16s %s\n' llama.cpp \
  "$(pinned models.yaml 'b[0-9]{4,}')" "$(latest ggml-org/llama.cpp)"
printf '%-22s %-16s %s\n' godot \
  "4.7.2-stable" "$(latest godotengine/godot-builds)"
printf '%-22s %-16s %s\n' opencode \
  "$(pinned harnesses.yaml '1\.[0-9]+\.[0-9]+')" "$(latest anomalyco/opencode)"
printf '%-22s %-16s %s\n' goose \
  "v1.50.1" "$(latest block/goose)"
printf '%-22s %-16s %s\n' godot-editor-mcp "2026.9.10" \
  "$(curl -fsS -m 20 https://pypi.org/pypi/godot-editor-mcp/json | sed -n 's/.*"version": "\([^"]*\)".*/\1/p' | head -1)"

cat <<'NOTE'

To move a pin, edit it in BOTH places it appears (models.yaml or
harnesses.yaml, and the matching ARG in the Dockerfile), then rebuild with
scripts/up.sh and re-run the benchmark. Numbers from different pins are not
comparable and should not share a table in BENCHMARKS.md.
NOTE
