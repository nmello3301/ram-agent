#!/usr/bin/env bash
# Build llama.cpp natively, for running the app outside Docker.
#
# The container builds its own copy (see the Dockerfile); this is for a native
# run, and for llama-bench, which is what the -ncmoe and quant sweeps use.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

LLAMA_REF="${LLAMA_REF:-b11115}"
CUDA_ARCH="$(detect_cuda_arch)"
SRC="$STATE_DIR/llama.cpp"
PREFIX="$STATE_DIR/llama"

check_cuda_version

mkdir -p "$STATE_DIR"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp.git "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin "$LLAMA_REF"
  git -C "$SRC" checkout -q FETCH_HEAD
fi

echo "Building llama.cpp $LLAMA_REF for compute capability $CUDA_ARCH"
# GGML_NATIVE is on by default and correct here: built and run on the same
# machine. This host is a Ryzen 7 4800H -- AVX2, FMA, F16C, BMI2, no AVX-512.
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=ON \
  -DBUILD_SHARED_LIBS=OFF
cmake --build "$SRC/build" --config Release -j"$(nproc)" \
  --target llama-server llama-cli llama-bench

mkdir -p "$PREFIX/bin"
install -m755 "$SRC/build/bin/llama-server" "$SRC/build/bin/llama-cli" \
              "$SRC/build/bin/llama-bench" "$PREFIX/bin/"
git -C "$SRC" rev-parse HEAD > "$PREFIX/LLAMA_COMMIT"

# The two flags the whole project depends on. Fail here, not mid-run.
for flag in --n-cpu-moe --jinja; do
  "$PREFIX/bin/llama-server" --help | grep -q -- "$flag" \
    || { echo "FATAL: this build has no $flag" >&2; exit 1; }
done

printf '\nBuilt: %s\n' "$PREFIX/bin/llama-server"
printf 'Commit: %s\n' "$(cat "$PREFIX/LLAMA_COMMIT")"
printf '\nTo use it natively:\n  export LLAMA_SERVER_BIN=%s/bin/llama-server\n' "$PREFIX"
