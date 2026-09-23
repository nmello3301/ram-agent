#
# RAM_Agent — one image holding the backend, llama.cpp, the agent harnesses,
# the Godot and Blender MCP servers, and a headless Godot for validation.
#
# Two stages: a CUDA devel image that compiles llama.cpp, and a CUDA runtime
# image carrying only what is needed to run it. The CUDA version matches the
# host driver (610.43.02 / CUDA UMD 13.3).
#
# DO NOT "fix" a problem by moving to CUDA 13.2. Qwen3.6 produces gibberish
# output on 13.2; 13.3 and anything below 13.2 are fine. This is upstream's
# warning, not a guess.

# ---------------------------------------------------------------------------
# Stage 1 — build llama.cpp
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04 AS build

# Pinned llama.cpp ref. Recorded in models.yaml too; change both together.
# A benchmark whose engine moved under it is not a benchmark.
ARG LLAMA_REF=b11115
# RTX 3060 Laptop (GA106) reports compute capability 8.6.
ARG CUDA_ARCH=86

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_REF}" \
        https://github.com/ggml-org/llama.cpp.git . \
    && git rev-parse HEAD > /src/LLAMA_COMMIT

# GGML_NATIVE is on by default and is correct here: the image is built and run
# on the same laptop. This host is a Ryzen 7 4800H -- AVX2 + FMA + F16C + BMI2,
# NO AVX-512. If this image is ever moved to another machine, set
# -DGGML_NATIVE=OFF -DGGML_AVX2=ON instead.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DLLAMA_CURL=ON \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j"$(nproc)" \
        --target llama-server llama-cli llama-bench

# Fail at image-build time rather than three hours into a run.
RUN ./build/bin/llama-server --version 2>&1 | tail -3 \
    && ./build/bin/llama-server --help | grep -q -- "--n-cpu-moe" \
        || (echo "FATAL: this llama.cpp build has no --n-cpu-moe" && exit 1) \
    && ./build/bin/llama-server --help | grep -q -- "--jinja" \
        || (echo "FATAL: this llama.cpp build has no --jinja" && exit 1)

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04

ARG GODOT_VERSION=4.7.2-stable
ARG GODOT_SHA512=9aa00f7a605200940bce3027a567b782f49bd8e940dd06ae9e987bd65aee1b1467edd56ed84fcdcbdd44354bf613bdbb4e5d2913e925850368e150c59ed54c65
# Harness versions, pinned. See harnesses.yaml for what each one is and why.
ARG OPENCODE_VERSION=1.18.31
ARG GOOSE_VERSION=v1.50.1
ARG PI_VERSION=0.85.1
ARG GODOT_MCP_VERSION=2026.9.10
# Blender MCP. Pinned loosely because the invocation is not yet verified
# against an installed server -- see mcp.yaml, `verified: false`.
ARG BLENDER_MCP_VERSION=0.1.3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        libgomp1 libcurl4 \
        ca-certificates curl unzip xz-utils git \
        procps tini \
        # headless Godot still links these even with --headless
        libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Node, for the OpenCode and Pi CLIs.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# --- llama.cpp -----------------------------------------------------------
WORKDIR /opt/llama.cpp
COPY --from=build /src/build/bin/llama-server \
                  /src/build/bin/llama-cli \
                  /src/build/bin/llama-bench \
                  ./bin/
COPY --from=build /src/LLAMA_COMMIT ./LLAMA_COMMIT
RUN ln -s /opt/llama.cpp/bin/llama-server /usr/local/bin/llama-server \
    && ln -s /opt/llama.cpp/bin/llama-cli /usr/local/bin/llama-cli \
    && ln -s /opt/llama.cpp/bin/llama-bench /usr/local/bin/llama-bench

# --- Godot (headless, validation only) -----------------------------------
# The interactive editor runs on the host under Wayland; this copy only loads
# projects and runs scenes to check them.
RUN curl -fsSLO "https://github.com/godotengine/godot-builds/releases/download/${GODOT_VERSION}/Godot_v${GODOT_VERSION}_linux.x86_64.zip" \
    && echo "${GODOT_SHA512}  Godot_v${GODOT_VERSION}_linux.x86_64.zip" | sha512sum -c - \
    && unzip -q "Godot_v${GODOT_VERSION}_linux.x86_64.zip" -d /opt/godot \
    && rm "Godot_v${GODOT_VERSION}_linux.x86_64.zip" \
    && mv /opt/godot/Godot_v${GODOT_VERSION}_linux.x86_64 /opt/godot/godot \
    && chmod +x /opt/godot/godot \
    && ln -s /opt/godot/godot /usr/local/bin/godot

# --- MCP servers ---------------------------------------------------------
# Godot: hybridindie/godot-mcp. NOT Coding-Solo/godot-mcp, which has a command
# injection RCE below 0.1.1 (CVE-2026-25546).
RUN pip install --no-cache-dir "godot-editor-mcp==${GODOT_MCP_VERSION}"

# Blender: the server runs here, Blender itself runs on the host (it needs a
# display and the GPU), exactly like the Godot editor bridge.
#
# NOT VERIFIED. The package name and invocation are the documented shape, not
# one confirmed against an install -- mcp.yaml marks this server
# `verified: false` and the UI warns on any run that uses it. A failure here is
# non-fatal so the Godot phase still works in a fresh image.
RUN pip install --no-cache-dir "mcp<2" "blender-mcp-server==${BLENDER_MCP_VERSION}" \
    || echo "blender-mcp-server not installed; Blender phase unavailable"

# --- Harnesses -----------------------------------------------------------
RUN npm install -g --no-fund --no-audit "opencode-ai@${OPENCODE_VERSION}"

# Pi: installed with --ignore-scripts, as pi.dev documents.
RUN npm install -g --ignore-scripts --no-fund --no-audit \
        "@earendil-works/pi-coding-agent@${PI_VERSION}"
# Pi ships no MCP support by design ("No MCP - build an extension"). This MIT
# adapter is what lets it reach the MCP servers; noted in the README.
RUN npm install -g --no-fund --no-audit "pi-mcp-adapter" || \
    echo "pi-mcp-adapter not on npm; installed from git at runtime instead"

# Goose: musl build, which is the most portable of the published assets.
RUN curl -fsSL -o /tmp/goose.tar.gz \
        "https://github.com/block/goose/releases/download/${GOOSE_VERSION}/goose-x86_64-unknown-linux-musl.tar.gz" \
    && tar -xzf /tmp/goose.tar.gz -C /usr/local/bin \
    && rm /tmp/goose.tar.gz \
    && chmod +x /usr/local/bin/goose

# --- Backend -------------------------------------------------------------
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY ui/ /app/ui/
COPY bench/ /app/bench/
COPY templates/ /app/templates/
COPY models.yaml harnesses.yaml mcp.yaml /app/

# The container runs as the host user, whose UID has no home directory in this
# image. The harnesses all need a writable HOME for their configs and session
# state; putting it on the state bind mount also means a harness session
# survives a restart, which is what lets a resumed run reuse context instead of
# re-prefilling it. (A comment cannot live inside an ENV continuation, hence
# its position here.)
ENV LLAMA_HOME=/opt/llama.cpp \
    LLAMA_SERVER_BIN=/opt/llama.cpp/bin/llama-server \
    GODOT_BIN=/opt/godot/godot \
    MODELS_DIR=/models \
    PROJECTS_DIR=/projects \
    ASSETS_DIR=/assets \
    STATE_DIR=/state \
    HOME=/state/home \
    XDG_CONFIG_HOME=/state/home/.config \
    XDG_DATA_HOME=/state/home/.local/share \
    XDG_CACHE_HOME=/state/home/.cache \
    NPM_CONFIG_CACHE=/state/home/.npm

EXPOSE 8088

# tini reaps the harness subprocess trees; without it a killed harness can
# leave zombies behind across a long benchmark matrix.
ENTRYPOINT ["/usr/bin/tini", "--"]
# --timeout-keep-alive is kept high. It was essential under Colibri, where a
# turn produced no bytes for an hour. It is now belt-and-braces: a Blender
# render or a headless Godot import can still hold a connection open past any
# default, and the backend owns the clock.
CMD ["python3", "-m", "uvicorn", "backend.main:app", \
     "--host", "127.0.0.1", "--port", "8088", \
     "--timeout-keep-alive", "3600"]
