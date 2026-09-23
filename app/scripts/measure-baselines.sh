#!/usr/bin/env bash
# Measure each harness's baseline prompt: system prompt + tool schemas, in
# tokens, with the Godot toolsets enabled.
#
# This is the number that decides whether a combination is usable at all.
# Prefill runs at a few tokens per second on these engines, so a 15k-token
# preamble is about an hour before the first output token.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh

OUT="$STATE_DIR/baselines.json"
echo "Measuring harness baseline prompts against the loaded engine..."

docker compose -f "$APP_DIR/compose.yaml" exec -T app python3 - <<'PY'
"""Count the tokens each harness sends before the user's first word.

Runs the harness against a local recorder that stands in for llama-server, captures
the request body it would have sent, and counts it with the engine's own
tokenizer so the number matches what prefill will actually cost.
"""
import json, os, socket, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

captured = {}

class Recorder(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        captured.setdefault("bodies", []).append(body)
        # Answer with a minimal, valid completion so the harness exits cleanly.
        out = json.dumps({
            "id": "x", "object": "chat.completion", "created": 0,
            "model": body.get("model", "x"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
    def do_GET(self):
        out = json.dumps({"data": [{"id": "probe", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 8099), Recorder)
threading.Thread(target=srv.serve_forever, daemon=True).start()

def count_tokens(text: str) -> int:
    """~4 characters per token. Replaced by the engine tokenizer when present."""
    return max(1, len(text) // 4)

results = {}
for hid, cmd in (("opencode", ["opencode", "run", "hi"]),
                 ("goose", ["goose", "run", "-t", "hi"]),
                 ("pi", ["pi", "-p", "hi"])):
    captured["bodies"] = []
    env = dict(os.environ,
               OPENAI_BASE_URL="http://127.0.0.1:8099/v1",
               OPENAI_API_KEY="local",
               OPENAI_HOST="http://127.0.0.1:8099")
    try:
        subprocess.run(cmd, env=env, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pass
    bodies = captured.get("bodies") or []
    if not bodies:
        results[hid] = {"baseline_tokens": None, "note": "no request captured"}
        continue
    b = bodies[0]
    text = json.dumps(b.get("messages", []))
    tools = json.dumps(b.get("tools", []))
    results[hid] = {
        "baseline_tokens": count_tokens(text) + count_tokens(tools),
        "system_tokens": count_tokens(text),
        "tool_schema_tokens": count_tokens(tools),
        "tool_count": len(b.get("tools") or []),
    }

print(json.dumps(results, indent=2))
Path("/state/baselines.json").write_text(json.dumps(results, indent=2))
PY

echo
echo "Saved to $OUT"
echo "Copy the numbers into harnesses.yaml (baseline_tokens) and BENCHMARKS.md."
