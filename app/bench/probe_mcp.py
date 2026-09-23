"""Speak MCP over stdio to a server and report its real tool catalogue."""
import json, subprocess, sys, time

cmd = sys.argv[1:]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)

def send(obj):
    p.stdin.write(json.dumps(obj) + "\n"); p.stdin.flush()

def read(timeout=25):
    end = time.time() + timeout
    while time.time() < end:
        line = p.stdout.readline()
        if not line:
            time.sleep(0.05); continue
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None

send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2025-06-18",
                 "capabilities": {},
                 "clientInfo": {"name": "ram-agent-probe", "version": "0"}}})
init = read()
if not init:
    print("NO INITIALIZE RESPONSE"); print(p.stderr.read()[:2000]); sys.exit(1)
info = init.get("result", {}).get("serverInfo", {})
print(f"server: {info.get('name')} {info.get('version')}")
print(f"protocol: {init.get('result', {}).get('protocolVersion')}")

send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
res = read()
tools = (res or {}).get("result", {}).get("tools", [])
print(f"\nTOOLS: {len(tools)}")

# Group by the namespace prefix, and price the catalogue the way a harness
# actually sends it: as OpenAI function-tools.
ns = {}
for t in tools:
    name = t.get("name", "")
    parts = name.split("_")
    key = "_".join(parts[:2]) if len(parts) > 2 else (parts[0] if parts else "?")
    ns.setdefault(key, []).append(t)

encoded = json.dumps([
    {"type": "function",
     "function": {"name": t.get("name"), "description": t.get("description", ""),
                  "parameters": t.get("inputSchema", {})}}
    for t in tools], separators=(",", ":"))
print(f"schema JSON: {len(encoded):,} bytes  (~{len(encoded)//4:,} tokens at 4 B/tok)")

print("\nBY NAMESPACE:")
for key in sorted(ns):
    enc = json.dumps([{"type": "function", "function": {
        "name": t.get("name"), "description": t.get("description", ""),
        "parameters": t.get("inputSchema", {})}} for t in ns[key]],
        separators=(",", ":"))
    print(f"  {key:26s} {len(ns[key]):3d} tools  ~{len(enc)//4:6,d} tok")
    for t in ns[key]:
        print(f"      {t.get('name')}")

p.terminate()
