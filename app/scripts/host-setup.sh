#!/usr/bin/env bash
# Every privileged step this project needs, in one script. Run it once:
#
#     bash "$HOME/Desktop/RAM_Agent/app/scripts/host-setup.sh"
#
# It is idempotent: re-running it is safe and skips anything already done.
# Everything it does is logged to HOST_CHANGES.md by the caller.
set -euo pipefail

say() { printf '\n==> %s\n' "$1"; }
ROOT="${RAM_AGENT_ROOT:-$HOME/Desktop/RAM_Agent}"

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as your normal user; it calls sudo itself where needed." >&2
  exit 1
fi

say "1/4  Packages"
# cuda      -> nvcc, to build the DeepSeek V4 CUDA expert tier (sm_86 / RTX 3060)
# fio       -> storage benchmarks in preflight; the engines are disk-bound
# python-yaml -> host-side scripts that read models.yaml
need=()
for p in cuda fio python-yaml; do
  pacman -Q "$p" >/dev/null 2>&1 || need+=("$p")
done
if [ ${#need[@]} -gt 0 ]; then
  echo "installing: ${need[*]}"
  sudo pacman -S --needed --noconfirm "${need[@]}"
else
  echo "nothing to install"
fi

say "2/4  Docker service"
# Docker already answers over its socket here, but nothing enables it at boot.
if ! systemctl is-enabled --quiet docker.socket 2>/dev/null; then
  sudo systemctl enable --now docker.socket
  echo "enabled docker.socket"
else
  echo "docker.socket already enabled"
fi

say "3/4  NVIDIA container runtime"
# Registers the nvidia runtime with the Docker daemon so the container can see
# the GPU. nvidia-container-toolkit is already installed on this host.
if docker info 2>/dev/null | grep -qi nvidia; then
  echo "nvidia runtime already registered with Docker"
else
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker 2>/dev/null || true
  echo "registered nvidia runtime; docker restarted"
fi

say "4/4  Sleep inhibition check"
# Runs are held awake per-run with systemd-inhibit, which needs no privileges.
# This only reports whether anything would override that.
printf '  logind idle action : %s\n' "$(loginctl show-session -p IdleAction --value 2>/dev/null || echo '?')"
printf '  sleep targets      : %s\n' "$(systemctl is-enabled sleep.target 2>/dev/null || echo '?')"
echo "  (runs use: systemd-inhibit --what=idle:sleep:handle-lid-switch)"

say "Done"
echo "Verify with: bash \"$ROOT/app/scripts/preflight.sh\""
