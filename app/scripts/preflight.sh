#!/usr/bin/env bash
# Preflight: measure the host, so the build and runtime can be configured from
# facts instead of assumptions. Read-only except for fio test files, which are
# deleted. Safe to re-run.
set -euo pipefail

ROOT="${RAM_AGENT_ROOT:-$HOME/Desktop/RAM_Agent}"
MODELS_DIR="$ROOT/models"
OUT="$ROOT/state/preflight.txt"
mkdir -p "$ROOT/state" "$MODELS_DIR"

# Every model download total, in GB, from the pinned models.yaml revisions.
NEEDED_GB="${NEEDED_GB:-872}"

h() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
# Report a missing tool without aborting: preflight's job is to find gaps.
miss() { printf '  MISSING: %s (%s)\n' "$1" "${2:-not installed}"; }

main() {
printf 'ram-agent preflight  %s\n' "$(date -Is)"
printf 'host=%s kernel=%s\n' "$(uname -n)" "$(uname -r)"

h "1. CPU"
if have lscpu; then
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz|L3 cache' || true
  printf '\n  physical cores: %s\n' "$(lscpu -p=CORE 2>/dev/null | grep -v "^#" | sort -u | wc -l || echo '?')"
  printf '  logical cores:  %s\n' "$(nproc)"
  printf '\n  ISA flags relevant to the llama.cpp build:\n'
  for f in avx2 fma avx512f avx512bw avx512vl avx512_vnni avx_vnni f16c bmi2 sha_ni; do
    if grep -qw "$f" /proc/cpuinfo; then printf '    %-14s yes\n' "$f"; else printf '    %-14s no\n' "$f"; fi
  done
else miss lscpu util-linux; fi

h "2. Memory"
free -h || true
printf '\n  swap devices:\n'; swapon --show 2>/dev/null || printf '    none\n'
printf '\n  zram:\n'
if [ -e /sys/block/zram0 ]; then
  for z in /sys/block/zram*; do
    printf '    %s disksize=%s\n' "$(basename "$z")" "$(cat "$z/disksize" 2>/dev/null || echo '?')"
  done
else printf '    none\n'; fi
printf '\n  MemAvailable: %s kB\n' "$(awk '/MemAvailable/{print $2}' /proc/meminfo)"

h "3. NVIDIA / GPU"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1 || nvidia-smi 2>&1 | head -12
  printf '\n  CUDA toolkit: '; if have nvcc; then nvcc --version | tail -2 | head -1; else printf 'nvcc MISSING (needed for the DeepSeek V4 CUDA tier)\n'; fi
  printf '  CUDA_HOME candidates: '; { ls -d /opt/cuda /usr/local/cuda* 2>/dev/null || true; } | tr '\n' ' '; printf '\n'
else miss nvidia-smi 'nvidia-utils'; fi
printf '\n  GPUs on the PCI bus (hybrid graphics check):\n'
lspci 2>/dev/null | grep -Ei 'vga|3d controller' | sed 's/^/    /' || printf '    lspci unavailable\n'
printf '\n  nvidia-container-toolkit: '
if have nvidia-ctk; then nvidia-ctk --version 2>&1 | head -1; else printf 'MISSING (needed to pass the GPU into the container)\n'; fi

h "4. Target disk and free space"
printf '  models dir: %s\n' "$MODELS_DIR"
df -hT "$MODELS_DIR" 2>/dev/null || true
avail_gb=$(df -PBG "$MODELS_DIR" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
printf '\n  need %s GB for all pinned models, have %s GB free -> ' "$NEEDED_GB" "${avail_gb:-?}"
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -ge "$NEEDED_GB" ] 2>/dev/null; then printf 'OK\n'; else printf 'NOT ENOUGH\n'; fi
src=$(findmnt -no SOURCE --target "$MODELS_DIR" 2>/dev/null || echo '?')
fstype=$(findmnt -no FSTYPE --target "$MODELS_DIR" 2>/dev/null || echo '?')
printf '  device=%s fstype=%s\n' "$src" "$fstype"

h "5. NVMe and storage performance"
for d in /sys/block/nvme*; do
  [ -e "$d" ] || continue
  n=$(basename "$d")
  printf '  %s model=%s\n' "$n" "$(cat "$d/device/model" 2>/dev/null | xargs || echo '?')"
  printf '    rotational=%s scheduler=%s\n' "$(cat "$d/queue/rotational" 2>/dev/null)" "$(cat "$d/queue/scheduler" 2>/dev/null | xargs || echo '?')"
  # PCIe link: the engines are disk-bandwidth bound, so gen/width is load-bearing.
  pci=$(readlink -f "$d/device" 2>/dev/null | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | tail -1 || true)
  if [ -n "${pci:-}" ] && have lspci; then
    printf '    PCIe: %s\n' "$(lspci -vv -s "$pci" 2>/dev/null | grep -oE 'Speed [0-9.]+GT/s.*Width x[0-9]+' | head -1 || echo 'needs root for LnkSta')"
  fi
done
printf '\n  fio on %s:\n' "$MODELS_DIR"
if have fio; then
  tmpd="$MODELS_DIR/.preflight_fio"; mkdir -p "$tmpd"
  run_fio() { # name, extra args
    local name="$1"; shift
    local j
    j=$(fio --name="$name" --directory="$tmpd" --size=1G --runtime=12 --time_based \
           --ioengine=libaio --group_reporting --output-format=json "$@" 2>/dev/null) || { printf '    %-26s failed\n' "$name"; return 0; }
    printf '    %-26s %s\n' "$name" "$(printf '%s' "$j" | python3 -c '
import sys,json
try:
    j=json.load(sys.stdin)["jobs"][0]["read"]
    print(f"{j[\"bw_bytes\"]/1e6:8.1f} MB/s  {j[\"iops\"]:9.0f} IOPS  lat_p99={j[\"clat_ns\"][\"percentile\"][\"99.000000\"]/1e6:.2f} ms")
except Exception as e: print("parse error")' )"
  }
  # The engines do 4k-ish random reads for engram rows and large reads for experts.
  run_fio 4k-randread-direct   --rw=randread --bs=4k --direct=1 --iodepth=32 --numjobs=4
  run_fio 4k-randread-buffered --rw=randread --bs=4k --direct=0 --iodepth=32 --numjobs=4
  run_fio 1M-seqread-direct    --rw=read     --bs=1M --direct=1 --iodepth=8  --numjobs=1
  run_fio 1M-seqread-buffered  --rw=read     --bs=1M --direct=0 --iodepth=8  --numjobs=1
  rm -rf "$tmpd"
  printf '    (fio test files deleted)\n'
else miss fio 'fio'; fi

h "6. Filesystem specifics (btrfs / snapshots / encryption)"
if [ "${fstype:-}" = "btrfs" ]; then
  printf '  mount options: %s\n' "$(findmnt -no OPTIONS --target "$MODELS_DIR" 2>/dev/null)"
  comp=$(findmnt -no OPTIONS --target "$MODELS_DIR" 2>/dev/null | grep -o 'compress[^,]*' || echo 'none')
  printf '  compression: %s  ' "$comp"
  # Under Colibri this mattered enormously: compression silently disabled
  # O_DIRECT and halved expert-streaming throughput. It is now almost
  # irrelevant -- the model is read once at load and never touched again --
  # so it is reported rather than warned about.
  printf '(affects load time only; weights are resident after that)\n'
  printf '  subvolume: %s\n' "$(btrfs subvolume show "$MODELS_DIR" 2>/dev/null | head -1 || echo 'not a subvolume (plain directory)')"
  printf '  chattr flags on models dir: %s\n' "$(lsattr -d "$MODELS_DIR" 2>/dev/null | awk '{print $1}' || echo '?')"
  printf '\n  snapper configs:\n'
  if have snapper; then snapper --no-dbus list-configs 2>/dev/null | sed 's/^/    /' || printf '    (needs root)\n'; else printf '    snapper not installed\n'; fi
fi
printf '\n  encryption:\n'
if have lsblk; then lsblk -o NAME,TYPE,FSTYPE,MOUNTPOINT | grep -E 'crypt|luks' | sed 's/^/    /' || printf '    none detected\n'; fi
if have cryptsetup; then
  printf '\n  cryptsetup benchmark (AES throughput caps disk reads):\n'
  cryptsetup benchmark 2>/dev/null | grep -E 'aes-xts|Algorithm|PBKDF2' | head -6 | sed 's/^/    /' || printf '    (needs permissions)\n'
fi

h "7. Docker"
if have docker; then
  docker --version
  printf '  daemon: '; docker info >/dev/null 2>&1 && printf 'reachable\n' || printf 'NOT reachable (service stopped, or user not in docker group)\n'
  printf '  user in docker group: '; id -nG | grep -qw docker && printf 'yes\n' || printf 'NO\n'
  printf '  cgroup version: %s\n' "$(stat -fc %T /sys/fs/cgroup 2>/dev/null)"
  printf '  compose: '; docker compose version 2>/dev/null | head -1 || printf 'MISSING\n'
  printf '  nvidia runtime registered: '; docker info 2>/dev/null | grep -q nvidia && printf 'yes\n' || printf 'no\n'
else miss docker docker; fi

h "8. Power and thermals"
if have powerprofilesctl; then
  printf '  power profile: %s\n' "$(powerprofilesctl get 2>/dev/null || echo '?')"
else printf '  powerprofilesctl not installed\n'; fi
printf '  governors: %s\n' "$( { cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true; } | sort -u | tr '\n' ' ')"
for ac in /sys/class/power_supply/A{C,DP}*/online; do
  [ -e "$ac" ] && printf '  on AC: %s\n' "$([ "$(cat "$ac")" = 1 ] && echo yes || echo 'NO - plug in before benchmarking')"
done
if have sensors; then sensors 2>/dev/null | grep -E 'Tctl|Package|Composite|edge' | sed 's/^/  /' || true
else printf '  lm_sensors not installed\n'; fi

h "Summary of gaps"
for t in docker fio sensors nvidia-smi nvidia-ctk git python3; do
  have "$t" || printf '  install: %s\n' "$t"
done
printf '\npreflight complete: %s\n' "$(date -Is)"
}

main 2>&1 | tee "$OUT"
printf '\nSaved to %s\n' "$OUT"
