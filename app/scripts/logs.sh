#!/usr/bin/env bash
# Follow the container logs. Pass a number to limit the backlog.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh
"${COMPOSE[@]}" logs -f --tail "${1:-200}"
