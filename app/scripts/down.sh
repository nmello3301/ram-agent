#!/usr/bin/env bash
# Stop the app. Models, projects and run history are on bind mounts and stay.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
. scripts/_common.sh
"${COMPOSE[@]}" down
echo "Stopped. models/, projects/ and state/ are untouched."
