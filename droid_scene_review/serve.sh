#!/usr/bin/env bash
set -euo pipefail

root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
port="${1:-8142}"

exec python3 -m http.server "$port" --bind 127.0.0.1 --directory "$root"
