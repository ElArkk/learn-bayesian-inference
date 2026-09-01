#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

uv sync

if [[ ! -f my_lab.py ]]; then
  uv run python scripts/sync_my_lab.py
else
  uv run python scripts/sync_my_lab.py --status
fi

lock_file="$project_dir/.my_lab.running"
if [[ -f "$lock_file" ]]; then
  running_pid="$(<"$lock_file")"
  if kill -0 "$running_pid" 2>/dev/null; then
    echo "my_lab.py already has an active launcher process: $running_pid" >&2
    exit 1
  fi
fi

printf '%s\n' "$$" > "$lock_file"
trap 'rm -f "$lock_file"' EXIT INT TERM

uv run marimo edit \
  my_lab.py \
  --no-token \
  --no-sandbox \
  --port 2722
