#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_dir/.venv-gigaam/bin/python"

if [[ ! -x "$python" ]]; then
  echo "Missing GigaAM environment. Run: scripts/setup_gigaam.sh" >&2
  exit 1
fi

exec "$python" "$repo_dir/scripts/transcribe_gigaam.py" "$@"

