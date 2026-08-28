#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_dir/.venv-xlsr/bin/python"

if [[ ! -x "$python" ]]; then
  echo "Missing shared Transformers environment. Run: scripts/setup_xlsr.sh" >&2
  exit 1
fi

exec "$python" "$repo_dir/scripts/transcribe_mms.py" "$@"
