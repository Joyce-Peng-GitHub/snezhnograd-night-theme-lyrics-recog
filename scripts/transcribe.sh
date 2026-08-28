#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_dir/.venv/bin/python"

if [[ ! -x "$python" ]]; then
  echo "Missing environment. Run: uv sync" >&2
  exit 1
fi

cublas_dir="$($python -c 'import os, nvidia.cublas.lib; print(os.path.dirname(nvidia.cublas.lib.__file__))')"
cudnn_dir="$($python -c 'import os, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cudnn.lib.__file__))')"
export LD_LIBRARY_PATH="$cublas_dir:$cudnn_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$python" "$repo_dir/scripts/transcribe.py" "$@"

