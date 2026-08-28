#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$repo_dir/.venv-xlsr"

uv venv --python 3.12 "$env_dir"
uv pip install \
  --python "$env_dir/bin/python" \
  --torch-backend=cu124 \
  -r "$repo_dir/requirements-xlsr.txt"

"$env_dir/bin/python" -c \
  "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
