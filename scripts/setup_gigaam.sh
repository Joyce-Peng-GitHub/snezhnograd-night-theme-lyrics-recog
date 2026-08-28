#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$repo_dir/.venv-gigaam"

uv venv --python 3.12 "$env_dir"
uv pip install \
  --python "$env_dir/bin/python" \
  --torch-backend=cu124 \
  -r "$repo_dir/requirements-gigaam.txt"

"$env_dir/bin/python" -c \
  "import torch, gigaam; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

