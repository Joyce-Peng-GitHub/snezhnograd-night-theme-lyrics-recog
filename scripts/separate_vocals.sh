#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
separator="$repo_dir/.venv/bin/audio-separator"
input="${1:-$repo_dir/audio/source.wav}"
output_dir="${2:-$repo_dir/work/stems}"
model_dir="$repo_dir/.cache/audio-separator"

if [[ ! -x "$separator" ]]; then
  echo "Missing environment. Run: uv sync" >&2
  exit 1
fi

mkdir -p "$output_dir" "$model_dir"

common_args=(
  "$input"
  --output_dir "$output_dir"
  --model_file_dir "$model_dir"
  --output_format WAV
  --sample_rate 44100
  --single_stem Vocals
  --use_autocast
)

"$separator" "${common_args[@]}" \
  --ensemble_preset vocal_full \
  --custom_output_names '{"Vocals":"vocals_full"}'

"$separator" "${common_args[@]}" \
  --ensemble_preset vocal_clean \
  --custom_output_names '{"Vocals":"vocals_clean"}'

"$separator" "${common_args[@]}" \
  --model_filename vocals_mel_band_roformer.ckpt \
  --custom_output_names '{"Vocals":"vocals_roformer"}'

