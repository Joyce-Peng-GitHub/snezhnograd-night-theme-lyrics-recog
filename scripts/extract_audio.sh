#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$repo_dir/resources/source.mp4}"
output="${2:-$repo_dir/audio/source.wav}"

mkdir -p "$(dirname -- "$output")"
ffmpeg -hide_banner -y -i "$input" \
  -map 0:a:0 -vn \
  -c:a pcm_s16le -ar 48000 -ac 2 \
  "$output"

