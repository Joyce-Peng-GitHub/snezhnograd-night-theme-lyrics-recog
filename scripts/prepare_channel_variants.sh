#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-$repo_dir/work/channel_candidates}"
mkdir -p "$output_dir"

inputs=(
  "$repo_dir/audio/candidates/vocals_full.flac"
  "$repo_dir/audio/candidates/vocals_clean.flac"
  "$repo_dir/audio/candidates/vocals_roformer.flac"
  "$repo_dir/audio/source.wav"
)

for input in "${inputs[@]}"; do
  stem="$(basename -- "${input%.*}")"
  ffmpeg -hide_banner -loglevel error -y -i "$input" \
    -filter:a "pan=mono|c0=c0" -c:a flac "$output_dir/${stem}_left.flac"
  ffmpeg -hide_banner -loglevel error -y -i "$input" \
    -filter:a "pan=mono|c0=c1" -c:a flac "$output_dir/${stem}_right.flac"
  ffmpeg -hide_banner -loglevel error -y -i "$input" \
    -filter:a "pan=mono|c0=0.5*c0-0.5*c1" -c:a flac "$output_dir/${stem}_side.flac"
done

for output in "$output_dir"/*.flac; do
  probe="$(ffprobe -v error \
    -show_entries stream=codec_name,channels:format=duration \
    -of default=noprint_wrappers=1 "$output")"
  grep -qx "codec_name=flac" <<<"$probe"
  grep -qx "channels=1" <<<"$probe"
  duration="$(sed -n 's/^duration=//p' <<<"$probe")"
  awk -v duration="$duration" 'BEGIN { exit !(duration >= 180 && duration < 181) }'
  printf '%s: mono FLAC, %s seconds\n' "$(basename -- "$output")" "$duration"
done
