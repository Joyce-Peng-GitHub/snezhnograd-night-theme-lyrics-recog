# Snezhnograd Night Theme lyrics recognition

This repository contains a reproducible, local-first pipeline for extracting and
transcribing the Russian choral lyrics in `resources/source.mp4`.

## Pipeline

1. Extract the original AAC track to a lossless 48 kHz stereo WAV:

   ```bash
   scripts/extract_audio.sh
   ```

2. Install the CUDA 12 / Python 3.12 environment:

   ```bash
   uv sync
   ```

3. Produce three vocal candidates with complementary separation settings:

   ```bash
   scripts/separate_vocals.sh
   ```

4. Decode each candidate twice with Whisper large-v3:

   ```bash
   scripts/transcribe.sh work/stems/*.wav
   ```

Models and intermediate candidates are cached under `.cache/` and `work/` and
are intentionally excluded from Git. Final audio and lyrics are stored under
`audio/` and `lyrics/`.

The lossless vocal candidates are committed in `audio/candidates/`. See
`reports/separation.md` for the models, measurements, and checksums.

## Method

The three vocal candidates trade off accompaniment rejection against retention
of quieter choir harmonies. Whisper uses a fixed Russian language token, beam
search, word timestamps, and both context-aware and context-independent decoding.
The final lyrics are selected by agreement between candidates and manual review
of low-confidence passages; singing transcription remains less reliable than
ordinary speech recognition.
