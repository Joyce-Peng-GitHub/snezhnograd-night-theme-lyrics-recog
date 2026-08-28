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

5. Cross-check with the Russian-specific GigaAM-v3 CTC models, whose training
   data includes music and atypical speech:

   ```bash
   scripts/setup_gigaam.sh
   scripts/transcribe_gigaam.sh audio/candidates/*.flac
   ```

   For phrase-boundary decoding after the broad pass:

   ```bash
   scripts/transcribe_gigaam.sh \
     --segment-manifest config/phrase-segments.json \
     audio/candidates/*.flac
   ```

   The passage beginning near 01:44 repeats after about 29.54 seconds. Reproduce
   the acoustic offset analysis and decode both performances at paired phrase
   boundaries with:

   ```bash
   .venv/bin/python scripts/analyze_repetition.py \
     audio/source.wav audio/candidates/*.flac
   scripts/transcribe_gigaam.sh \
     --segment-manifest config/repeated-segments.json \
     audio/candidates/*.flac
   ```

   The separated stems retain decorrelated left and right choir layers. Decode
   lossless left, right, and side variants when checking uncertain phrases:

   ```bash
   scripts/prepare_channel_variants.sh
   scripts/transcribe_gigaam.sh \
     --segment-manifest config/phrase-segments.json \
     work/channel_candidates/*.flac
   ```

6. Independently cross-check with the Russian XLSR-53 CTC model, preserving
   both raw greedy decoding and a KenLM beam-search result:

   ```bash
   scripts/setup_xlsr.sh
   scripts/transcribe_xlsr.sh audio/candidates/*.flac
   ```

   The model runs unquantized in FP32. `--device auto` falls back from CUDA to
   system RAM if GPU memory is exhausted; use `--device cpu` to select that path
   explicitly.

7. For an independent billion-parameter acoustic check, decode phrase segments
   with Meta MMS-1B and its Russian adapter:

   ```bash
   scripts/transcribe_mms.sh audio/candidates/vocals_full.flac
   ```

   MMS-1B is loaded unquantized in FP32 and can fall back to system RAM. Its
   CC-BY-NC 4.0 license permits this local non-commercial analysis but is more
   restrictive than the Apache-2.0 XLSR checkpoint.

Models and intermediate candidates are cached under `.cache/` and `work/` and
are intentionally excluded from Git. Final audio and lyrics are stored under
`audio/` and `lyrics/`.

The final deliverables are `lyrics/lyrics.ru.txt`, synchronized
`lyrics/lyrics.ru.srt`, and the confidence/evidence record in
`lyrics/notes.md`. Braces and explicit `[неразборчиво]` markers are retained
because this recording does not support a defensible complete transcript.

The lossless vocal candidates are committed in `audio/candidates/`. See
`reports/separation.md` for the models, measurements, and checksums.

## Method

The three vocal candidates trade off accompaniment rejection against retention
of quieter choir harmonies. Whisper uses a fixed Russian language token, beam
search, word timestamps, and both context-aware and context-independent decoding.
The final lyrics are selected by agreement between candidates and manual review
of low-confidence passages; singing transcription remains less reliable than
ordinary speech recognition.
