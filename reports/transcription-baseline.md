# Transcription baseline report

## Whisper large-v3

Whisper large-v3 was run on all three vocal candidates with Russian fixed as the
language, beam size 10, and both context-aware and independent decoding. All six
runs produced known subtitle-credit hallucinations such as `Субтитры создавал`
and `Продолжение следует`. The high token probabilities are therefore not valid
confidence estimates for this singing domain. These outputs are retained only
as a documented negative result under `transcripts/raw/whisper/`.

## GigaAM-v3

The Russian-specific `v3_ctc` and `v3_e2e_ctc` models were run on all three
candidates with two overlapping window configurations:

- 24 second windows with a 12 second stride for broad context.
- 12 second windows with a 6 second stride for phrase-level context.

Unlike Whisper, the non-autoregressive GigaAM models generated audio-dependent
phonetic strings with substantial agreement across separation candidates and
overlapping windows. The main vocal regions are approximately 14.8-74.1 seconds
and 100-164 seconds. The raw CTC text is still not a finished transcript: choral
singing, sustained vowels, overlapping voices, and separation artifacts produce
merged or invalid word forms. Final lyrics must be based on cross-model phonetic
agreement rather than grammatical correction of a single output.
