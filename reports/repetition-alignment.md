# Repetition alignment

The user-supplied structure, that the lyrics from about 01:44 consist of two
repetitions, is strongly supported by the audio. A search over 27-32 second
offsets found the same narrow peak at 29.52-29.56 seconds in the original mix,
all three separated vocal candidates, and the left, right, and side channels.

The search used three complementary frame-level representations: log-mel
energy, MFCCs, and chroma. The median optimum at 20 ms resolution was 29.54
seconds. Similarity rises sharply at about 103.5 seconds and continues through
the corresponding second passage ending near 162.6 seconds.

The paired boundaries in `config/repeated-segments.json` use nearby energy
minima in each performance instead of applying one fixed offset. Corresponding
boundaries differ from the 29.54 second median by at most about 0.1 seconds.
Eight short phrase pairs support boundary-sensitive decoding, while two broad
context pairs preserve enough language context for each half of the passage.

Reproduce the offset search with:

```bash
.venv/bin/python scripts/analyze_repetition.py \
  audio/source.wav \
  audio/candidates/vocals_full.flac \
  audio/candidates/vocals_clean.flac \
  audio/candidates/vocals_roformer.flac
```

This alignment establishes repeated timing and musical material. It does not,
by itself, justify filling a word that no decoder recovers from either pass.
