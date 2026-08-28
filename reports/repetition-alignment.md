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

Unrestricted ASR assigns very different words to the two passes even though the
human transcript identifies an exact chorus repetition. Forced alignment shows
that the human text fits the first pass and its final lines explain the stable
ASR fragments. The same text is therefore mapped onto the second pass using
the acoustic phrase pairs; free-ASR differences are treated as choral decoding
errors rather than evidence for a second lyric.

Reproduce the offset search with:

```bash
.venv/bin/python scripts/analyze_repetition.py \
  audio/source.wav \
  audio/candidates/vocals_full.flac \
  audio/candidates/vocals_clean.flac \
  audio/candidates/vocals_roformer.flac
```

This alignment establishes repeated timing and musical material. The wording is
supported separately by the human transcript and the CTC validation documented
in `reports/human-transcript-validation.md`.

## Full-precision check

The two GigaAM CTC variants were also run with FP32 encoder weights on the CPU
for `vocals_full` and the three most informative right-channel candidates. Of
160 paired-window outputs, 147 were exactly the same as CUDA/FP16. The 13
changes were limited to individual letters, spaces, or short unstable tokens;
none supplied a new defensible lyric. System RAM therefore removes precision
as a plausible bottleneck, but does not resolve the overlapping choir.
