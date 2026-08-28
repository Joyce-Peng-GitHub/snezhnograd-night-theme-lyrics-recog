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

Three additional GigaAM models were decoded with 12 second windows and a 6
second stride. `v3_rnnt` and `v3_e2e_rnnt` were substantially more conservative
than CTC on the choir; their sparse output is retained as negative evidence.
The 2.18 GB `multilingual_large_ctc` checkpoint agreed with the Russian CTC
models on much of the first vocal passage, including phonetic support for
`по тебе ночь синую`, repeated invocations containing `Боже` and `мои края`,
and a closing description resembling `... и честь хорошая`. It emitted little
for the denser second passage except `сокруши` and fragments ending in `души`.
All raw outputs are archived under `transcripts/raw/gigaam-extra/`; these
phonetic readings are evidence, not editorially corrected lyrics.

## XLSR-53 Russian

For an architecture-independent check, the Apache-2.0 Russian XLSR-53 model
`jonatasgrosman/wav2vec2-large-xlsr-53-russian` was pinned to revision
`2329100508896c6d9b157019803ab5601e6f3406`. The unquantized FP32 checkpoint was
run on all three vocal candidates with the same 12/6 second window scheme. Both
greedy CTC and beam search with the model's 1.49 GB KenLM were retained.

XLSR was substantially less intelligible on this singing domain than GigaAM.
Agreement between candidates provides limited independent support for phonetic
fragments such as `по тебе`; most beam outputs are language-model rewrites of
unstable acoustic tokens and must not be treated as lyrics. The complete raw
greedy and beam output is stored under `transcripts/raw/xlsr-12s/`.

## Original mix cross-check

The two Russian GigaAM CTC models and `multilingual_large_ctc` were also run
directly on the unseparated `audio/source.wav`. This tests whether vocal
separation removed consonant transients along with the accompaniment. The
original mix improved agreement on several first-passage fragments, especially
`Боже`, `мои края`, `по тебе`, and `ночь синую`; it was less useful for the
denser second passage. Results are stored under
`transcripts/raw/gigaam-source/` and are used alongside, not in place of, the
three separated candidates.

## Phrase-boundary decoding

Energy minima and repeated word alignments from the broad passes were used to
define 23 phrase segments plus 8 wider context segments in
`config/phrase-segments.json`. Both Russian GigaAM CTC models decoded all
segments from the three vocal candidates and the original mix. The targeted
windows reduced cross-line token merging and strengthened agreement on the
second passage, including `до конца`, `путь`, `истина`, `скорей`, `сокруши`,
and `наша победа`. Later fragments repeatedly resemble roots from `Отечество`,
`любви`, `души`, and `священный`, but their surrounding words remain unstable.
The 248 raw phrase results are archived under
`transcripts/raw/gigaam-phrases/`; no grammatical repair has been applied.

## Channel-layer decoding

The separated vocal candidates have nearly decorrelated stereo channels, so
mono downmix can cancel or merge independent choir layers. Lossless left,
right, and side variants of all three candidates and the original mix were
decoded against the same phrase manifest with both Russian CTC models. This
produced 744 additional model/input/segment readings. The right and side layers
added repeated support for phrases including `Русь моя`, `мои края`,
`ночь милую`, `до конца`, `синий путь`, `наша победа`, `на нашей земле`, and
`мою душу`. These are still evaluated by cross-input agreement; a plausible
word from one channel alone is not accepted. Raw results are stored under
`transcripts/raw/gigaam-channels/`.

## MMS-1B Russian adapter

Meta's `facebook/mms-1b-all` was pinned to revision
`3d33597edbdaaba14a8e858e2c8caa76e3cec0cd` and loaded in unquantized FP32 with
its `rus` adapter. The billion-parameter model decoded 217 phrase/input pairs
from the full vocal candidate and its channel layers plus the original-mix
layers. It fit on the RTX 4060 without a RAM fallback, although the script
supports running the same FP32 checkpoint on CPU.

MMS was markedly weaker than GigaAM on this choir and sometimes emitted invalid
characters. It independently reproduced a few roots such as `Боже`, `ночь`,
and `сокруши`, but cannot support complete lines. Its raw output is retained as
a documented weak cross-architecture baseline under
`transcripts/raw/mms-phrases/`. The model is CC-BY-NC 4.0, unlike the
Apache-2.0 XLSR checkpoint.

## Refined acoustic boundaries

The remaining wide uncertain phrases contain multiple short-time-energy minima.
`config/refined-segments.json` splits only those regions at the measured minima,
without stretching, pitch shifting, or otherwise modifying the audio. This
final targeted pass tests whether earlier invalid word forms came from joining
adjacent sung lines.
