# Human transcript validation

## Scope

The user supplied an unverified transcript from a native Russian listener. It
is substantially more coherent than unrestricted ASR and explains many stable
fragments that were previously misassembled. It is still treated as evidence,
not as an official lyric.

## Method

XLSR-53 Russian was run locally in FP32 as a character-level acoustic model.
No language-model decoder was used. The supplied text and close alternatives
were scored and force-aligned over 16 inputs: the original mix, three separated
vocal stems, and their left, right, and side variants.

Whole-section alignment produces plausible word times for both verses and the
first chorus. The final two lines of that chorus also recover the old free-ASR
fragments `нами`, `истина/стена`, and `сокруши`, which provides an independent
cross-check. The second chorus uses the separately measured 29.54-second
acoustic mapping because its quieter, overlapping voices make unconstrained
word timing drift.

## Wording decisions

| Question | Result | Decision |
| --- | --- | --- |
| `ночью длинную / ночью длинною / ночь длинную` | The instrumental adjective wins 11/16 local windows; mean loss per token is 4.307 versus 4.370 and 4.413. | Use `ночью длинною`. |
| `Пусть / Русь` and optional `была` | `Пусть некогда прекрасна и сильна` has the best aggregate mean and wins 13/32 paired windows; forms with `была` win several first-pass stems. | Retain the supplied `Пусть...`; document residual uncertainty. |
| `в мир сией / в мире сем` | The supplied form wins 15/16 clearer first-pass endings, while weak second-pass windows favor other morphology. | Keep both forms as an unresolved alternative. |
| `Царице-то / Царице той` | `Царице той` wins 32/32 isolated noun-phrase windows. | Use `Царице той`. |
| `послужи / послужить / послужим / послужишь` | `не послужить` wins 20/32 windows and has the lowest mean loss, 5.797 versus 6.187 for `не послужи`. | Use `не послужить`. |

CTC loss is most meaningful between near-identical candidates in the same
window. It is not used to infer unconstrained words or to claim that a
grammatical phrase is automatically the sung phrase.

## Reproduction

```bash
HF_HOME="$PWD/.cache/huggingface" TRANSFORMERS_OFFLINE=1 \
  .venv-xlsr/bin/python scripts/score_lyrics_ctc.py \
  --manifest config/human-transcript-hypotheses.json \
  audio/source.wav audio/candidates/*.flac work/channel_candidates/*.flac
```

The committed JSON results are in `transcripts/raw/ctc-human/`.
