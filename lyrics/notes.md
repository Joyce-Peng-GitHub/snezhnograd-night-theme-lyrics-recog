# Lyrics reconstruction notes

## Status and notation

This is a conservative working reconstruction, not a verified official lyric.
The choir, sustained vowels, overlapping parts, accompaniment, and stereo phase
differences prevent a reliable word-for-word transcript throughout.

- Plain text: supported across multiple models, windows, and input variants.
- `{candidate}`: a phonetically coherent line, but not every word is independently
  stable.
- `{A / B}`: the acoustic evidence does not distinguish the alternatives.
- `[неразборчиво]`: the words cannot be recovered reliably.
- `[неразборчиво: «...root...»]`: only the shown word or root is repeatable.

The left, right, and side channels are nearly decorrelated and sometimes yield
different words at the same time. This may reflect overlapping choir parts, so
a single linear lyric can itself be an approximation.

## Evidence by passage

| Time | Retained reading | Confidence | Evidence summary |
| --- | --- | --- | --- |
| 14.5-23.7 | `Под синей ночью ... / ... лунною` | Low | Repeated `синей`, `ночь`, `тебе`, and `пою`-like phonemes; full grammar is inferred. |
| 23.7-28.1 | `Сады снегом укрывая` | Medium | Three aligned groups consistently resemble `сады / снегом / укрывая`; decoders often merge them as `Садись ... края`. |
| 28.1-35.5 | `Укрой, Боже, Русь мою / ... мои края` | High | `Боже`, `Русь/русь`, `мою/моя`, and `мои края` recur across separated and original audio. |
| 35.5-45.5 | `Русь моя привольная ... чистая душою` | Low | Stable roots resemble `привольная/вольная`, `чисто-`, and `душа`; morphology and intervening words are unresolved. |
| 45.5-50.2 | `Пою тебе ночь синюю` | Medium | Most GigaAM variants emit `по тебе ночь синую`; `Пою` is a phonetic reconstruction. |
| 50.2-53.3 | `Пою тебе ночь {длинную / милую}` | Medium | `по тебе ночь` is stable; the final adjective alternates between `длинную`- and `милую`-like forms. |
| 53.3-65.0 | Repetition of snow/protection lines | Medium | Repeated structure plus `Боже`, `Русь`, and `мои/твои края`; the snow line remains a candidate. |
| 65.0-74.5 | `... привольная ... чистая душою ... вечно славящая` | Low | Same roots as 35.5-45.5; the ending repeatedly resembles `вечно славящая` but is not exact. |
| 99.5-103.4 | Unresolved | Very low | This lead-in falls before the acoustically aligned repeated form. |
| 103.4-118.2 | `... жили ... души` | Low | `жили` appears in 8 paired short-window decodes; `душ-` appears in 25 of 32 broad-window decodes. The intervening words remain unstable. |
| 118.2-124.4 | `... его ... до конца ... путь` | Medium | `его` appears in 7 short-window decodes; `до конца` and `путь` each appear in 8 broad-window decodes. |
| 124.4-128.1 | `Пусть ... но истина` | Medium | The broad decodes repeatedly supply this frame, while `истин-` appears in 11 paired short and 11 broad outputs. |
| 128.1-133.0 | `... скорей ... сокруши` | High | `скор-` appears in 17 short-window outputs and `сокруш-/согруш-` in 18 broad-window outputs. |
| 133.0-136.5 | `Наша победа, наша победа` | High | `побед-` appears in 19 of 32 paired short-window outputs and 14 broad-window outputs. |
| 136.5-139.1 | `Любовь на свете ...` | Low | Several right-channel and broad E2E decodes align all three words here; the line ending is unstable. |
| 139.1-142.8 | Unresolved | Very low | Outputs disagree and no content word survives enough independent variants. |
| 142.8-147.8 | `Отечество люби ... ненавид...` | Low | Four broad-window variants retain each of `Отечеств-` and `ненавид-`; two independently retain `Отечество люби`. |
| 147.8-153.9 | `... разруш... сердц... душ...` | Very low | Broad paired windows retain these roots, but do not establish dependable syntax. |
| 153.9-157.5 | `... его ... имя ...` | Low | `его` appears in 10 broad outputs and 4 short outputs; `имя` appears in 3 short and 2 broad outputs. |
| 157.5-162.6 | `... священн... душ...` | Very low | These roots recur in broad windows, while short windows remain mostly undecodable. |

## Models and checks

- FFmpeg extracted the original 48 kHz stereo PCM track losslessly.
- Three lossless vocal candidates were generated with Roformer-based separation.
- Whisper large-v3 produced known Russian subtitle-credit hallucinations and was
  rejected as a lyric source.
- GigaAM `v3_ctc`, `v3_e2e_ctc`, both RNNT variants, and
  `multilingual_large_ctc` were run with broad, overlapping, phrase, channel,
  and refined-boundary inputs. The Russian CTC models were the most useful.
- Acoustic self-similarity places the two final musical passes 29.54 seconds
  apart. Twenty paired phrase/context windows were decoded over 16 mix, stem,
  and channel inputs. The corresponding windows support the same metrical
  structure but clearly different content words, so text was not copied from
  one pass to the other.
- A CPU/FP32 GigaAM check matched 147 of 160 CUDA/FP16 window outputs exactly.
  The remaining changes were minor and supplied no new defensible words, so
  encoder precision is not the limiting factor here.
- Russian XLSR-53 was decoded both greedily and with its KenLM; it was weaker
  than GigaAM and the language model over-corrected unstable acoustics.
- MMS-1B with the Russian adapter was run in full FP32. It was also weaker than
  GigaAM and is retained only as an independent negative baseline.

All unedited model outputs are under `transcripts/raw/`. The segment manifests
are `config/phrase-segments.json`, `config/refined-segments.json`, and
`config/repeated-segments.json`. The repetition analysis is documented in
`reports/repetition-alignment.md`.
