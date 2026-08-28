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
| 99.5-109.5 | Unresolved | Very low | Outputs disagree completely between mix and vocal layers. |
| 109.5-118.2 | `... души` | Medium | `души` recurs at the phrase end across mono, right, and side layers. |
| 118.2-124.4 | `... до конца ... путь` | Medium | `до конца` and `путь` recur; the rest of the line is unstable. |
| 124.4-128.1 | `Пусть ... но истина` | Medium | `Пусть` and `истина` recur across phrase, context, and refined windows. |
| 128.1-133.0 | `... скорей ... сокруши` | High | Both words recur across Russian CTC models, vocal candidates, and channel layers. |
| 133.0-137.2 | `Наша победа, наша победа` | High | The repeated phrase is stable in right-channel and broad-context decodes. |
| 137.2-144.7 | `... Отечество ...` | Low | Only the `Отечеств-` root survives in several right-channel context/refined outputs. |
| 144.7-153.9 | `... ненавид... разруш...` | Very low | Only these roots repeat; no dependable syntax remains. |
| 153.9-162.5 | `... его ... имя ... священн...` | Very low | Broad windows suggest these roots, while short windows mostly reject the passage. |

## Models and checks

- FFmpeg extracted the original 48 kHz stereo PCM track losslessly.
- Three lossless vocal candidates were generated with Roformer-based separation.
- Whisper large-v3 produced known Russian subtitle-credit hallucinations and was
  rejected as a lyric source.
- GigaAM `v3_ctc`, `v3_e2e_ctc`, both RNNT variants, and
  `multilingual_large_ctc` were run with broad, overlapping, phrase, channel,
  and refined-boundary inputs. The Russian CTC models were the most useful.
- Russian XLSR-53 was decoded both greedily and with its KenLM; it was weaker
  than GigaAM and the language model over-corrected unstable acoustics.
- MMS-1B with the Russian adapter was run in full FP32. It was also weaker than
  GigaAM and is retained only as an independent negative baseline.

All unedited model outputs are under `transcripts/raw/`. The segment manifests
are `config/phrase-segments.json` and `config/refined-segments.json`.
