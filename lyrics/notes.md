# Lyrics reconstruction notes

## Status and notation

This remains a working reconstruction, not a verified official lyric. A native
Russian listener's unverified transcript, supplied by the user, provides the
main wording. Local forced alignment confirms that the text fits the recording
closely, but cannot independently prove every word in a dense choral mix.

- Plain text: retained from the human transcript and acoustically corroborated.
- `{A / B}`: close wording variants that the available audio does not resolve.

The left, right, and side channels are nearly decorrelated. Free-running ASR
often invents different words for the same repeated chorus; close-hypothesis
CTC scoring is more reliable here than selecting the most grammatical ASR
output.

## Evidence by passage

| Time | Retained reading | Confidence | Evidence summary |
| --- | --- | --- | --- |
| 14.5-23.7 | `Зори / Пойте мне ночью зимнею / Пойте мне в ночь угрюмую` | Medium | All lines receive coherent XLSR-53 forced times; earlier GigaAM outputs repeatedly retained `по-/пой-`, `ноч-`, and adjective endings. |
| 23.7-35.5 | `Скажите, не тая / У кого же пусты поля / У кого же мои/твои края` | High | Forced alignment is strong, and earlier raw decodes independently retained `у кого`, `пусты`, `мои/твои`, and `края`. |
| 35.5-45.5 | `У кого доля вольная / И жизнь хорошая` | Medium | The human reading explains the previously stable `вольная` and `жизн-/хорош-`-like phonemes. |
| 45.5-50.2 | `Кто о ней / Пел тебе ночью зимнею` | Medium | The supplied wording aligns continuously at 46.6-49.1 seconds and explains the earlier `по/пел тебе ночь...` outputs. |
| 50.2-53.3 | `Пел тебе ночью длинною` | High | `длинною` wins 11 of 16 local CTC comparisons and has the lowest mean and median loss; it also agrees with instrumental `ночью`. |
| 53.3-74.5 | Repeated question and answer lines | Medium-high | The second `Скажите...` and `У кого...` sequence aligns at the same established phrase boundaries; `И жизнь хорошая` is audibly repeated. |
| 103.4-109.6 | `Не хочу забыть её / Пусть некогда прекрасна и сильна` | Medium | The two lines align inside the first paired chorus window. `Пусть...` has the best aggregate wording score, though variants containing `была` remain competitive in the first pass. |
| 109.6-113.3 | `Свободу прославляла {в мир сией / в мире сем}` | Low | The supplied ending dominates the clearer first pass, while the weaker second pass favors alternative morphology. Neither reading is secure enough to collapse. |
| 113.3-118.2 | `Царице той по-прежнему не послужить` | Medium-high | `Царице той` beats `Царице-то` in all 32 narrow windows. `не послужить` wins 20 of 32 ending windows and has the lowest aggregate loss. |
| 118.2-124.4 | `И когда кольцо легло / Когда кольцом заточены сын и дочь` | Medium | The supplied lines fit the paired phrase boundaries and explain earlier `его`, `до конца`, `путь`, and vowel-heavy misdecodes. |
| 124.4-133.0 | `Пусть между нами вновь стоит стена / Но наша песня эту стену сокрушит` | High | Earlier free decodes repeatedly produced `нами`, `истина/стена`, `скорей`, and `сокруши`; the human text resolves those fragments coherently. |
| 133.0-162.6 | Exact chorus repetition | Medium-high | Acoustic self-similarity fixes the repeated form at about 29.54 seconds. The same eight lines fit the paired boundaries through the end of the vocal. |

## Models and checks

- FFmpeg extracted the original 48 kHz stereo PCM track losslessly.
- Three lossless vocal candidates were generated with Roformer-based separation.
- Whisper large-v3 produced Russian subtitle-credit hallucinations and was
  rejected as a lyric source.
- GigaAM variants were run over broad, phrase, channel, refined, and paired
  windows. Their stable fragments corroborate parts of the human text, but
  unrestricted decoding remains unreliable for this choir.
- Acoustic self-similarity across the original mix, three stems, and channel
  variants places the repeated chorus 29.52-29.56 seconds apart.
- Russian XLSR-53 in FP32 scored and force-aligned the supplied transcript on 16
  mix/stem/channel inputs. It was used only as an acoustic model, without its
  language-model decoder, for wording comparisons.
- A CPU/FP32 GigaAM check previously matched 147 of 160 CUDA/FP16 outputs
  exactly, ruling out encoder precision as the limiting factor.
- MMS-1B remained weaker and is retained only as an independent negative
  baseline.

All unedited outputs are under `transcripts/raw/`. Human-hypothesis results are
under `transcripts/raw/ctc-human/`; their manifests are
`config/human-transcript-hypotheses.json`, `config/human-wording-variants.json`,
and `config/human-phonetic-variants.json`. Detailed interpretation is in
`reports/human-transcript-validation.md`.
