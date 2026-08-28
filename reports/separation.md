# Vocal separation report

All candidates were generated locally on an NVIDIA RTX 4060 Laptop GPU with
`audio-separator` 0.47.0 and PyTorch 2.6.0+cu124. The input was the extracted
48 kHz stereo PCM track. Outputs are lossless 44.1 kHz, 16-bit stereo FLAC and
remain exactly 180.010680 seconds long.

| Candidate | Separation strategy | Integrated loudness | True peak |
| --- | --- | ---: | ---: |
| `vocals_full.flac` | `vocal_full`: Revive 3e + becruily, max-FFT ensemble | -24.9 LUFS | -9.4 dBFS |
| `vocals_clean.flac` | `vocal_clean`: Revive 2 + Kim FT2 Bleedless, min-FFT ensemble | -26.0 LUFS | -11.5 dBFS |
| `vocals_roformer.flac` | Kimberley Jensen `vocals_mel_band_roformer.ckpt` baseline | -25.7 LUFS | -10.9 dBFS |

The `full` candidate is intended to preserve quieter choir harmonies. The
`clean` candidate rejects more accompaniment. The single-model `roformer`
candidate provides an independent baseline. All three are retained for
cross-decoding because no objective separation score can be computed without
the original studio stems.

## SHA-256

```text
05a69a73023133ed72295457967fbc5d161a0eca1b5ab6494babc578295c70ee  vocals_clean.flac
9215c5a90baae42068681610523f0e5632f81ab8757c0672b7bfbad8c1248eb3  vocals_full.flac
fad8e6552cdad47f1e305f29d54a1b1a755ccd0bdc2b584cc193b774d1bf3fab  vocals_roformer.flac
```
