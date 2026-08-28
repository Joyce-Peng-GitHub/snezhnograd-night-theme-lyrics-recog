#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def normalized_columns(feature: np.ndarray) -> np.ndarray:
    feature = (feature - feature.mean(axis=1, keepdims=True)) / (
        feature.std(axis=1, keepdims=True) + 1e-8
    )
    return feature / (np.linalg.norm(feature, axis=0, keepdims=True) + 1e-8)


def extract_features(audio: Path, sample_rate: int, hop_length: int) -> dict[str, np.ndarray]:
    samples, _ = librosa.load(audio, sr=sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=sample_rate,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=64,
        fmin=80,
        fmax=7600,
        power=2,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return {
        "logmel": normalized_columns(logmel),
        "mfcc": normalized_columns(librosa.feature.mfcc(S=logmel, n_mfcc=20)[1:]),
        "chroma": normalized_columns(
            librosa.feature.chroma_cqt(
                y=samples,
                sr=sample_rate,
                hop_length=hop_length,
                bins_per_octave=24,
                n_chroma=12,
            )
        ),
    }


def score_lags(
    feature: np.ndarray,
    start: float,
    end: float,
    lag_min: float,
    lag_max: float,
    sample_rate: int,
    hop_length: int,
) -> tuple[float, float]:
    start_frame = round(start * sample_rate / hop_length)
    end_frame = round(end * sample_rate / hop_length)
    best_score = -float("inf")
    best_lag = lag_min
    for lag_frames in range(
        round(lag_min * sample_rate / hop_length),
        round(lag_max * sample_rate / hop_length) + 1,
    ):
        first = feature[:, start_frame:end_frame]
        second = feature[:, start_frame + lag_frames : end_frame + lag_frames]
        if first.shape != second.shape:
            continue
        score = float(np.sum(first * second, axis=0).mean())
        if score > best_score:
            best_score = score
            best_lag = lag_frames * hop_length / sample_rate
    return best_lag, best_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate a repeated musical passage's offset from acoustic features"
    )
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--start", type=float, default=103.5)
    parser.add_argument("--end", type=float, default=132.9)
    parser.add_argument("--lag-min", type=float, default=27.0)
    parser.add_argument("--lag-max", type=float, default=32.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not 0 <= args.start < args.end:
        parser.error("start must be non-negative and earlier than end")
    if not 0 < args.lag_min < args.lag_max:
        parser.error("lag bounds must be positive and ordered")

    measurements = []
    for audio in args.audio:
        for feature_name, feature in extract_features(
            audio, args.sample_rate, args.hop_length
        ).items():
            lag, score = score_lags(
                feature,
                args.start,
                args.end,
                args.lag_min,
                args.lag_max,
                args.sample_rate,
                args.hop_length,
            )
            measurements.append(
                {
                    "audio": str(audio),
                    "feature": feature_name,
                    "best_lag_seconds": round(lag, 3),
                    "mean_cosine_similarity": round(score, 6),
                }
            )

    consensus = float(np.median([item["best_lag_seconds"] for item in measurements]))
    payload = {
        "search": {
            "first_passage_start": args.start,
            "first_passage_end": args.end,
            "lag_min": args.lag_min,
            "lag_max": args.lag_max,
            "resolution_seconds": args.hop_length / args.sample_rate,
        },
        "consensus_lag_seconds": round(consensus, 3),
        "measurements": measurements,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
