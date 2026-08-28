#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def transcribe(model: WhisperModel, audio: Path, output_dir: Path, mode: str) -> None:
    condition_on_previous_text = mode == "coherent"
    segments_iter, info = model.transcribe(
        str(audio),
        language="ru",
        task="transcribe",
        beam_size=10,
        patience=2.0,
        temperature=0.0,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=True,
        vad_filter=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    segments = list(segments_iter)
    stem = f"{audio.stem}.{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "audio": str(audio),
        "mode": mode,
        "model": "large-v3",
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "compression_ratio": segment.compression_ratio,
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in (segment.words or [])
                ],
            }
            for segment in segments
        ],
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"{stem}.txt").write_text(
        "\n".join(segment.text.strip() for segment in segments) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{stem}.srt").open("w", encoding="utf-8") as output:
        for index, segment in enumerate(segments, start=1):
            output.write(
                f"{index}\n{timestamp(segment.start)} --> {timestamp(segment.end)}\n"
                f"{segment.text.strip()}\n\n"
            )

    mean_logprob = sum(segment.avg_logprob for segment in segments) / max(len(segments), 1)
    print(f"{audio.name} [{mode}]: {len(segments)} segments, mean logprob {mean_logprob:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe Russian singing with Whisper large-v3")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("work/transcripts"))
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    model = WhisperModel("large-v3", device="cuda", compute_type=args.compute_type)
    for audio in args.audio:
        for mode in ("coherent", "independent"):
            transcribe(model, audio, args.output_dir, mode)


if __name__ == "__main__":
    main()

