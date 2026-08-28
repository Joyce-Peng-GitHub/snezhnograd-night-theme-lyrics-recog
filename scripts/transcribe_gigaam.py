#!/usr/bin/env python3
import argparse
import gc
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import gigaam
import torch


def duration(audio: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def make_chunks(audio: Path, chunk_dir: Path, window: float, stride: float) -> list[dict]:
    audio_duration = duration(audio)
    chunks = []
    start = 0.0
    while start < audio_duration:
        end = min(start + window, audio_duration)
        chunk = chunk_dir / audio.stem / f"{start:06.1f}.wav"
        chunk.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(audio),
                "-t",
                f"{end - start:.3f}",
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(chunk),
            ],
            check=True,
        )
        chunks.append({"start": start, "end": end, "path": chunk})
        start += stride
    return chunks


def run_model(model_name: str, audio_files: list[Path], args: argparse.Namespace) -> None:
    print(f"Loading {model_name}...", flush=True)
    model = gigaam.load_model(
        model_name,
        device="cuda",
        fp16_encoder=True,
        download_root=str(args.model_dir),
    )
    for audio in audio_files:
        chunks = make_chunks(audio, args.chunk_dir, args.window, args.stride)
        payload = {
            "audio": str(audio),
            "model": model_name,
            "window_seconds": args.window,
            "stride_seconds": args.stride,
            "chunks": [],
        }
        for index, chunk in enumerate(chunks, start=1):
            result = model.transcribe(str(chunk["path"]), word_timestamps=True)
            words = []
            for word in result.words or []:
                item = asdict(word)
                item["start"] = round(item["start"] + chunk["start"], 3)
                item["end"] = round(item["end"] + chunk["start"], 3)
                words.append(item)
            payload["chunks"].append(
                {
                    "index": index,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": result.text.strip(),
                    "words": words,
                }
            )
            print(
                f"{model_name} {audio.stem} {chunk['start']:6.1f}-{chunk['end']:6.1f}: "
                f"{result.text.strip()}",
                flush=True,
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{audio.stem}.{model_name}"
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (args.output_dir / f"{stem}.txt").open("w", encoding="utf-8") as output:
            for chunk in payload["chunks"]:
                output.write(
                    f"[{chunk['start']:06.1f}-{chunk['end']:06.1f}] {chunk['text']}\n"
                )

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-decode Russian singing with GigaAM")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="model to run; repeat for multiple models (default: v3_ctc and v3_e2e_ctc)",
    )
    parser.add_argument("--window", type=float, default=24.0)
    parser.add_argument("--stride", type=float, default=12.0)
    parser.add_argument("--chunk-dir", type=Path, default=Path("work/gigaam_chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/gigaam_transcripts"))
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/gigaam"))
    args = parser.parse_args()

    if args.window >= 25:
        parser.error("GigaAM short-form windows must be shorter than 25 seconds")
    if not 0 < args.stride <= args.window:
        parser.error("stride must be greater than zero and no larger than window")

    for model_name in args.models or ["v3_ctc", "v3_e2e_ctc"]:
        run_model(model_name, args.audio, args)


if __name__ == "__main__":
    main()
