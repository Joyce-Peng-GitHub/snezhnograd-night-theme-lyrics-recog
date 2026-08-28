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


def write_chunk(audio: Path, chunk: Path, start: float, end: float) -> None:
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


def make_chunks(audio: Path, chunk_dir: Path, window: float, stride: float) -> list[dict]:
    audio_duration = duration(audio)
    chunks = []
    start = 0.0
    while start < audio_duration:
        end = min(start + window, audio_duration)
        if end - start < 1.0:
            break
        chunk = chunk_dir / audio.stem / f"{start:06.1f}.wav"
        write_chunk(audio, chunk, start, end)
        chunks.append({"start": start, "end": end, "path": chunk})
        start += stride
    return chunks


def load_segments(manifest: Path, audio_duration: float) -> list[dict]:
    segments = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise ValueError("segment manifest must contain a non-empty JSON list")
    seen = set()
    for segment in segments:
        segment_id = segment.get("id")
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen:
            raise ValueError(f"invalid or duplicate segment id: {segment_id!r}")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"segment {segment_id} must have numeric bounds")
        if start < 0 or end > audio_duration or not 1 <= end - start < 25:
            raise ValueError(f"segment {segment_id} must be 1-25 seconds within the audio")
        seen.add(segment_id)
    return segments


def make_segment_chunks(audio: Path, chunk_dir: Path, manifest: Path) -> list[dict]:
    segments = load_segments(manifest, duration(audio))
    chunks = []
    for segment in segments:
        chunk = chunk_dir / audio.stem / f"{segment['id']}.wav"
        write_chunk(audio, chunk, segment["start"], segment["end"])
        chunks.append({**segment, "path": chunk})
    return chunks


def run_model(model_name: str, audio_files: list[Path], args: argparse.Namespace) -> None:
    print(f"Loading {model_name}...", flush=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model = gigaam.load_model(
        model_name,
        device=device,
        fp16_encoder=not args.fp32_encoder,
        download_root=str(args.model_dir),
    )
    for audio in audio_files:
        if args.segment_manifest:
            chunks = make_segment_chunks(audio, args.chunk_dir, args.segment_manifest)
        else:
            chunks = make_chunks(audio, args.chunk_dir, args.window, args.stride)
        payload = {
            "audio": str(audio),
            "model": model_name,
            "device": device,
            "encoder_precision": "fp32" if args.fp32_encoder else "fp16",
            "window_seconds": None if args.segment_manifest else args.window,
            "stride_seconds": None if args.segment_manifest else args.stride,
            "segment_manifest": str(args.segment_manifest) if args.segment_manifest else None,
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
            item = {
                "index": index,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": result.text.strip(),
                "words": words,
            }
            if "id" in chunk:
                item["id"] = chunk["id"]
                item["kind"] = chunk.get("kind")
            payload["chunks"].append(item)
            print(
                f"{model_name} {audio.stem} {chunk.get('id', ''):6} "
                f"{chunk['start']:6.1f}-{chunk['end']:6.1f}: "
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
                label = f" {chunk['id']}" if "id" in chunk else ""
                text = f" {chunk['text']}" if chunk["text"] else ""
                output.write(
                    f"[{chunk['start']:06.1f}-{chunk['end']:06.1f}{label}]{text}\n"
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
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        help="JSON list of named start/end segments; replaces regular windows",
    )
    parser.add_argument("--chunk-dir", type=Path, default=Path("work/gigaam_chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/gigaam_transcripts"))
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/gigaam"))
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--fp32-encoder",
        action="store_true",
        help="keep encoder weights in FP32; useful for CPU verification",
    )
    args = parser.parse_args()

    if args.window >= 25:
        parser.error("GigaAM short-form windows must be shorter than 25 seconds")
    if not 0 < args.stride <= args.window:
        parser.error("stride must be greater than zero and no larger than window")

    for model_name in args.models or ["v3_ctc", "v3_e2e_ctc"]:
        run_model(model_name, args.audio, args)


if __name__ == "__main__":
    main()
