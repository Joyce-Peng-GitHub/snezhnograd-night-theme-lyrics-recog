#!/usr/bin/env python3
import argparse
import gc
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL_ID = "facebook/mms-1b-all"
MODEL_REVISION = "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
TARGET_LANGUAGE = "rus"


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


def load_segments(manifest: Path, audio_duration: float) -> list[dict]:
    segments = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        raise ValueError("segment manifest must contain a non-empty JSON list")
    for segment in segments:
        start, end = segment.get("start"), segment.get("end")
        if not isinstance(segment.get("id"), str):
            raise ValueError("every segment must have a string id")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"segment {segment.get('id')} must have numeric bounds")
        if start < 0 or end > audio_duration or end - start < 1:
            raise ValueError(f"segment {segment['id']} is outside the audio")
    return segments


def make_chunks(audio: Path, chunk_dir: Path, manifest: Path) -> list[dict]:
    chunks = []
    for segment in load_segments(manifest, duration(audio)):
        chunk = chunk_dir / audio.stem / f"{segment['id']}.wav"
        chunk.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{segment['start']:.3f}",
                "-i",
                str(audio),
                "-t",
                f"{segment['end'] - segment['start']:.3f}",
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
        chunks.append({**segment, "path": chunk})
    return chunks


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if channels != 1 or sample_width != 2:
        raise RuntimeError(f"Unexpected chunk format: {path}")
    waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return waveform, sample_rate


def load_model(cache_dir: Path, device: str) -> tuple[object, Wav2Vec2ForCTC]:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        target_lang=TARGET_LANGUAGE,
        cache_dir=cache_dir,
    )
    processor.tokenizer.set_target_lang(TARGET_LANGUAGE)
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model.load_adapter(
        TARGET_LANGUAGE,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        use_safetensors=True,
    )
    model.eval()
    return processor, model.to(device)


def decode(audio_files: list[Path], args: argparse.Namespace, device: str) -> None:
    print(
        f"Loading {MODEL_ID}@{MODEL_REVISION} language={TARGET_LANGUAGE} "
        f"on {device} in FP32...",
        flush=True,
    )
    processor, model = load_model(args.model_dir, device)
    for audio in audio_files:
        payload = {
            "audio": str(audio),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "language_adapter": TARGET_LANGUAGE,
            "device": device,
            "dtype": "float32",
            "segment_manifest": str(args.segment_manifest),
            "chunks": [],
        }
        for index, chunk in enumerate(
            make_chunks(audio, args.chunk_dir, args.segment_manifest), start=1
        ):
            waveform, sample_rate = read_pcm16_mono(chunk["path"])
            inputs = processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = model(inputs.input_values.to(device)).logits
            text = processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip().lower()
            item = {
                "index": index,
                "id": chunk["id"],
                "kind": chunk.get("kind"),
                "start": chunk["start"],
                "end": chunk["end"],
                "text": text,
            }
            payload["chunks"].append(item)
            print(
                f"{audio.stem} {chunk['id']:6} "
                f"{chunk['start']:6.1f}-{chunk['end']:6.1f}: {text}",
                flush=True,
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{audio.stem}.mms-1b-rus"
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (args.output_dir / f"{stem}.txt").open("w", encoding="utf-8") as output:
            for chunk in payload["chunks"]:
                output.write(
                    f"[{chunk['start']:06.1f}-{chunk['end']:06.1f} {chunk['id']}] "
                    f"{chunk['text']}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode Russian singing with MMS-1B")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        default=Path("config/phrase-segments.json"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--chunk-dir", type=Path, default=Path("work/mms_chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/mms_transcripts"))
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/huggingface"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    try:
        decode(args.audio, args, device)
    except torch.OutOfMemoryError:
        if args.device != "auto" or device != "cuda":
            raise
        print("CUDA ran out of memory; reloading the same FP32 model in RAM...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        decode(args.audio, args, "cpu")


if __name__ == "__main__":
    main()
