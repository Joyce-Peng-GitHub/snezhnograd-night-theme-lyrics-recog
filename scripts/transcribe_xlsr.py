#!/usr/bin/env python3
import argparse
import gc
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
MODEL_REVISION = "2329100508896c6d9b157019803ab5601e6f3406"


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
        if end - start < 1.0:
            break
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


def select_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_model(cache_dir: Path, device: str) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        use_safetensors=False,
    )
    model.eval()
    return model.to(device)


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


def decode_files(audio_files: list[Path], args: argparse.Namespace, device: str) -> None:
    print(f"Loading {MODEL_ID}@{MODEL_REVISION} on {device} in FP32...", flush=True)
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.model_dir,
    )
    model = load_model(args.model_dir, device)

    for audio in audio_files:
        chunks = make_chunks(audio, args.chunk_dir, args.window, args.stride)
        payload = {
            "audio": str(audio),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": device,
            "dtype": "float32",
            "window_seconds": args.window,
            "stride_seconds": args.stride,
            "chunks": [],
        }
        for index, chunk in enumerate(chunks, start=1):
            waveform, sample_rate = read_pcm16_mono(chunk["path"])
            if sample_rate != 16000:
                raise RuntimeError(f"Unexpected chunk format: {chunk['path']}")
            inputs = processor.feature_extractor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.inference_mode():
                logits = model(input_values, attention_mask=attention_mask).logits[0].cpu()

            greedy_ids = torch.argmax(logits, dim=-1)
            greedy = processor.tokenizer.decode(greedy_ids).strip().lower()
            beam = processor.decoder.decode(
                logits.numpy(),
                beam_width=args.beam_width,
            ).strip().lower()
            item = {
                "index": index,
                "start": chunk["start"],
                "end": chunk["end"],
                "greedy": greedy,
                "beam_lm": beam,
            }
            payload["chunks"].append(item)
            print(
                f"{audio.stem} {chunk['start']:6.1f}-{chunk['end']:6.1f} "
                f"greedy={greedy!r} beam_lm={beam!r}",
                flush=True,
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{audio.stem}.xlsr53-ru"
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (args.output_dir / f"{stem}.txt").open("w", encoding="utf-8") as output:
            for chunk in payload["chunks"]:
                span = f"[{chunk['start']:06.1f}-{chunk['end']:06.1f}]"
                output.write(f"{span} greedy: {chunk['greedy']}\n")
                output.write(f"{span} beam_lm: {chunk['beam_lm']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-decode Russian singing with XLSR-53")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--window", type=float, default=12.0)
    parser.add_argument("--stride", type=float, default=6.0)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--chunk-dir", type=Path, default=Path("work/xlsr_chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/xlsr_transcripts"))
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/huggingface"))
    args = parser.parse_args()

    if not 0 < args.stride <= args.window:
        parser.error("stride must be greater than zero and no larger than window")
    if args.beam_width < 1:
        parser.error("beam width must be positive")

    device = select_device(args.device)
    try:
        decode_files(args.audio, args, device)
    except torch.OutOfMemoryError:
        if args.device != "auto" or device != "cuda":
            raise
        print("CUDA ran out of memory; reloading the same FP32 model in system RAM...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        decode_files(args.audio, args, "cpu")


if __name__ == "__main__":
    main()
