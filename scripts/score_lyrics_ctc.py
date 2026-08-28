#!/usr/bin/env python3
import argparse
import gc
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from transcribe_xlsr import MODEL_ID, MODEL_REVISION


def normalize(text: str) -> str:
    text = re.sub(r"[^а-яё -]+", " ", text.lower())
    text = re.sub(r"(?<![а-яё])-(?![а-яё])", " ", text)
    return " ".join(text.split())


def validate_manifest(path: Path) -> list[dict]:
    hypotheses = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("manifest must contain a non-empty JSON list")
    for hypothesis in hypotheses:
        if not isinstance(hypothesis.get("id"), str):
            raise ValueError("every hypothesis needs a string id")
        start, end = hypothesis.get("start"), hypothesis.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"{hypothesis['id']} needs numeric bounds")
        if start < 0 or not 1 <= end - start <= 35:
            raise ValueError(f"invalid bounds for {hypothesis['id']}")
        variants = hypothesis.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"{hypothesis['id']} needs at least one variant")
        for variant in variants:
            if not isinstance(variant.get("id"), str):
                raise ValueError(f"a variant in {hypothesis['id']} has no string id")
            if not isinstance(variant.get("lines"), list) or not variant["lines"]:
                raise ValueError(f"{hypothesis['id']}/{variant['id']} has no lines")
            if not all(isinstance(line, str) and normalize(line) for line in variant["lines"]):
                raise ValueError(f"{hypothesis['id']}/{variant['id']} has an invalid line")
    return hypotheses


def write_chunk(audio: Path, output: Path, start: float, end: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
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
            str(output),
        ],
        check=True,
    )


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getframerate() != 16000
            or source.getsampwidth() != 2
        ):
            raise RuntimeError(f"unexpected chunk format: {path}")
        frames = source.readframes(source.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def load_model(cache_dir: Path, device: str) -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=True,
        use_safetensors=False,
        torch_dtype=torch.float32,
    )
    return processor, model.eval().to(device)


def word_spans(
    processor: Wav2Vec2Processor,
    token_ids: torch.Tensor,
    aligned_tokens: torch.Tensor,
    scores: torch.Tensor,
    start: float,
    duration: float,
) -> list[dict]:
    blank = processor.tokenizer.pad_token_id
    spans = torchaudio.functional.merge_tokens(aligned_tokens, scores, blank=blank)
    if len(spans) != len(token_ids):
        raise RuntimeError("forced alignment did not return one span per target token")
    frame_seconds = duration / len(aligned_tokens)
    words = []
    characters = []
    starts = []
    ends = []
    token_scores = []
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids.tolist())
    for token, span in zip(tokens, spans):
        if token == processor.tokenizer.word_delimiter_token:
            if characters:
                words.append(
                    {
                        "word": "".join(characters),
                        "start": round(start + min(starts) * frame_seconds, 3),
                        "end": round(start + max(ends) * frame_seconds, 3),
                        "mean_log_probability": round(float(np.mean(token_scores)), 6),
                    }
                )
            characters, starts, ends, token_scores = [], [], [], []
            continue
        characters.append(token)
        starts.append(span.start)
        ends.append(span.end)
        token_scores.append(float(span.score))
    if characters:
        words.append(
            {
                "word": "".join(characters),
                "start": round(start + min(starts) * frame_seconds, 3),
                "end": round(start + max(ends) * frame_seconds, 3),
                "mean_log_probability": round(float(np.mean(token_scores)), 6),
            }
        )
    return words


def score_variant(
    processor: Wav2Vec2Processor,
    log_probs: torch.Tensor,
    variant: dict,
    start: float,
    duration: float,
) -> dict:
    normalized_lines = [normalize(line) for line in variant["lines"]]
    normalized_text = " ".join(normalized_lines)
    token_ids = torch.tensor(processor.tokenizer(normalized_text).input_ids, dtype=torch.long)
    input_lengths = torch.tensor([len(log_probs)])
    target_lengths = torch.tensor([len(token_ids)])
    loss = torch.nn.functional.ctc_loss(
        log_probs.unsqueeze(1),
        token_ids.unsqueeze(0),
        input_lengths,
        target_lengths,
        blank=processor.tokenizer.pad_token_id,
        reduction="sum",
        zero_infinity=True,
    )
    aligned, scores = torchaudio.functional.forced_align(
        log_probs.unsqueeze(0),
        token_ids.unsqueeze(0),
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=processor.tokenizer.pad_token_id,
    )
    words = word_spans(
        processor, token_ids, aligned[0], scores[0], start, duration
    )
    lines = []
    word_offset = 0
    for source_line, normalized_line in zip(variant["lines"], normalized_lines):
        word_count = len(normalized_line.split())
        line_words = words[word_offset : word_offset + word_count]
        word_offset += word_count
        lines.append(
            {
                "text": source_line,
                "start": line_words[0]["start"],
                "end": line_words[-1]["end"],
                "mean_log_probability": round(
                    float(np.mean([word["mean_log_probability"] for word in line_words])),
                    6,
                ),
            }
        )
    return {
        "id": variant["id"],
        "normalized_text": normalized_text,
        "token_count": len(token_ids),
        "ctc_loss": round(float(loss), 6),
        "ctc_loss_per_token": round(float(loss) / len(token_ids), 6),
        "lines": lines,
        "words": words,
    }


def run(args: argparse.Namespace, device: str) -> dict:
    hypotheses = validate_manifest(args.manifest)
    print(f"Loading {MODEL_ID}@{MODEL_REVISION} on {device} in FP32...", flush=True)
    processor, model = load_model(args.model_dir, device)
    results = []
    for audio in args.audio:
        for hypothesis in hypotheses:
            chunk = args.chunk_dir / audio.stem / f"{hypothesis['id']}.wav"
            write_chunk(audio, chunk, hypothesis["start"], hypothesis["end"])
            waveform = read_pcm16(chunk)
            inputs = processor.feature_extractor(
                waveform, sampling_rate=16000, return_tensors="pt"
            )
            with torch.inference_mode():
                logits = model(inputs.input_values.to(device)).logits[0].cpu()
            log_probs = logits.log_softmax(dim=-1)
            variants = [
                score_variant(
                    processor,
                    log_probs,
                    variant,
                    hypothesis["start"],
                    len(waveform) / 16000,
                )
                for variant in hypothesis["variants"]
            ]
            results.append(
                {
                    "audio": str(audio),
                    "hypothesis": hypothesis["id"],
                    "start": hypothesis["start"],
                    "end": hypothesis["end"],
                    "variants": variants,
                }
            )
            summary = ", ".join(
                f"{variant['id']}={variant['ctc_loss_per_token']:.3f}"
                for variant in variants
            )
            print(f"{audio.stem} {hypothesis['id']}: {summary}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": device,
        "dtype": "float32",
        "manifest": str(args.manifest),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and align Russian lyric hypotheses")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/human-transcript-hypotheses.json"),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--chunk-dir", type=Path, default=Path("work/ctc_hypothesis_chunks"))
    parser.add_argument("--output", type=Path, default=Path("work/ctc_hypotheses.json"))
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/huggingface"))
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        payload = run(args, device)
    except torch.OutOfMemoryError:
        if args.device != "auto" or device != "cuda":
            raise
        print("CUDA ran out of memory; retrying in system RAM...", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        payload = run(args, "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
