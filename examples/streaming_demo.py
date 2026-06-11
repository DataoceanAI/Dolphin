#!/usr/bin/env python3
"""Experimental file-streaming demo for Dolphin streaming models.

This script reads an audio file, feeds it to Dolphin in chunks, and prints
recognition results as each chunk is processed. It is intended as a simple
terminal demo for streaming models such as ``small.cn.streaming``.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple


SAMPLE_RATE = 16000


def _format_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _iter_chunk_ranges(
    total_samples: int,
    chunk_samples: int,
    max_chunks: Optional[int] = None,
) -> Iterator[Tuple[int, int, int]]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")

    index = 0
    start = 0
    while start < total_samples:
        if max_chunks is not None and index >= max_chunks:
            break
        end = min(start + chunk_samples, total_samples)
        yield index, start, end
        index += 1
        start = end


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Dolphin file-streaming ASR demo."
    )
    parser.add_argument("audio", type=Path, help="audio file to stream")
    parser.add_argument(
        "--model",
        default="small.cn.streaming",
        help="streaming model name (default: small.cn.streaming)",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="model checkpoint directory; defaults to ~/.cache/dolphin/<model>",
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    parser.add_argument("--lang_sym", default="zh", help="language symbol (default: zh)")
    parser.add_argument("--region_sym", default=None, help="region symbol, e.g. CN")
    parser.add_argument("--chunk_duration", type=float, default=4.0, help="chunk size in seconds (default: 4.0)")
    parser.add_argument("--beam_size", type=int, default=5, help="beam size (default: 5)")
    parser.add_argument(
        "--decoding_method",
        default="attention_rescoring",
        choices=("attention", "attention_rescoring"),
        help="decoding method (default: attention_rescoring)",
    )
    parser.add_argument("--decoding_chunk_size", type=int, default=16, help="encoder decoding chunk size (default: 16)")
    parser.add_argument("--num_decoding_left_chunks", type=int, default=4, help="left chunks kept by encoder (default: 4)")
    parser.add_argument(
        "--mode",
        choices=("chunk", "rolling"),
        default="chunk",
        help="chunk prints each chunk independently; rolling prints accumulated partial text (default: chunk)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="sleep between chunks to approximate real-time playback",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="limit chunks for quick smoke tests",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.chunk_duration <= 0:
        raise ValueError("--chunk_duration must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch

    import dolphin
    from dolphin.audio import load_audio
    from dolphin.transcribe import transcribe

    model_dir = args.model_dir or Path.home() / ".cache" / "dolphin" / args.model

    print(f"loading model: {args.model} ({model_dir})", flush=True)
    model = dolphin.load_model(args.model, model_dir, args.device)
    waveform = torch.from_numpy(load_audio(str(args.audio))).float().unsqueeze(0)

    chunk_samples = int(args.chunk_duration * SAMPLE_RATE)
    total_samples = waveform.size(-1)
    print(
        f"streaming file: {args.audio} "
        f"duration={total_samples / SAMPLE_RATE:.2f}s "
        f"chunk={args.chunk_duration:.2f}s mode={args.mode}",
        flush=True,
    )

    started_at = time.time()
    for index, start, end in _iter_chunk_ranges(total_samples, chunk_samples, args.max_chunks):
        chunk = waveform[:, :end] if args.mode == "rolling" else waveform[:, start:end]
        result = transcribe(
            model,
            chunk,
            lang_sym=args.lang_sym,
            region_sym=args.region_sym,
            predict_time=False,
            word_timestamp=False,
            decoding_method=args.decoding_method,
            beam_size=args.beam_size,
            decoding_chunk_size=args.decoding_chunk_size,
            num_decoding_left_chunks=args.num_decoding_left_chunks,
            simulate_streaming=True,
        )

        if args.mode == "rolling":
            span = f"00:00.000-{_format_time(end / SAMPLE_RATE)}"
            label = "partial"
        else:
            span = f"{_format_time(start / SAMPLE_RATE)}-{_format_time(end / SAMPLE_RATE)}"
            label = "chunk"
        print(f"[{index + 1:04d} {label} {span}] {result.text_nospecial}", flush=True)

        if args.realtime:
            target_elapsed = end / SAMPLE_RATE
            sleep_for = target_elapsed - (time.time() - started_at)
            if sleep_for > 0:
                time.sleep(sleep_for)

    return 0


if __name__ == "__main__":
    sys.exit(main())
