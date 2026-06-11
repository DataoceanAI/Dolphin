#!/usr/bin/env python3
"""Experimental microphone streaming demo for Dolphin streaming models."""

import argparse
import queue
import sys
import time
from pathlib import Path
from typing import Optional


SAMPLE_RATE = 16000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Dolphin microphone streaming ASR demo."
    )
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
    parser.add_argument("--lang_sym", default=None, help="language symbol for final attention rescoring, e.g. zh")
    parser.add_argument("--region_sym", default=None, help="region symbol for final attention rescoring, e.g. CN")
    parser.add_argument(
        "--input_device",
        default=None,
        help="sounddevice input device id/name (default: system default)",
    )
    parser.add_argument(
        "--list_devices",
        action="store_true",
        help="list microphone devices and exit",
    )
    parser.add_argument(
        "--chunk_size",
        "--decoding_chunk_size",
        dest="chunk_size",
        type=int,
        default=16,
        help="encoder streaming chunk size in subsampled frames (default: 16)",
    )
    parser.add_argument(
        "--left_chunks",
        "--num_decoding_left_chunks",
        dest="left_chunks",
        type=int,
        default=4,
        help="left chunks kept by encoder cache; use -1 for all history (default: 4)",
    )
    parser.add_argument(
        "--block_ms",
        type=int,
        default=40,
        help="microphone callback block size in milliseconds (default: 40)",
    )
    parser.add_argument(
        "--feature_update_ms",
        type=int,
        default=80,
        help="minimum interval between feature refreshes in milliseconds (default: 80)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="optional maximum recording duration in seconds",
    )
    parser.add_argument(
        "--emit",
        choices=("delta", "line"),
        default="delta",
        help="delta prints only newly recognized text; line prints timestamped partials (default: delta)",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="limit encoder chunks for quick smoke tests",
    )
    parser.add_argument(
        "--final_rescore",
        choices=("none", "attention"),
        default="none",
        help="run final attention rescoring after streaming partials (default: none)",
    )
    parser.add_argument(
        "--rescore_beam_size",
        type=int,
        default=10,
        help="beam size for final attention rescoring (default: 10)",
    )
    parser.add_argument(
        "--rescore_ctc_weight",
        type=float,
        default=0.0,
        help="CTC score weight for final attention rescoring (default: 0.0)",
    )
    parser.add_argument(
        "--reverse_weight",
        type=float,
        default=0.0,
        help="right-to-left decoder weight for final attention rescoring (default: 0.0)",
    )
    parser.add_argument(
        "--disable_endpoint",
        action="store_true",
        help="disable CTC endpoint segmentation",
    )
    parser.add_argument(
        "--endpoint_blank_threshold",
        type=float,
        default=0.8,
        help="blank probability threshold treated as silence (default: 0.8)",
    )
    parser.add_argument(
        "--endpoint_rule1_min_trailing_silence_ms",
        type=int,
        default=5000,
        help="endpoint rule1: silence timeout without decoded text (default: 5000)",
    )
    parser.add_argument(
        "--endpoint_rule2_min_trailing_silence_ms",
        type=int,
        default=1000,
        help="endpoint rule2: silence timeout after decoded text (default: 1000)",
    )
    parser.add_argument(
        "--endpoint_rule3_min_utterance_length_ms",
        type=int,
        default=20000,
        help="endpoint rule3: maximum utterance length (default: 20000)",
    )
    return parser.parse_args()


def _load_sounddevice():
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The microphone demo requires sounddevice. Install it with: "
            "python -m pip install sounddevice"
        ) from exc
    return sd


def _extract_feats_from_samples(samples, configs):
    import torch
    from dolphin.processor import extract_feats

    waveform = torch.from_numpy(samples.copy()).float().unsqueeze(0)
    return extract_feats([waveform], configs)["feats"]


def _drain_audio(audio_queue: "queue.Queue", timeout: Optional[float]):
    blocks = []
    try:
        blocks.append(audio_queue.get(timeout=timeout))
    except queue.Empty:
        return blocks

    while True:
        try:
            blocks.append(audio_queue.get_nowait())
        except queue.Empty:
            break
    return blocks


def main() -> int:
    args = _parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.block_ms <= 0:
        raise ValueError("--block_ms must be positive")
    if args.feature_update_ms <= 0:
        raise ValueError("--feature_update_ms must be positive")

    sd = _load_sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import numpy as np
    import dolphin
    from dolphin.tokenizer import init_tokenizer
    from streaming_demo import (
        StreamingCtcGreedyDecoder,
        StreamingOutputState,
        _emit_final_rescore,
        _emit_streaming_event,
        _endpoint_config_from_args,
        _finish_current_segment,
        _frame_shift_seconds,
    )

    model_dir = args.model_dir or Path.home() / ".cache" / "dolphin" / args.model

    print(f"loading model: {args.model} ({model_dir})", file=sys.stderr, flush=True)
    model = dolphin.load_model(args.model, model_dir, args.device)
    tokenizer = init_tokenizer(model.model_configs)
    decoder = StreamingCtcGreedyDecoder(
        model,
        tokenizer,
        chunk_size=args.chunk_size,
        left_chunks=args.left_chunks,
        endpoint_config=_endpoint_config_from_args(args),
    )
    frame_shift_seconds = _frame_shift_seconds(model.model_configs)
    chunk_ms = decoder.stride * frame_shift_seconds * 1000
    print(
        f"listening: sample_rate={SAMPLE_RATE} chunk_size={args.chunk_size} "
        f"(~{chunk_ms:.0f}ms) left_chunks={args.left_chunks}",
        file=sys.stderr,
        flush=True,
    )

    audio_queue: "queue.Queue" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr, flush=True)
        audio_queue.put(indata[:, 0].copy())

    blocksize = int(SAMPLE_RATE * args.block_ms / 1000)
    samples = np.empty((0,), dtype=np.float32)
    last_feature_update = 0.0
    started_at = time.time()
    output_state = StreamingOutputState()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
            device=args.input_device,
        ):
            while True:
                if args.duration is not None and time.time() - started_at >= args.duration:
                    break
                if args.max_chunks is not None and decoder.chunks_processed >= args.max_chunks:
                    break

                blocks = _drain_audio(audio_queue, timeout=0.1)
                if blocks:
                    samples = np.concatenate([samples, *blocks])

                now = time.time()
                if now - last_feature_update < args.feature_update_ms / 1000.0:
                    continue
                last_feature_update = now

                if samples.size == 0:
                    continue

                feats = _extract_feats_from_samples(samples, model.model_configs).to(model.device)
                for event in decoder.decode_available(
                    feats,
                    frame_shift_seconds,
                    max_chunks=args.max_chunks,
                    flush=False,
                ):
                    if event.is_endpoint:
                        _finish_current_segment(decoder, args, output_state)
                        continue
                    _emit_streaming_event(event, args.emit, output_state)
    except KeyboardInterrupt:
        pass

    if samples.size:
        feats = _extract_feats_from_samples(samples, model.model_configs).to(model.device)
        for event in decoder.decode_available(
            feats,
            frame_shift_seconds,
            max_chunks=args.max_chunks,
            flush=True,
        ):
            if event.is_endpoint:
                _finish_current_segment(decoder, args, output_state)
                continue
            _emit_streaming_event(event, args.emit, output_state)

    if args.emit == "delta" and output_state.wrote_delta:
        print(flush=True)
    if args.final_rescore == "attention" and decoder.encoder_out_chunks:
        _emit_final_rescore(decoder, args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
