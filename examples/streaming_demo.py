#!/usr/bin/env python3
"""Experimental cache-level streaming demo for Dolphin streaming models.

This script drives the model through ``forward_encoder_chunk`` with encoder
caches and prints CTC greedy partial results as each encoder chunk is decoded.
For a file demo, fbank features are prepared up front; recognition itself is
performed chunk by chunk instead of by hard-cutting audio and calling
``transcribe`` repeatedly.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


DEFAULT_FRAME_SHIFT_SECONDS = 0.01


@dataclasses.dataclass
class StreamingEvent:
    index: int
    processed_seconds: float
    text: str
    delta: str
    is_endpoint: bool = False


@dataclasses.dataclass
class StreamingOutputState:
    wrote_delta: bool = False
    previous_display_text: str = ""

    def reset(self):
        self.wrote_delta = False
        self.previous_display_text = ""


@dataclasses.dataclass
class CtcEndpointRule:
    must_decoded_something: bool = True
    min_trailing_silence_ms: int = 1000
    min_utterance_length_ms: int = 0


@dataclasses.dataclass
class CtcEndpointConfig:
    blank_id: int = 0
    blank_scale: float = 1.0
    blank_threshold: float = 0.8
    rule1: CtcEndpointRule = dataclasses.field(
        default_factory=lambda: CtcEndpointRule(False, 5000, 0)
    )
    rule2: CtcEndpointRule = dataclasses.field(
        default_factory=lambda: CtcEndpointRule(True, 1000, 0)
    )
    rule3: CtcEndpointRule = dataclasses.field(
        default_factory=lambda: CtcEndpointRule(False, 0, 20000)
    )


class CtcEndpoint:
    def __init__(self, config: CtcEndpointConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.num_frames_decoded = 0
        self.num_frames_trailing_blank = 0

    def _rule_activated(
        self,
        rule: CtcEndpointRule,
        decoded_something: bool,
        trailing_silence_ms: int,
        utterance_length_ms: int,
    ) -> bool:
        return (
            (decoded_something or not rule.must_decoded_something)
            and trailing_silence_ms >= rule.min_trailing_silence_ms
            and utterance_length_ms >= rule.min_utterance_length_ms
        )

    def update(
        self,
        ctc_log_probs,
        decoded_something: bool,
        frame_shift_ms: int,
    ) -> bool:
        import math

        for logp_t in ctc_log_probs.squeeze(0):
            blank_prob = math.exp(float(logp_t[self.config.blank_id]))
            self.num_frames_decoded += 1
            if blank_prob > self.config.blank_threshold * self.config.blank_scale:
                self.num_frames_trailing_blank += 1
            else:
                self.num_frames_trailing_blank = 0

        utterance_length_ms = self.num_frames_decoded * frame_shift_ms
        trailing_silence_ms = self.num_frames_trailing_blank * frame_shift_ms
        return (
            self._rule_activated(
                self.config.rule1,
                decoded_something,
                trailing_silence_ms,
                utterance_length_ms,
            )
            or self._rule_activated(
                self.config.rule2,
                decoded_something,
                trailing_silence_ms,
                utterance_length_ms,
            )
            or self._rule_activated(
                self.config.rule3,
                decoded_something,
                trailing_silence_ms,
                utterance_length_ms,
            )
        )


def _format_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _iter_encoder_chunk_ranges(
    total_frames: int,
    chunk_size: int,
    subsampling: int,
    right_context: int,
    max_chunks: Optional[int] = None,
) -> Iterator[Tuple[int, int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if subsampling <= 0:
        raise ValueError("subsampling must be positive")

    context = right_context + 1
    stride = subsampling * chunk_size
    decoding_window = (chunk_size - 1) * subsampling + context

    index = 0
    for start in range(0, total_frames - context + 1, stride):
        if max_chunks is not None and index >= max_chunks:
            break
        end = min(start + decoding_window, total_frames)
        yield index, start, end
        index += 1


def _frame_shift_seconds(configs) -> float:
    dataset_conf = configs.get("dataset_conf", {})
    if "frontend_conf" in dataset_conf:
        frontend_conf = dataset_conf["frontend_conf"]
        sample_rate = frontend_conf.get("fs", 16000)
        hop_length = frontend_conf.get("hop_length", 128)
        return float(hop_length) / float(sample_rate)
    fbank_conf = dataset_conf.get("fbank_conf", {})
    return float(fbank_conf.get("frame_shift", 10)) / 1000.0


def _filter_nonspecial_ids(token_ids: List[int], tokenizer) -> List[int]:
    try:
        last_time_id = tokenizer.tokens2ids(["<30.00>"])[0]
    except Exception:
        last_time_id = -1
    return [token_id for token_id in token_ids if token_id > last_time_id]


def _ids_to_text(token_ids: List[int], tokenizer) -> str:
    if not token_ids:
        return ""
    return tokenizer.detokenize(_filter_nonspecial_ids(token_ids, tokenizer))[0]


def _endpoint_config_from_args(args) -> Optional[CtcEndpointConfig]:
    if getattr(args, "disable_endpoint", False):
        return None
    return CtcEndpointConfig(
        blank_threshold=args.endpoint_blank_threshold,
        rule1=CtcEndpointRule(
            False,
            args.endpoint_rule1_min_trailing_silence_ms,
            0,
        ),
        rule2=CtcEndpointRule(
            True,
            args.endpoint_rule2_min_trailing_silence_ms,
            0,
        ),
        rule3=CtcEndpointRule(
            False,
            0,
            args.endpoint_rule3_min_utterance_length_ms,
        ),
    )


def _emit_streaming_event(
    event: StreamingEvent,
    emit: str,
    output_state: Optional[StreamingOutputState] = None,
) -> bool:
    if event.is_endpoint:
        return False
    if emit == "line":
        print(
            f"[{event.index + 1:04d} {_format_time(event.processed_seconds)}] "
            f"{event.text}",
            flush=True,
        )
        if output_state is not None:
            output_state.previous_display_text = event.text
        return False

    print(event.delta, end="", flush=True)
    if output_state is not None:
        output_state.previous_display_text = event.text
        output_state.wrote_delta = True
    return True


def _emit_final_rescore(
    decoder: "StreamingCtcGreedyDecoder",
    args,
) -> str:
    final_text = decoder.final_attention_rescore(
        beam_size=args.rescore_beam_size,
        ctc_weight=args.rescore_ctc_weight,
        reverse_weight=args.reverse_weight,
        lang_sym=args.lang_sym,
        region_sym=args.region_sym,
    )
    if final_text:
        print(f"[final attention_rescoring] {final_text}", flush=True)
    return final_text


def _finish_current_segment(
    decoder: "StreamingCtcGreedyDecoder",
    args,
    output_state: StreamingOutputState,
    force_rescore: bool = False,
):
    if args.emit == "delta" and output_state.wrote_delta:
        print(flush=True)
    if (force_rescore or args.final_rescore == "attention") and decoder.encoder_out_chunks:
        _emit_final_rescore(decoder, args)
    decoder.reset_segment()
    output_state.reset()


class StreamingCtcGreedyDecoder:
    def __init__(
        self,
        model,
        tokenizer,
        chunk_size: int,
        left_chunks: int,
        endpoint_config: Optional[CtcEndpointConfig] = None,
    ):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.left_chunks = left_chunks
        self.subsampling = int(model.subsampling_rate())
        self.right_context = int(model.right_context())
        self.context = self.right_context + 1
        self.stride = self.subsampling * chunk_size
        self.decoding_window = (chunk_size - 1) * self.subsampling + self.context
        self.required_cache_size = chunk_size * left_chunks
        self.next_start = 0
        self.chunks_processed = 0
        self.endpoint = CtcEndpoint(endpoint_config) if endpoint_config else None
        self.reset_segment()

    def reset_segment(self):
        import torch

        self.att_cache = torch.zeros((0, 0, 0, 0), device=self.model.device)
        self.cnn_cache = torch.zeros((0, 0, 0, 0), device=self.model.device)
        self.emitted_ids: List[int] = []
        self.previous_frame_id = 0
        self.previous_text = ""
        self.encoder_offset = 0
        self.encoder_out_chunks = []
        if self.endpoint is not None:
            self.endpoint.reset()

    def decode_available(
        self,
        feats,
        frame_shift_seconds: float,
        max_chunks: Optional[int] = None,
        flush: bool = False,
    ) -> Iterator[StreamingEvent]:
        import torch

        total_frames = feats.size(1)
        endpoint_frame_shift_ms = max(
            1,
            int(round(frame_shift_seconds * self.subsampling * 1000)),
        )
        while self.next_start + self.context <= total_frames:
            if max_chunks is not None and self.chunks_processed >= max_chunks:
                break
            if not flush and self.next_start + self.decoding_window > total_frames:
                break

            end = min(self.next_start + self.decoding_window, total_frames)
            chunk_feats = feats[:, self.next_start:end, :]
            encoder_out, self.att_cache, self.cnn_cache = self.model.forward_encoder_chunk(
                chunk_feats,
                self.encoder_offset,
                self.required_cache_size,
                self.att_cache,
                self.cnn_cache,
            )
            self.encoder_offset += encoder_out.size(1)
            self.encoder_out_chunks.append(encoder_out)

            ctc_log_probs = self.model.ctc_activation(encoder_out)
            frame_ids = torch.argmax(ctc_log_probs, dim=-1).squeeze(0).tolist()
            for token_id in frame_ids:
                if token_id != 0 and token_id != self.previous_frame_id:
                    self.emitted_ids.append(token_id)
                self.previous_frame_id = token_id

            text = _ids_to_text(self.emitted_ids, self.tokenizer)
            event = None
            if text != self.previous_text:
                if text.startswith(self.previous_text):
                    delta = text[len(self.previous_text):]
                else:
                    delta = text
                self.previous_text = text
                event = StreamingEvent(
                    index=self.chunks_processed,
                    processed_seconds=end * frame_shift_seconds,
                    text=text,
                    delta=delta,
                )

            self.chunks_processed += 1
            self.next_start += self.stride
            if event is not None:
                yield event

            if self.endpoint is not None and self.endpoint.update(
                ctc_log_probs,
                decoded_something=bool(text),
                frame_shift_ms=endpoint_frame_shift_ms,
            ):
                yield StreamingEvent(
                    index=self.chunks_processed - 1,
                    processed_seconds=end * frame_shift_seconds,
                    text=text,
                    delta="",
                    is_endpoint=True,
                )

    def final_attention_rescore(
        self,
        beam_size: int = 10,
        ctc_weight: float = 0.0,
        reverse_weight: float = 0.0,
        lang_sym: Optional[str] = None,
        region_sym: Optional[str] = None,
    ) -> str:
        import torch
        from dolphin.search import attention_rescoring, ctc_prefix_beam_search

        if not self.encoder_out_chunks:
            return ""

        encoder_out = torch.cat(self.encoder_out_chunks, dim=1)
        encoder_lens = torch.tensor(
            [encoder_out.size(1)],
            dtype=torch.long,
            device=encoder_out.device,
        )
        encoder_mask = torch.ones(
            1,
            1,
            encoder_out.size(1),
            dtype=torch.bool,
            device=encoder_out.device,
        )
        ctc_probs = self.model.ctc_activation(encoder_out)
        ctc_prefix_results = ctc_prefix_beam_search(
            ctc_probs,
            encoder_lens,
            beam_size,
        )

        rescore_encoder_out = encoder_out
        rescore_encoder_mask = encoder_mask
        if getattr(self.model, "apply_non_blank_embedding", False):
            rescore_encoder_out, rescore_encoder_mask = self.model.filter_blank_embedding(
                ctc_probs,
                encoder_out,
            )

        infos = {
            "tokenizer": self.tokenizer,
            "need_timestamp": False,
            "word_timestamp": False,
        }
        if lang_sym is not None:
            infos["langs"] = [f"<{lang_sym}>"]
        if region_sym is not None:
            infos["regions"] = [f"<{region_sym}>"]

        results = attention_rescoring(
            self.model,
            ctc_prefix_results,
            rescore_encoder_out,
            encoder_lens,
            ctc_weight=ctc_weight,
            reverse_weight=reverse_weight,
            infos=infos,
            encoder_mask=rescore_encoder_mask,
        )
        return _ids_to_text(results[0].tokens, self.tokenizer)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Dolphin cache-level streaming ASR demo."
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
    parser.add_argument("--lang_sym", default=None, help="language symbol for final attention rescoring, e.g. zh")
    parser.add_argument("--region_sym", default=None, help="region symbol for final attention rescoring, e.g. CN")
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
        "--emit",
        choices=("delta", "line"),
        default="delta",
        help="delta prints only newly recognized text; line prints timestamped partials (default: delta)",
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


def main() -> int:
    args = _parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import dolphin
    from dolphin.processor import extract_feats
    from dolphin.tokenizer import init_tokenizer

    model_dir = args.model_dir or Path.home() / ".cache" / "dolphin" / args.model

    print(f"loading model: {args.model} ({model_dir})", file=sys.stderr, flush=True)
    model = dolphin.load_model(args.model, model_dir, args.device)
    batch = extract_feats([str(args.audio)], model.model_configs)
    feats = batch["feats"].to(model.device)
    tokenizer = init_tokenizer(model.model_configs)
    frame_shift_seconds = _frame_shift_seconds(model.model_configs)

    decoder = StreamingCtcGreedyDecoder(
        model,
        tokenizer,
        chunk_size=args.chunk_size,
        left_chunks=args.left_chunks,
        endpoint_config=_endpoint_config_from_args(args),
    )
    chunk_ms = decoder.stride * frame_shift_seconds * 1000
    print(
        f"streaming file: {args.audio} "
        f"frames={feats.size(1)} "
        f"chunk_size={args.chunk_size} (~{chunk_ms:.0f}ms) "
        f"left_chunks={args.left_chunks}",
        file=sys.stderr,
        flush=True,
    )

    started_at = time.time()
    output_state = StreamingOutputState()
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

        if args.realtime:
            sleep_for = event.processed_seconds - (time.time() - started_at)
            if sleep_for > 0:
                time.sleep(sleep_for)

    if args.emit == "delta" and output_state.wrote_delta:
        print(flush=True)

    if args.final_rescore == "attention" and decoder.encoder_out_chunks:
        _emit_final_rescore(decoder, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
