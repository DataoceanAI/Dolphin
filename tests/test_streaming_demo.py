import importlib.util
import sys
from pathlib import Path

import torch


def _load_streaming_demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "streaming_demo.py"
    spec = importlib.util.spec_from_file_location("streaming_demo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_format_time():
    demo = _load_streaming_demo()

    assert demo._format_time(0) == "00:00.000"
    assert demo._format_time(65.4321) == "01:05.432"


def test_iter_encoder_chunk_ranges():
    demo = _load_streaming_demo()

    assert list(demo._iter_encoder_chunk_ranges(140, 16, 4, 6)) == [
        (0, 0, 67),
        (1, 64, 131),
        (2, 128, 140),
    ]


def test_iter_encoder_chunk_ranges_respects_max_chunks():
    demo = _load_streaming_demo()

    assert list(demo._iter_encoder_chunk_ranges(140, 16, 4, 6, max_chunks=2)) == [
        (0, 0, 67),
        (1, 64, 131),
    ]


def test_parser_chunk_size_alias(monkeypatch):
    demo = _load_streaming_demo()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "streaming_demo.py",
            "audio.wav",
            "--chunk_size",
            "16",
            "--left_chunks",
            "4",
            "--final_rescore",
            "attention",
            "--rescore_beam_size",
            "3",
        ],
    )

    args = demo._parse_args()

    assert args.chunk_size == 16
    assert args.left_chunks == 4
    assert args.final_rescore == "attention"
    assert args.rescore_beam_size == 3


def test_endpoint_rule2_after_decoded_text_and_silence():
    demo = _load_streaming_demo()
    endpoint = demo.CtcEndpoint(demo.CtcEndpointConfig())
    blank_frames = torch.log(torch.tensor([[[0.9, 0.1]] * 25]))

    assert endpoint.update(blank_frames, decoded_something=True, frame_shift_ms=40)


def test_endpoint_rule3_long_utterance():
    demo = _load_streaming_demo()
    endpoint = demo.CtcEndpoint(demo.CtcEndpointConfig())
    speech_frames = torch.log(torch.tensor([[[0.1, 0.9]] * 500]))

    assert endpoint.update(speech_frames, decoded_something=True, frame_shift_ms=40)
