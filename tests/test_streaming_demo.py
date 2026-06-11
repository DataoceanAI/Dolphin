import importlib.util
from pathlib import Path


def _load_streaming_demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "streaming_demo.py"
    spec = importlib.util.spec_from_file_location("streaming_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_time():
    demo = _load_streaming_demo()

    assert demo._format_time(0) == "00:00.000"
    assert demo._format_time(65.4321) == "01:05.432"


def test_iter_chunk_ranges():
    demo = _load_streaming_demo()

    assert list(demo._iter_chunk_ranges(10, 4)) == [
        (0, 0, 4),
        (1, 4, 8),
        (2, 8, 10),
    ]


def test_iter_chunk_ranges_respects_max_chunks():
    demo = _load_streaming_demo()

    assert list(demo._iter_chunk_ranges(10, 4, max_chunks=2)) == [
        (0, 0, 4),
        (1, 4, 8),
    ]
