import importlib
import json
import sys
import types


def _install_modelscope_stub():
    modelscope = types.ModuleType("modelscope")
    modelscope.snapshot_download = lambda *args, **kwargs: None

    model_module = types.ModuleType("modelscope.models.audio.funasr.model")

    class GenericFunASR:
        pass

    model_module.GenericFunASR = GenericFunASR

    sys.modules.setdefault("modelscope", modelscope)
    sys.modules.setdefault("modelscope.models", types.ModuleType("modelscope.models"))
    sys.modules.setdefault("modelscope.models.audio", types.ModuleType("modelscope.models.audio"))
    sys.modules.setdefault("modelscope.models.audio.funasr", types.ModuleType("modelscope.models.audio.funasr"))
    sys.modules.setdefault("modelscope.models.audio.funasr.model", model_module)


_install_modelscope_stub()
transcribe = importlib.import_module("dolphin.transcribe")


def test_format_cli_output_short_text():
    result = transcribe.TranscribeResult(
        text="<zh><CN><asr><notimestamp>你好",
        text_nospecial="你好",
        language="zh",
        region="CN",
    )

    assert transcribe._format_cli_output(result) == "你好"


def test_format_cli_output_segments_text():
    segments = [
        transcribe.TranscribeSegmentResult(
            text="a",
            text_nospecial="第一段",
            language="zh",
            region="CN",
            start=0.0,
            end=1.0,
        ),
        transcribe.TranscribeSegmentResult(
            text="b",
            text_nospecial="第二段",
            language="zh",
            region="CN",
            start=1.0,
            end=2.0,
        ),
    ]

    assert transcribe._format_cli_output(segments) == "第一段\n第二段"


def test_format_cli_output_json_preserves_metadata():
    result = transcribe.TranscribeResult(
        text="<zh><CN><asr><notimestamp>你好",
        text_nospecial="你好",
        language="zh",
        region="CN",
        word_timestamps=[{"word": "你好", "start": 0.0, "end": 0.4}],
    )

    payload = json.loads(transcribe._format_cli_output(result, "json"))

    assert payload["text_nospecial"] == "你好"
    assert payload["language"] == "zh"
    assert payload["word_timestamps"] == [{"word": "你好", "start": 0.0, "end": 0.4}]


def test_format_cli_output_short_result_srt_uses_word_timestamps():
    result = transcribe.TranscribeResult(
        text="raw",
        text_nospecial="你好",
        language="zh",
        region="CN",
        word_timestamps=[
            {"word": "你", "start": 0.12, "end": 0.31},
            {"word": "好", "start": 0.31, "end": 0.62},
        ],
    )

    assert transcribe._format_cli_output(result, "srt") == (
        "1\n"
        "00:00:00,120 --> 00:00:00,620\n"
        "你好"
    )


def test_format_cli_output_segments_srt():
    segments = [
        transcribe.TranscribeSegmentResult(
            text="a",
            text_nospecial="第一段",
            language="zh",
            region="CN",
            start=0.0,
            end=1.25,
        ),
        transcribe.TranscribeSegmentResult(
            text="b",
            text_nospecial="第二段",
            language="zh",
            region="CN",
            start=61.0,
            end=62.5,
        ),
    ]

    assert transcribe._format_cli_output(segments, "srt") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "第一段\n\n"
        "2\n"
        "00:01:01,000 --> 00:01:02,500\n"
        "第二段"
    )


def test_emit_cli_output_stdout(capsys):
    result = transcribe.TranscribeResult(
        text="raw",
        text_nospecial="plain",
        language="zh",
        region="CN",
    )

    transcribe._emit_cli_output(result, "txt", None)

    assert capsys.readouterr().out == "plain\n"


def test_emit_cli_output_file(tmp_path):
    result = transcribe.TranscribeResult(
        text="raw",
        text_nospecial="plain",
        language="zh",
        region="CN",
    )
    output = tmp_path / "nested" / "result.txt"

    transcribe._emit_cli_output(result, "txt", output)

    assert output.read_text(encoding="utf-8") == "plain\n"
