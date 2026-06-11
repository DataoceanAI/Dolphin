import importlib
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


def test_remove_punctuation_preserves_special_tokens():
    text = "<zh><CN><asr><notimestamp> 你好，world!"

    assert transcribe._remove_punctuation_preserving_special_tokens(text) == (
        "<zh><CN><asr><notimestamp> 你好world"
    )


def test_remove_result_punctuation_cleans_text_and_timestamps():
    result = transcribe.TranscribeResult(
        text="<zh><CN><asr><notimestamp> 你好，world!",
        text_nospecial="你好，world!",
        language="zh",
        region="CN",
        word_timestamps=[
            {"word": "你好", "start": 0.0, "end": 0.4},
            {"word": "，", "start": 0.4, "end": 0.5},
            {"word": "world!", "start": 0.5, "end": 0.9},
        ],
    )

    cleaned = transcribe._remove_result_punctuation(result)

    assert cleaned.text == "<zh><CN><asr><notimestamp> 你好world"
    assert cleaned.text_nospecial == "你好world"
    assert cleaned.word_timestamps == [
        {"word": "你好", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
    ]


def test_parser_remove_punctuation_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dolphin", "audio.wav", "--remove_punctuation", "true"])

    assert transcribe.parser_args().remove_punctuation is True
