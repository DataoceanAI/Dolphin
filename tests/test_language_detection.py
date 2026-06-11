import importlib
import sys
import types

import torch


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
dolphin = importlib.import_module("dolphin")


class FakeTokenizer:
    def ids2tokens(self, ids):
        assert ids == [1, 2]
        return ["<zh>", "<CN>"]


class FakeModel:
    device = torch.device("cpu")
    model_configs = {"dummy": True}

    def __init__(self):
        self.called = False

    def detect_language(self, feats, feats_lengths):
        self.called = True
        assert feats.device == self.device
        assert feats_lengths.device == self.device
        return torch.tensor([[1, 2]])


def test_detect_language_returns_language_and_region(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(
        transcribe,
        "extract_feats",
        lambda audio, configs: {
            "feats": torch.zeros(1, 2, 3),
            "feats_lengths": torch.tensor([2]),
        },
    )
    monkeypatch.setattr(transcribe, "init_tokenizer", lambda configs: FakeTokenizer())

    assert transcribe.detect_language(model, "audio.wav", max_duration=None) == ("zh", "CN")
    assert model.called


def test_detect_language_is_exported_at_package_top_level():
    assert dolphin.detect_language is transcribe.detect_language


def test_cli_detect_language_task_prints_only_language_result(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dolphin",
            "audio.wav",
            "--model",
            "base",
            "--model_dir",
            "/tmp/model",
            "--device",
            "cpu",
            "--task",
            "detect_language",
        ],
    )
    monkeypatch.setattr(transcribe, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(transcribe, "detect_language", lambda model, audio, max_duration=None: ("hi", "IN"))

    transcribe.cli()

    assert capsys.readouterr().out == "hi\tIN\n"


def test_limit_audio_duration_crops_tensor_audio():
    audio = torch.arange(20, dtype=torch.float32).unsqueeze(0)

    limited = transcribe._limit_audio_duration(audio, max_duration=0.0005)

    assert torch.equal(limited, audio[:, :8])
