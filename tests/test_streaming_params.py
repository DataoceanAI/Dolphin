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
model_module = importlib.import_module("dolphin.model")


class FakeTokenizer:
    symbol_table = {}

    def tokens2ids(self, tokens):
        mapping = {"<30.00>": 2}
        return [mapping[item] for item in tokens]

    def ids2tokens(self, ids):
        mapping = {1: "<zh>", 2: "<CN>", 3: "你", 4: "好"}
        return [mapping[item] for item in ids]

    def detokenize(self, tokens):
        token_text = "".join(self.ids2tokens(tokens))
        if tokens == [3, 4]:
            token_text = "你好"
        return [token_text]


class FakeHyp:
    tokens = [1, 2, 3, 4]
    times = None


class FakeModel:
    device = torch.device("cpu")
    model_configs = {"support_timestamp": False}

    def __init__(self):
        self.decode_kwargs = None

    def decode(self, **kwargs):
        self.decode_kwargs = kwargs
        return {"attention_rescoring": [FakeHyp()]}


def test_transcribe_passes_streaming_decode_parameters(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(
        transcribe,
        "extract_feats",
        lambda audio, configs: {
            "feats": torch.zeros(1, 4, 8),
            "feats_lengths": torch.tensor([4]),
        },
    )
    monkeypatch.setattr(transcribe, "init_tokenizer", lambda configs: FakeTokenizer())

    result = transcribe.transcribe(
        model,
        torch.zeros(1, 16000),
        decoding_chunk_size=16,
        num_decoding_left_chunks=4,
        simulate_streaming=True,
    )

    assert result.text_nospecial == "你好"
    assert model.decode_kwargs["decoding_chunk_size"] == 16
    assert model.decode_kwargs["num_decoding_left_chunks"] == 4
    assert model.decode_kwargs["simulate_streaming"] is True


def test_parser_streaming_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dolphin",
            "audio.wav",
            "--decoding_chunk_size",
            "16",
            "--num_decoding_left_chunks",
            "4",
            "--simulate_streaming",
            "true",
        ],
    )

    args = transcribe.parser_args()

    assert args.decoding_chunk_size == 16
    assert args.num_decoding_left_chunks == 4
    assert args.simulate_streaming is True


def test_rel_position_sdpa_accepts_empty_streaming_mask():
    attention = model_module.RelPositionMultiHeadedAttention(
        2,
        8,
        0.0,
        use_sdpa=True,
    )
    query = torch.randn(1, 4, 8)
    pos_emb = torch.randn(1, 4, 8)

    output, _ = attention(query, query, query, pos_emb=pos_emb)

    assert output.shape == query.shape
