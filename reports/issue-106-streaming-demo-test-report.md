# Issue #106 Streaming Demo Test Report

Date: 2026-06-11
Branch: `codex/integration-issues-80-92-93`
Issue: https://github.com/DataoceanAI/Dolphin/issues/106

## Scope

- Added `examples/streaming_demo.py` as an experimental terminal cache-level streaming demo for `small.cn.streaming` and other streaming models.
- Added `examples/microphone_streaming_demo.py` for live microphone streaming output.
- Added optional final attention rescoring after streaming partials.
- Added WeNet-style CTC endpointing so silence or long utterances finalize and rescore segments automatically.
- Added CLI/API passthrough for `decoding_chunk_size`, `num_decoding_left_chunks`, and `simulate_streaming`.
- Fixed the SDPA relative-position attention path so streaming chunk inference can run when no explicit mask is supplied.
- Updated `README.md` with terminal demo commands.
- Added `reports/issue-triage.md` so handled and candidate issues are tracked in the repository.

## How To Run

```shell
python examples/streaming_demo.py audio.wav --model small.cn.streaming --device cuda --chunk_size 16 --final_rescore attention
```

CPU smoke test:

```shell
python examples/streaming_demo.py audio.wav --model small.cn.streaming --device cpu --chunk_size 16 --emit line --max_chunks 2
```

Microphone demo:

```shell
python -m pip install sounddevice
python examples/microphone_streaming_demo.py --model small.cn.streaming --device cuda --chunk_size 16 --final_rescore attention
```

## Verification

```shell
env TRANSFORMERS_NO_TF=1 USE_TF=0 python -m pytest
```

Result: `23 passed in 1.19s`.

```shell
python -m py_compile examples/streaming_demo.py examples/microphone_streaming_demo.py
```

Result: passed.

```shell
python examples/streaming_demo.py --help
python examples/microphone_streaming_demo.py --help
```

Result: passed.

Real model smoke test:

```shell
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python examples/streaming_demo.py /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model small.cn.streaming --model_dir /private/tmp/dolphin-models/small.cn.streaming --device cpu --chunk_size 16 --emit line --max_chunks 2 --final_rescore attention
```

Result:

```text
[0002 00:01.310] 诚然
[final attention_rescoring] 诚然
```

Full short-audio cache-level streaming smoke test:

```shell
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python examples/streaming_demo.py /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model small.cn.streaming --model_dir /private/tmp/dolphin-models/small.cn.streaming --device cpu --chunk_size 16 --emit line --final_rescore attention
```

Result:

```text
[0002 00:01.310] 诚然
[0003 00:01.950] 诚然时代正
[0004 00:02.590] 诚然时代正在推陈
[0005 00:03.230] 诚然时代正在推陈初新
[0006 00:03.870] 诚然时代正在推陈初新但文化
[0007 00:04.510] 诚然时代正在推陈初新但文化之精髓
[0008 00:05.150] 诚然时代正在推陈初新但文化之精髓切需
[0009 00:05.790] 诚然时代正在推陈初新但文化之精髓切需传承
[final attention_rescoring] 诚然时代正在推陈出新但文化之精髓切需传承
```

Forced endpoint smoke test:

```shell
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python examples/streaming_demo.py /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model small.cn.streaming --model_dir /private/tmp/dolphin-models/small.cn.streaming --device cpu --chunk_size 16 --emit line --final_rescore attention --endpoint_rule3_min_utterance_length_ms 3000
```

Result:

```text
[0002 00:01.310] 诚然
[0003 00:01.950] 诚然时代正
[0004 00:02.590] 诚然时代正在推陈
[0005 00:03.230] 诚然时代正在推陈初新
[final attention_rescoring] 诚然时代正在推陈出新
[0006 00:03.870] 文化
[0007 00:04.510] 文化之精髓
[0008 00:05.150] 文化之精髓切需
[0009 00:05.790] 文化之精髓切需传承
[final attention_rescoring] 文化之精髓切需传承
```

## Notes

- `--chunk_size 16` maps to the encoder streaming `decoding_chunk_size`.
- The file demo prepares fbank features from the file first, then runs model inference chunk by chunk with encoder caches.
- The microphone demo uses live audio input and emits CTC greedy partial text as new chunks decode.
- CTC greedy partial text may temporarily contain unstable words before later chunks arrive.
- `--final_rescore attention` runs attention rescoring once at the end and prints `[final attention_rescoring] ...`.
- CTC endpointing is enabled by default: 5s initial silence, 1s trailing silence after decoded text, or 20s max utterance length.
- This is intentionally an experimental terminal demo, not a production streaming server.
