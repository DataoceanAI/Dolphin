# Issue #106 Streaming Demo Test Report

Date: 2026-06-11
Branch: `codex/integration-issues-80-92-93`
Issue: https://github.com/DataoceanAI/Dolphin/issues/106

## Scope

- Added `examples/streaming_demo.py` as an experimental terminal file-streaming demo for `small.cn.streaming` and other streaming models.
- Added CLI/API passthrough for `decoding_chunk_size`, `num_decoding_left_chunks`, and `simulate_streaming`.
- Fixed the SDPA relative-position attention path so streaming chunk inference can run when no explicit mask is supplied.
- Updated `README.md` with terminal demo commands.
- Added `reports/issue-triage.md` so handled and candidate issues are tracked in the repository.

## How To Run

```shell
python examples/streaming_demo.py audio.wav --model small.cn.streaming --device cuda
```

CPU smoke test:

```shell
python examples/streaming_demo.py audio.wav --model small.cn.streaming --device cpu --chunk_duration 3 --max_chunks 1
```

## Verification

```shell
env TRANSFORMERS_NO_TF=1 USE_TF=0 python -m pytest
```

Result: `20 passed in 1.17s`.

```shell
python -m py_compile examples/streaming_demo.py
```

Result: passed.

```shell
python examples/streaming_demo.py --help
```

Result: passed.

Real model smoke test:

```shell
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python examples/streaming_demo.py /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model small.cn.streaming --model_dir /private/tmp/dolphin-models/small.cn.streaming --device cpu --chunk_duration 3 --max_chunks 1
```

Result:

```text
[0001 chunk 00:00.000-00:03.000] 诚然时代正在推陈出新
```

## Notes

- The demo streams an existing audio file in fixed-size chunks and prints each decoded chunk immediately.
- `--mode chunk` decodes each chunk independently. `--mode rolling` re-decodes accumulated audio and prints partial text.
- This is intentionally an experimental terminal demo, not a production microphone or socket streaming server.
