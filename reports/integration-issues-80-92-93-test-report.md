# Integration Test Report: Issues #80, #92, #93

Date: 2026-06-11
Branch: `codex/integration-issues-80-92-93`
Base: `main` at `78ea615`

## Scope

This branch integrates the three issue branches that are ready for review:

| Issue | Branch | Commit | Summary |
| --- | --- | --- | --- |
| #80 | `codex/issue-80-cli-output` | `5fff9c8` | CLI output files and `txt/json/srt` formats |
| #93 | `codex/issue-93-language-detection` | `7f6d188` | CLI/API language-detection-only mode |
| #92 | `codex/issue-92-disable-punctuation` | `a256cf7` | CLI/API punctuation removal option |

The integration branch cherry-picks these changes onto current `main` and resolves overlapping CLI parser edits in `dolphin/transcribe.py`.

## Integration Commits

```text
8ed0086 Add punctuation removal option
5e429b2 Expose language detection task
9a36c51 Add CLI output formats
78ea615 Merge pull request #105 from DataoceanAI/update-readme
```

## Conflict Resolution

The only conflicts were in `dolphin/transcribe.py`, where all three branches extended imports and CLI options. The resolved CLI keeps all options together:

- `--remove_punctuation`
- `--lid_duration`
- `--task {transcribe,detect_language}`
- `--output`
- `--output_format {txt,json,srt}`

The final CLI flow is:

1. Load model and audio.
2. If `--task detect_language`, print `language<TAB>region` and return.
3. Otherwise transcribe normally.
4. Optionally remove punctuation from returned text and timestamps.
5. Emit output to stdout or `--output` in `txt`, `json`, or `srt`.

## Test Environment

- Runtime used for validation: `/private/tmp/dolphin-venv/bin/python`
- Python: `3.10.9`
- Key packages:
  - `numpy 1.23.5`
  - `torch 2.4.1`
  - `torchaudio 2.4.1`
  - `modelscope 1.36.3`
  - `funasr 1.1.5`
- Environment variables used:
  - `TRANSFORMERS_NO_TF=1`
  - `USE_TF=0`
  - `MODELSCOPE_CACHE=/private/tmp/modelscope-cache`
  - `NUMBA_CACHE_DIR=/private/tmp/numba-cache`
  - `HOME=/private/tmp/dolphin-home` for long-audio VAD cache isolation
- Model:
  - Dolphin `base`
  - Local path: `/private/tmp/dolphin-models/base`

## Test Audio

| Case | Source | Local file | Duration |
| --- | --- | --- | --- |
| Short zh-CN | `https://so-algorithm-test.oss-cn-beijing.aliyuncs.com/samples/asr/zh-cn-demo.wav` | `/private/tmp/dolphin-test-audio/zh-cn-demo.wav` | 6.267s |
| Long zh-CN | `https://so-algorithm-test.oss-cn-beijing.aliyuncs.com/samples/asr/zh-cn-long.wav` | `/private/tmp/dolphin-test-audio/zh-cn-long.wav` | 752.880s |
| Long hi-IN | `https://so-algorithm-test.oss-cn-beijing.aliyuncs.com/samples/asr/hi_in/%E5%8D%95%E4%BA%BA%E5%BD%95%E9%9F%B3_%E5%B7%A5%E4%BD%9C%E6%8A%A5%E5%91%8A_12112.wav` | `/private/tmp/dolphin-test-audio/hi-in-work-report.wav` | 1274.009s |

## Automated Tests

Command:

```bash
env TRANSFORMERS_NO_TF=1 USE_TF=0 python -m pytest
```

Result:

```text
collected 14 items
tests/test_cli_output.py .......      [ 50%]
tests/test_language_detection.py .... [ 78%]
tests/test_punctuation.py ...         [100%]
14 passed
```

Static whitespace check:

```bash
git diff --check
```

Result: passed.

## CLI Help Check

Command:

```bash
env TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin --help
```

Result: help output includes all integrated options:

```text
--remove_punctuation REMOVE_PUNCTUATION
--lid_duration LID_DURATION
--task {transcribe,detect_language}
--output OUTPUT
--output_format {txt,json,srt}
```

Exit code: `0`.

## Real Language Detection Tests

All three requested audio files were tested with:

```bash
/private/tmp/dolphin-venv/bin/python -m dolphin AUDIO --model base --model_dir /private/tmp/dolphin-models/base --device cpu --task detect_language
```

Results:

| Audio | Result |
| --- | --- |
| Short zh-CN | `zh	CN` |
| Long zh-CN | `zh	CN` |
| Long hi-IN | `hi	IN` |

All commands exited with code `0`.

## Real SRT And Punctuation Tests

All three requested audio files were tested with integrated SRT output and punctuation removal:

```bash
/private/tmp/dolphin-venv/bin/python -m dolphin AUDIO --model base --model_dir /private/tmp/dolphin-models/base --device cpu --remove_punctuation true --output OUTPUT.srt --output_format srt
```

Output files:

| Audio | Output | Cue count | Last end | Body punctuation |
| --- | --- | ---: | ---: | ---: |
| Short zh-CN | `/private/tmp/dolphin-integration-test-output/demo-clean.srt` | 1 | 5.867s | 0 |
| Long zh-CN | `/private/tmp/dolphin-integration-test-output/zh-cn-long-clean.srt` | 60 | 748.680s | 0 |
| Long hi-IN | `/private/tmp/dolphin-integration-test-output/hi-in-clean.srt` | 174 | 1273.240s | 0 |

Structural validation checked:

- cue numbers are continuous from 1;
- time lines match `HH:MM:SS,mmm --> HH:MM:SS,mmm`;
- cue start/end times are ordered and non-overlapping;
- subtitle body text has no Unicode punctuation.

Result:

```text
demo-clean: cues=1, valid=True, last_end=5.867s, body_punctuation=0
zh-cn-long-clean: cues=60, valid=True, last_end=748.680s, body_punctuation=0
hi-in-clean: cues=174, valid=True, last_end=1273.240s, body_punctuation=0
```

## Notes And Risks

- The integration does not change model weights or decoding internals.
- `--remove_punctuation` is output post-processing, so spacing may contain doubled spaces where punctuation was removed.
- Language detection still loads a Dolphin ASR model; this does not add a separate lightweight LID-only model.
- The global Python environment has `numpy 2.2.6`, which emits noisy TensorFlow/Whisper/numba import warnings. Real validation used a temporary venv with `numpy 1.23.5`.

## Recommendation

For easiest review, keep the three per-issue branches available. For easiest merge, use `codex/integration-issues-80-92-93` after reviewing this report, because it already resolves the overlapping CLI changes and has been tested with the requested short audio, long zh-CN audio, long hi-IN audio, and SRT output.
