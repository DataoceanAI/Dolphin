# Issue #93 Language Detection Test Report

Date: 2026-06-11
Branch: `codex/issue-93-language-detection`
Issue: [#93 Language Detection Model](https://github.com/DataoceanAI/Dolphin/issues/93)

## Scope

Issue #93 asks whether Dolphin can be used for language detection only, and whether a separate lightweight language-detection model exists.

This branch makes the existing language-identification capability directly usable:

- Exports `dolphin.detect_language(model, audio, max_duration=30)` from the package top level.
- Adds CLI task mode: `--task detect_language`.
- Adds `--lid_duration` to control how many seconds are used for LID.
- Documents that LID is built into the ASR model; this package does not include a separate lightweight LID-only model.
- Adds unit tests for the Python API, package export, CLI task, and tensor-duration limiting.

## Implementation Notes

Dolphin already had an internal `detect_language(model, audio)` function backed by `ASRModel.detect_language`. The model predicts the language token and region token from the ASR model's encoder/decoder path.

During validation, sending full long audio directly into this function did not complete within about 90 seconds on CPU. The branch now limits language detection to the first `SPEECH_LENGTH` seconds by default, which is 30 seconds. This keeps language-detection-only usage responsive while preserving the option to use full audio:

- CLI full audio: `--lid_duration 0`
- Python full audio: `dolphin.detect_language(model, audio, max_duration=None)`

## Test Environment

- Host Python: `Python 3.10.9`
- Runtime used for real ASR/LID tests: `/private/tmp/dolphin-venv/bin/python`
- Runtime Python: `Python 3.10.9`
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
- Model:
  - Dolphin `base`
  - Local path: `/private/tmp/dolphin-models/base`

Note: the global environment currently has `numpy 2.2.6`, which causes noisy TensorFlow/Whisper/numba compatibility stderr during imports. Real validation used the temporary venv above with `numpy 1.23.5`.

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
collected 4 items
tests/test_language_detection.py .... [100%]
4 passed in 0.83s
```

Covered behavior:

- `detect_language` returns language and region tokens through the existing model LID path.
- `dolphin.detect_language` is exported from package top level.
- `--task detect_language` prints `language<TAB>region`.
- Tensor audio is cropped by `max_duration`.

## CLI Help Check

Command:

```bash
env TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin --help
```

Result:

```text
--lid_duration LID_DURATION
                      seconds of audio to use for language detection; set 0
                      to use full audio (default: 30)
--task {transcribe,detect_language}
                      task to run: transcribe or detect_language (default:
                      transcribe)
```

Exit code: `0`.

## Real CLI LID Tests

### Short zh-CN

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --task detect_language
```

Result:

```text
zh	CN
```

Exit code: `0`.

### Long zh-CN

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/zh-cn-long.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --task detect_language
```

Result:

```text
zh	CN
```

Exit code: `0`.

### Long hi-IN

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/hi-in-work-report.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --task detect_language
```

Result:

```text
hi	IN
```

Exit code: `0`.

## Real Python API LID Test

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -c "import dolphin; model=dolphin.load_model('base', '/private/tmp/dolphin-models/base', 'cpu'); print(dolphin.detect_language(model, '/private/tmp/dolphin-test-audio/zh-cn-demo.wav')); print(dolphin.detect_language(model, '/private/tmp/dolphin-test-audio/zh-cn-long.wav')); print(dolphin.detect_language(model, '/private/tmp/dolphin-test-audio/hi-in-work-report.wav'))"
```

Result:

```text
('zh', 'CN')
('zh', 'CN')
('hi', 'IN')
```

Exit code: `0`.

## Notes And Risks

- This branch does not introduce or claim a separate lightweight LID model.
- LID still requires loading a Dolphin ASR model.
- Default LID duration is 30 seconds to prevent long-audio language detection from sending an entire long recording into the decoder.
- `--lid_duration 0` and `max_duration=None` remain available for full-audio detection.
- ASR transcription behavior is unchanged when `--task` is omitted.

## Recommendation

This branch is ready for user review for issue #93. It answers the issue directly and gives both CLI and Python users a language-detection-only workflow.
