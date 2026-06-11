# Issue #92 Disable Punctuation Test Report

Date: 2026-06-11
Branch: `codex/issue-92-disable-punctuation`
Issue: [#92 希望后面能有手动关闭标点符号的选项](https://github.com/DataoceanAI/Dolphin/issues/92)

## Scope

Issue #92 requests an option to manually disable punctuation in recognition output.

This branch adds:

- CLI option: `--remove_punctuation true`
- Python API option: `transcribe(..., remove_punctuation=True)`
- Long-audio support through `transcribe_long(..., remove_punctuation=True)`
- Shared Unicode punctuation removal for short and segmented results
- Preservation of special task/language tokens in `result.text`
- Punctuation cleanup for `result.text_nospecial`
- Filtering/cleaning punctuation tokens in `word_timestamps`
- README examples and unit tests

This is an output post-processing option. It does not change model decoding, training, or punctuation generation behavior inside the model.

## Test Environment

- Host Python: `Python 3.10.9`
- Runtime used for real ASR tests: `/private/tmp/dolphin-venv/bin/python`
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
  - `HOME=/private/tmp/dolphin-home` for long-audio VAD cache isolation
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
collected 3 items
tests/test_punctuation.py ... [100%]
3 passed in 0.89s
```

Covered behavior:

- Unicode punctuation is removed from recognition text.
- Special tokens such as `<zh><CN><asr><notimestamp>` are preserved in `result.text`.
- `text_nospecial` is cleaned.
- Pure punctuation word timestamps are removed.
- Punctuation attached to timestamp words is stripped.
- CLI parser accepts `--remove_punctuation true`.

## CLI Help Check

Command:

```bash
env TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin --help
```

Result:

```text
--remove_punctuation REMOVE_PUNCTUATION
                      remove punctuation from transcription text output
                      (default: false)
```

Exit code: `0`.

## Real ASR Tests

### Short zh-CN Python API

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -c '
import logging
import unicodedata
import dolphin
logging.getLogger("dolphin").setLevel(logging.ERROR)
model = dolphin.load_model("base", "/private/tmp/dolphin-models/base", "cpu")
result = dolphin.transcribe(model, "/private/tmp/dolphin-test-audio/zh-cn-demo.wav", predict_time=True, word_timestamp=True, remove_punctuation=True)
has_punctuation = any(unicodedata.category(ch).startswith("P") for ch in result.text_nospecial)
timestamp_punctuation = [item for item in result.word_timestamps or [] if any(unicodedata.category(ch).startswith("P") for ch in str(item.get("word", "")))]
print(result.text)
print(result.text_nospecial)
print("text_has_punctuation=", has_punctuation)
print("timestamp_punctuation_count=", len(timestamp_punctuation))
'
```

Result:

```text
<zh><CN><asr><notimestamp> 诚然  时代正在推崇初心  但文化之精髓切虚传承
诚然  时代正在推崇初心  但文化之精髓切虚传承
text_has_punctuation= False
timestamp_punctuation_count= 0
```

Exit code: `0`.

### Short zh-CN CLI

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --remove_punctuation true
```

Result:

```text
decode result, language: zh, region: CN, text: 诚然  时代正在推崇初心  但文化之精髓切虚传承
```

The logged word timestamps no longer include the comma token.

Exit code: `0`.

### Long zh-CN And hi-IN Python API

Command:

```bash
env HOME=/private/tmp/dolphin-home MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -c '
import logging
import unicodedata
import dolphin
from dolphin.transcribe import transcribe_long
logging.getLogger("dolphin").setLevel(logging.ERROR)
model = dolphin.load_model("base", "/private/tmp/dolphin-models/base", "cpu")
for label, path in [("zh-long", "/private/tmp/dolphin-test-audio/zh-cn-long.wav"), ("hi-long", "/private/tmp/dolphin-test-audio/hi-in-work-report.wav")]:
    results = transcribe_long(model, path, remove_punctuation=True)
    merged = "".join(item.text_nospecial for item in results)
    has_punctuation = any(unicodedata.category(ch).startswith("P") for ch in merged)
    ts_count = sum(1 for item in results for ts in (item.word_timestamps or []) if any(unicodedata.category(ch).startswith("P") for ch in str(ts.get("word", ""))))
    print(label, "segments=", len(results), "text_has_punctuation=", has_punctuation, "timestamp_punctuation_count=", ts_count)
    print(results[0].text_nospecial[:120] if results else "")
'
```

Result:

```text
zh-long segments= 61 text_has_punctuation= False timestamp_punctuation_count= 0
hi-long segments= 177 text_has_punctuation= False timestamp_punctuation_count= 0
मेरा नामेश कुमार मैं 
```

Exit code: `0`.

## Notes And Risks

- This option removes punctuation as Unicode punctuation categories, so it covers CJK punctuation, ASCII punctuation, and punctuation attached to Devanagari/Latin tokens.
- Special task and language tags in `result.text` are preserved.
- The model can still internally generate punctuation; this option cleans the returned/logged result.
- Because punctuation is removed after recognition, spacing may contain doubled spaces where punctuation was originally surrounded by spaces. The branch keeps this conservative behavior instead of normalizing whitespace across languages.

## Recommendation

This branch is ready for user review for issue #92. It gives CLI and Python users an explicit way to disable punctuation in returned transcription text.
