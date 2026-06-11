# Issue #80 CLI Output Test Report

Date: 2026-06-11
Branch: `codex/issue-80-cli-output`
Issue: [#80 没有直接的输出文本](https://github.com/DataoceanAI/Dolphin/issues/80)

## Scope

Issue #80 requests direct CLI output to text instead of only printing mixed logs/results in the terminal. This branch adds:

- `--output PATH` for writing transcription output to a file.
- `--output_format {txt,json,srt}` for plain text, structured JSON, and subtitle output.
- SRT support requested during validation.
- README CLI examples for all three output formats.
- Unit coverage for text, JSON, SRT, stdout, and nested output file writes.

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
- VAD model for long audio:
  - `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
  - Downloaded to `/private/tmp/dolphin-home/.cache/dolphin/speech_fsmn_vad`

Note: the global environment currently has `numpy 2.2.6`, which causes noisy TensorFlow/Whisper/numba compatibility stderr during imports. Real ASR validation used the temporary venv above with `numpy 1.23.5`.

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
collected 7 items
tests/test_cli_output.py ....... [100%]
7 passed in 1.01s
```

Covered behavior:

- Short result defaults to plain `text_nospecial`.
- Long segmented results are joined as one text block for `txt`.
- JSON preserves metadata and word timestamps.
- Short-audio SRT uses first and last word timestamp.
- Long-audio SRT uses segment start/end times.
- Stdout output still works when `--output` is omitted.
- Nested output directories are created automatically.

## CLI Help Check

Command:

```bash
env TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin --help
```

Result:

```text
--output OUTPUT       write transcription output to file
--output_format {txt,json,srt}
                      output format for stdout or --output (default: txt)
```

Exit code: `0`.

## Real ASR Output Tests

### Short zh-CN Plain Text

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --output /private/tmp/dolphin-test-output/demo.txt --output_format txt
```

Result:

```text
诚然 , 时代正在推崇初心 , 但文化之精髓切虚传承。
```

Output file: `/private/tmp/dolphin-test-output/demo.txt`
Exit code: `0`.

### Short zh-CN JSON

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --output /private/tmp/dolphin-test-output/demo.json --output_format json
```

Validation:

```bash
python -m json.tool /private/tmp/dolphin-test-output/demo.json
```

Result:

- JSON parsed successfully.
- File contains `text`, `text_nospecial`, `language`, `region`, and `word_timestamps`.
- Detected language/region: `zh` / `CN`.

Output file: `/private/tmp/dolphin-test-output/demo.json`
Exit code: `0`.

### Short zh-CN SRT

Command:

```bash
env MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/zh-cn-demo.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --output /private/tmp/dolphin-test-output/demo.srt --output_format srt
```

Result:

```srt
1
00:00:00,600 --> 00:00:05,867
诚然 , 时代正在推崇初心 , 但文化之精髓切虚传承。
```

Output file: `/private/tmp/dolphin-test-output/demo.srt`
Exit code: `0`.

### Long zh-CN SRT

Command:

```bash
env HOME=/private/tmp/dolphin-home MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/zh-cn-long.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --output /private/tmp/dolphin-test-output/zh-cn-long.srt --output_format srt
```

Result:

- Output file: `/private/tmp/dolphin-test-output/zh-cn-long.srt`
- File size: `12K`
- Cue count: `60`
- First cue: `00:00:03,150 --> 00:00:05,000`
- Last cue: `00:12:26,870 --> 00:12:28,680`
- Exit code: `0`

### Long hi-IN SRT

Command:

```bash
env HOME=/private/tmp/dolphin-home MODELSCOPE_CACHE=/private/tmp/modelscope-cache NUMBA_CACHE_DIR=/private/tmp/numba-cache TRANSFORMERS_NO_TF=1 USE_TF=0 /private/tmp/dolphin-venv/bin/python -m dolphin /private/tmp/dolphin-test-audio/hi-in-work-report.wav --model base --model_dir /private/tmp/dolphin-models/base --device cpu --output /private/tmp/dolphin-test-output/hi-in-work-report.srt --output_format srt
```

Result:

- Output file: `/private/tmp/dolphin-test-output/hi-in-work-report.srt`
- File size: `32K`
- Cue count: `175`
- First cue: `00:00:02,690 --> 00:00:08,900`
- Last cue: `00:21:11,550 --> 00:21:13,240`
- Exit code: `0`

## SRT Structural Validation

Command:

```bash
python -c 'import re, pathlib
for path in ["/private/tmp/dolphin-test-output/demo.srt", "/private/tmp/dolphin-test-output/zh-cn-long.srt", "/private/tmp/dolphin-test-output/hi-in-work-report.srt"]:
    blocks = [b for b in pathlib.Path(path).read_text(encoding="utf-8").strip().split("\n\n") if b.strip()]
    prev_end = -1.0
    ok = True
    for expected, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if int(lines[0]) != expected:
            ok = False
            break
        m = re.match(r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)$", lines[1])
        if not m:
            ok = False
            break
        vals = list(map(int, m.groups()))
        start = vals[0]*3600 + vals[1]*60 + vals[2] + vals[3]/1000
        end = vals[4]*3600 + vals[5]*60 + vals[6] + vals[7]/1000
        if end < start or start < prev_end:
            ok = False
            break
        prev_end = end
    print(f"{path}: cues={len(blocks)}, valid={ok}, last_end={prev_end:.3f}s")'
```

Result:

```text
/private/tmp/dolphin-test-output/demo.srt: cues=1, valid=True, last_end=5.867s
/private/tmp/dolphin-test-output/zh-cn-long.srt: cues=60, valid=True, last_end=748.680s
/private/tmp/dolphin-test-output/hi-in-work-report.srt: cues=175, valid=True, last_end=1273.240s
```

## Notes And Risks

- This change does not alter ASR decoding, VAD, language detection, or timestamp generation.
- Long-audio SRT uses VAD segment start/end times from `TranscribeSegmentResult`.
- Short-audio SRT uses first and last word timestamp from `TranscribeResult.word_timestamps`.
- If no text is returned, SRT output is empty rather than creating blank cues.
- JSON output is intentionally full dataclass data, including timestamps and language metadata.
- The existing CLI still prints the formatted result when `--output` is omitted.
- Some upstream ASR logs still go to stderr/stdout during real inference. The new file output keeps the requested transcription artifact clean.

## Recommendation

This branch is ready for user review for issue #80. The new SRT format also passed short and long real-audio validation.
