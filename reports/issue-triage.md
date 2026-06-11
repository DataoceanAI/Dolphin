# Issue Triage Record

Date: 2026-06-11
Branch: `codex/integration-issues-80-92-93`
PR: https://github.com/DataoceanAI/Dolphin/pull/109

This record tracks issues reviewed during the Codex pass so future work does not need to re-triage from scratch.

## Done In PR #109

| Issue | Status | Notes |
| --- | --- | --- |
| #80 | Done | Added CLI `--output` and `--output_format {txt,json,srt}`. |
| #92 | Done | Added CLI/API punctuation removal via `--remove_punctuation` and `remove_punctuation=True`. |
| #93 | Done | Added `dolphin.detect_language(...)`, `--task detect_language`, and `--lid_duration`. |
| #106 | Done | Added cache-level file streaming and microphone demos, streaming decode options, and SDPA chunk mask support for `small.cn.streaming`. |

## In Progress

None.

## Already Fixed Or Mostly Addressed Upstream

| Issue | Status | Notes |
| --- | --- | --- |
| #72 | Already fixed | Current `main` exports `dolphin.load_audio`. |
| #81 | Partially fixed | VAD cache directory is created before download. Remaining dependency/cache guidance can be documented. |

## Good Next Candidates

| Issue | Type | Proposed work |
| --- | --- | --- |
| #44 | Code | Wire `maxlenratio` or a replacement max decode length setting through the decode path. |
| #83 | Docs | Add word-level timestamp examples and explain `--word_timestamp`. |
| #86 / #62 | Docs | Expand hotword docs with CLI/API examples and state current ONNX status. |
| #95 | Docs/example | Add a simple FastAPI or HTTP service deployment example. |
| #42 | Docs/API | Document long-audio Python usage and safer `transcribe_long` patterns. |
| #33 | Code/API | Explore `transcribe_batch` or CLI multi-file batch inference. |
| #50 / #67 / #41 / #89 | Docs/errors | Add installation and environment troubleshooting guidance. |
| #20 / #53 / #78 | Docs | Clarify base/small vs `*.cn` dialect models and dialect tag behavior. |

## Not Directly Actionable Without Maintainer Or Model-Team Input

| Issues | Reason |
| --- | --- |
| #7 / #43 | Large model release policy. |
| #8 / #10 / #35 / #56 / #88 | Fine-tuning code release and training recipe policy. |
| #15 / #49 / #84 / #108 | Benchmark methodology or paper-result discussion. |
| #18 / #24 / #51 / #90 / #107 | New model capability, resource release, or language support decisions. |
