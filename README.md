# Dolphin

[Dolphin](https://arxiv.org/abs/2503.20212)
[Dolphin-CN-Dialect](https://arxiv.org/abs/2605.08961)
[Github](https://github.com/DataoceanAI/Dolphin)
[Huggingface](https://huggingface.co/DataoceanAI)
[Modelscope](https://www.modelscope.cn/organization/DataoceanAI)
[Openi](https://openi.pcl.ac.cn/DataoceanAI/Dolphin)
[Wisemodel](https://wisemodel.cn/models/lijp22/dolphin-base)

Dolphin is a multilingual, multitask ASR model developed through a collaboration between Dataocean AI and Tsinghua University. It supports 40 Eastern languages across East Asia, South Asia, Southeast Asia, and the Middle East, while also supporting 22 Chinese dialects. It is trained on over 210,000 hours of data, which includes both DataoceanAI's proprietary datasets and open-source datasets. The model can perform speech recognition, voice activity detection (VAD), segmentation, and language identification (LID).

## 🔥 News
- [2026-05-09] Dolphin-CN-Dialect small/base released, including base, base.streaming, small, small.prompt, small.streaming; Support Word timestamps prediction for all Dolphin models.

## Approach

![Mulitask data format](https://raw.githubusercontent.com/DataoceanAI/Dolphin/refs/heads/main/figures/multitask-data-format.png)
Dolphin largely follows the innovative design approach of [Whisper](https://github.com/openai/whisper) and [OWSM](https://github.com/espnet/espnet/tree/master/egs2/owsm_v3.1/s2t1). A joint CTC-Attention architecture is adopted, with encoder based on E-Branchformer and decoder based on standard Transformer. Several key modifications are introduced for its specific focus on ASR. Dolphin does not support translation tasks, and eliminates the use of previous text and its related tokens.

A significant enhancement in Dolphin is the introduction of a two-level language token system to better handle linguistic and regional diversity, especially in Dataocean AI dataset. The first token specifies the language (e.g., `<zh>`, `<ja>`), while the second token indicates the region (e.g., `<CN>`, `<JP>`). See details in [paper](https://arxiv.org/abs/2503.20212).


## Setup
Dolphin requires FFmpeg to convert audio file to WAV format. If FFmpeg is not installed on your system, please install it first:

```shell
# Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# MacOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

You can install the latest version of Dolphin using the following command:
```shell
pip install -U dataoceanai-dolphin
```

Alternatively, it can also be installed from the source:
```shell
pip install git+https://github.com/SpeechOceanTech/Dolphin.git 
```

## Available Models and Languages

### Models

There are 8 models in Dolphin, and 6 of them are available now. See details in [Dolphin](https://arxiv.org/abs/2503.20212) and [Dolphin-CN-Dialect](https://arxiv.org/abs/2605.08961).

|  Model  | Parameters |Publicly Available |
|:------:|:----------:|:------------------:|
|  [base](https://modelscope.cn/models/DataoceanAI/dolphin-base)  |    0.1 B    |      ✅        |
| [small](https://modelscope.cn/models/DataoceanAI/dolphin-small)  |   0.4 B    |      ✅       |
| medium |   0.9 B    |            |
| large  |   1.7B   |             |
| [base.cn](https://modelscope.cn/models/DataoceanAI/dolphin-cn-dialect-base) | 0.1 B | ✅ |
| [base.cn.streaming](https://modelscope.cn/models/DataoceanAI/dolphin-cn-dialect-small-prompt) | 0.1 B | ✅ |
| [small.cn](https://modelscope.cn/models/DataoceanAI/dolphi-cn-dialect-small) | 0.4 B | ✅ |
| [small.cn.streaming](https://modelscope.cn/models/DataoceanAI/dolphin-cn-dialect-small-streaming) | 0.4 B | ✅ |
| [small.cn.prompt](https://modelscope.cn/models/DataoceanAI/dolphin-cn-dialect-small-prompt) | 0.4 B | ✅ |

### Languages

Dolphin supports 40 Eastern languages and 22 Chinese dialects. For a complete list of supported languages, see [languages.md](./languages.md).

## Usage

### Command-line usage

```shell
# default model:small
dolphin audio.wav

# Write plain text output to a file
dolphin audio.wav --output result.txt

# Write structured output with metadata
dolphin audio.wav --output result.json --output_format json

# Write subtitle output
dolphin audio.wav --output result.srt --output_format srt

# Download model and specify the model path
dolphin audio.wav --model small.cn

# Specify language and region
dolphin audio.wav --model small.cn --lang_sym "zh" --region_sym "CN"

# Specify the hotwords file with Encoder-biased method
dolphin audio.wav --model small.cn --hotword_list_path hotwords.txt --use_deep_biasing true

# Using prompt-based model
dolphin audio.wav --model small.cn.prompt --hotword_list_path hotwords.txt --use_prompt_hotword true --use_two_stage_filter true

# predict word timestamp
dolphin audio.wav --model small.cn.prompt --word_timestamp true

```

### Python usage

```python
import dolphin
from dolphin import transcribe

model_name = 'small.cn'
model = dolphin.load_model(model_name, device="cuda")

result = transcribe(model, 'audio.wav')
print(result.text)

# Specify language
result = transcribe(model, 'audio.wav', lang_sym="zh")
print(result.text)

# Specify language and region and encoder-biased hotwords
result = transcribe(model, 'audio.wav', lang_sym="zh", region_sym="CN", hotwords=['诺香丹青牌科研胶囊'], use_deep_biasing=True, use_two_stage_filter=True)
print(result.text)

## prompt-based hotwords

model_name = 'small.cn.prompt'
model = dolphin.load_model(model_name, device="cuda")

result = transcribe(model, 'audio.wav', hotwords=['诺香丹青牌科研胶囊'], use_prompt_hotword=True, use_two_stage_filter=True, decoding_method='attention')

print(result.text)
```

## Acknowledgements

Thanks to the following excellent open-source works:

- [Espnet](https://github.com/espnet/espnet)
- [Wenet](https://github.com/wenet-e2e/wenet)
- [FunASR](https://github.com/modelscope/FunASR)
- [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)

## License

Dolphin's code and model weights are released under the [Apache 2.0 License](./LICENSE).
