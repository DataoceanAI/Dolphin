# Dolphin

[Paper](https://arxiv.org/abs/2503.20212)
[Github](https://github.com/DataoceanAI/Dolphin)
[Huggingface](https://huggingface.co/DataoceanAI)
[Modelscope](https://www.modelscope.cn/organization/DataoceanAI)
[Openi](https://openi.pcl.ac.cn/DataoceanAI/Dolphin)
[Wisemodel](https://wisemodel.cn/models/lijp22/dolphin-base)

Dolphin is a multilingual, multitask ASR model developed through a collaboration between Dataocean AI and Tsinghua University. It supports 40 Eastern languages across East Asia, South Asia, Southeast Asia, and the Middle East, while also supporting 22 Chinese dialects. It is trained on over 210,000 hours of data, which includes both DataoceanAI's proprietary datasets and open-source datasets. The model can perform speech recognition, voice activity detection (VAD), segmentation, and language identification (LID).

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

There are 4 models in Dolphin, and 2 of them are available now. See details in [paper](https://arxiv.org/abs/2503.20212).

|  Model  | Parameters | Average WER | Publicly Available |
|:------:|:----------:|:------------------:|:------------------:|
|  base  |    140 M    |     33.3      |      ✅        |
| small  |   372 M    |     25.2     |      ✅       |
| medium |   910 M    |    23.1     |            |
| large  |   1679 M   |        21.6         |             |

### Languages

Dolphin supports 40 Eastern languages and 22 Chinese dialects. For a complete list of supported languages, see [languages.md](./languages.md).

## Usage

### Command-line usage

```shell
dolphin audio.wav

# Download model and specify the model path
dolphin audio.wav --model small --model_dir /data/models/dolphin/

# Specify language and region
dolphin audio.wav --model small --model_dir /data/models/dolphin/ --lang_sym "zh" --region_sym "CN"

# padding speech to 30 seconds
dolphin audio.wav --model small --model_dir /data/models/dolphin/ --lang_sym "zh" --region_sym "CN" --padding_speech true
```

### Python usage

```python
import dolphin

waveform = dolphin.load_audio("audio.wav")
model = dolphin.load_model("small", device="cuda")
result = model(waveform)

# Specify language
result = model(waveform, lang_sym="zh")

# Specify language and region
result = model(waveform, lang_sym="zh", region_sym="CN")
print(result.text)
```

## License

Dolphin's code and model weights are released under the [Apache 2.0 License](./LICENSE).
