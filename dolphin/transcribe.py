# encoding: utf8

import logging
import warnings
import json

LOGGING_FORMAT="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
warnings.filterwarnings("ignore", category=FutureWarning)
# filter framework internal logs
logging.getLogger("espnet").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("dolphin").setLevel(logging.INFO)

import math
import yaml
import tqdm
import pydub
import hashlib
import os.path
import argparse
import dataclasses
import numpy as np
from pathlib import Path
from argparse import Namespace
from distutils.util import strtobool
from typing import Union, Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import modelscope
from modelscope.models.audio.funasr.model import GenericFunASR

from dolphin.audio import convert_audio
# from dolphin.model import DolphinSpeech2Text, TranscribeResult, TranscribeSegmentResult
from dolphin.processor import extract_feats
from dolphin.model import (ASRModel, init_speech_model,
                           TranscribeResult, TranscribeSegmentResult)
from dolphin.languages import LANGUAGE_REGION_CODES, LANGUAGE_CODES
from dolphin.constants import SPEECH_LENGTH
from dolphin.model_registry import MODELS
from dolphin.tokenizer import init_tokenizer, BaseTokenizer, BpeTokenizer
from dolphin.hotword import HotwordEncoder, prepare_hotword_tensor, two_stage_filtering


logger = logging.getLogger("dolphin")

VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"



def str2bool(value: str) -> bool:

    return bool(strtobool(value))


def parser_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str, help="audio file path")
    parser.add_argument("--model", type=str, default="small", help="model name (default: small)")
    parser.add_argument("--model_dir", type=Path, default=None, help="model checkpoint download directory")
    parser.add_argument("--lang_sym", type=str, default=None, help="language symbol (e.g. zh)")
    parser.add_argument("--region_sym", type=str, default=None, help="region symbol (e.g. CN)")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: None)")
    parser.add_argument("--predict_time", type=str2bool, default=True, help="whether predict timestamp (default: true)")
    parser.add_argument("--word_timestamp", type=str2bool, default=True, help="whether predict word timestamp (default: true)")
    parser.add_argument("--beam_size", type=int, default=10, help="number of beams in beam search (default: 10)")
    parser.add_argument("--decoding_method", type=str, default="attention_rescoring",
                        help="decoding methods, supports: attention, attention_rescoring (default: attention_rescoring)")
    parser.add_argument("--maxlenratio", type=float, default=0.0, help="deprecated, Input length ratio to obtain max output length (default: 0.0)")
    parser.add_argument("--padding_speech", type=str2bool, default=False, help="deprecated, whether padding speech to 30 seconds (default: false)")
    parser.add_argument("--normalize_length", type=str2bool, default=False, help="deprecated, whether to normalize length (default: false)")
    parser.add_argument("--hotword_str", type=str, default=None, help="comma-separated hotwords (e.g. '张三,李四')")
    parser.add_argument("--hotword_list_path", type=Path, default=None, help="path to hotword list file")
    parser.add_argument("--use_deep_biasing", type=str2bool, default=False, help="use deep biasing for hotwords (default: false)")
    parser.add_argument("--use_two_stage_filter", type=str2bool, default=False, help="use two-stage filtering for hotwords (default: false)")
    parser.add_argument("--use_prompt_hotword", type=str2bool, default=False, help="use prompt-based hotword (default: false)")
    parser.add_argument("--prompt_filter_threshold", type=float, default=-2.0, help="filter threshold for prompt hotwords (default: -2.0)")
    parser.add_argument("--output", type=Path, default=None, help="write transcription output to file")
    parser.add_argument(
        "--output_format",
        type=str,
        default="txt",
        choices=("txt", "json", "srt"),
        help="output format for stdout or --output (default: txt)",
    )

    args = parser.parse_args()
    return args


def detect_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def seconds_to_hms(total_seconds: int):
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def convert_v1_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    def generate_pe(d_model: int):
        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        return pe

    d_model = state_dict["decoder.output_layer.weight"].size(-1)
    pe = generate_pe(d_model)

    state_dict["encoder.global_cmvn.mean"] = state_dict["normalize.mean"]
    state_dict["encoder.global_cmvn.std"] = state_dict["normalize.std"]
    state_dict["decoder.embed.1.pe"] = pe

    del state_dict["normalize.mean"]
    del state_dict["normalize.std"]
    del state_dict["frontend.logmel.melmat"]

    return state_dict


def load_model(
    model_name: str,
    model_dir: Union[Path, str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> ASRModel:
    """
    Load model.

    Args:
        model_name: model name (e.g. small)
        model_dir: model download directory
        device: the accelerator device

    Returns:
        ASRModel instance
    """
    if model_dir is None:
        model_dir = os.path.expanduser(f"~/.cache/dolphin/{model_name}")
        
    if device is None:
        device = detect_device()
        logger.info(f"auto detect device: {device}")

    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)

    download_model = True
    model_ckpt_file = model_dir / f"{model_name}.pt"
    if model_ckpt_file.exists():
        with open(model_ckpt_file, "rb") as f:
            model_bytes = f.read()

        if hashlib.sha256(model_bytes).hexdigest() == MODELS[model_name]["sha256"]:
            download_model = False
        else:
            logger.warning("model SHA256 checksum mismatch, redownload model...")
    else:
        logger.info(f"model {model_ckpt_file} not exists, download...")

    if download_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        modelscope.snapshot_download(
            model_id=MODELS[model_name]["model_id"],
            local_dir=model_dir,
            allow_file_pattern=None,
            repo_type="model",
        )

    assert (model_dir / "train.yaml").exists(), "model config not found, please redownload model!"
    with open(model_dir / "train.yaml") as f:
        configs = yaml.load(f, Loader=yaml.Loader)
        if configs["cmvn"] == "global_mvn":
            configs["cmvn_conf"]["cmvn_file"] = str(model_dir / "feats_stats.npz")
        else:
            configs["cmvn_conf"]["cmvn_file"] = str(model_dir / "global_cmvn")


        configs["tokenizer_conf"]["symbol_table_path"] = str(model_dir / "units.txt")
        if "bpe_path" in configs["tokenizer_conf"]:
            configs["tokenizer_conf"]["bpe_path"] = str(model_dir / 'bpe.model')

    model = init_speech_model(configs)
    state_dict = torch.load(model_dir / f"{model_name}.pt", map_location="cpu")
    state_dict = convert_v1_state_dict(state_dict) if "normalize.mean" in state_dict else state_dict

    # Handle context_module (hotword encoder) loading
    context_module_state = {}
    non_context_state = {}
    for k, v in state_dict.items():
        if k.startswith("context_module."):
            context_module_state[k.replace("context_module.", "")] = v
        else:
            non_context_state[k] = v

    model.load_state_dict(non_context_state)

    # Initialize and load context_module if present
    if context_module_state and configs.get("context_module") == "cppn":
        ctx_conf = configs.get("context_module_conf", {})
        encoder_conf = configs.get("encoder_conf", {})
        encoder_d_model = encoder_conf.get("output_size", 768)
        ctx_embedding_size = ctx_conf.get("embedding_size", encoder_d_model)
        hotword_encoder = HotwordEncoder(
            vocab_size=model.vocab_size,
            embedding_size=ctx_embedding_size,
            encoder_layers=ctx_conf.get("encoder_layers", 2),
            attention_heads=ctx_conf.get("attention_heads", 4),
            dropout_rate=ctx_conf.get("dropout_rate", 0.0)
        )
        # Filter out keys that aren't in HotwordEncoder (context_decoder is for training only)
        hotword_keys = set(hotword_encoder.state_dict().keys())
        filtered_context_state = {
            k: v for k, v in context_module_state.items()
            if k in hotword_keys
        }
        hotword_encoder.load_state_dict(filtered_context_state, strict=False)
        model.context_module = hotword_encoder
        logger.info("Loaded context_module (hotword encoder) from model")

    model.model_configs = configs
    model = model.to(device)
    model.device = device
    model.eval()

    # Compatible with old version model framework
    if "layer_norm_eps" in model.model_configs:
        layer_norm_eps = float(model.model_configs["layer_norm_eps"])
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.eps = layer_norm_eps

    return model


def validate_lang_region(lang_sym: str, region_sym: str):

    if all([lang_sym, region_sym]):
        if f"{lang_sym}-{region_sym}" not in LANGUAGE_REGION_CODES:
            raise Exception("Unsupported language or region!")
    elif any([lang_sym, region_sym]):
        if lang_sym is not None and region_sym is None:
            assert lang_sym in LANGUAGE_CODES, "Unsupported language!"
        elif lang_sym is None and region_sym is not None:
            assert False, "If you specify a dialect, you must configure the language!"

    return True


def transcribe_long(
    model: ASRModel,
    audio: str,
    lang_sym: str = None,
    region_sym: str = None,
    predict_time: bool = True,
    word_timestamp: bool =True,
    padding_speech: bool = False,
    decoding_method: str = "attention_rescoring",
    beam_size: int = 10,
    hotwords: Optional[List[str]] = None,
    use_deep_biasing: bool = False,
    use_two_stage_filter: bool = False,
    use_prompt_hotword: bool = False,
    prompt_filter_threshold: float = -2.0,
    **kwargs,
) -> List[TranscribeSegmentResult]:
    """
    Transcribe audio to text.

    Args:
        model: model instance
        audio: audio path
        lang_sym: language symbol (e.g. zh)
        region_sym: region symbol (e.g. CN)
        predict_time: whether predict timestamp (default: false)
        word_timestamp: whether predict word timestamp (default: true)
        padding_speech: deprecated, whether padding speech to 30 seconds (default: false)
        decoding_method: decoding methods, supports: attention, attention_rescoring (default: attention_rescoring)
        hotwords: list of hotword strings (default: None)
        use_deep_biasing: whether use deep biasing for hotwords (default: false)
        use_two_stage_filter: whether use two-stage filtering (default: false)
        use_prompt_hotword: whether use prompt-based hotword (default: false)
        prompt_filter_threshold: filter threshold for prompt hotwords (default: -2.0)

    Returns:
        List[TranscribeSegmentResult]
    """
    results = []

    validate_lang_region(lang_sym, region_sym)

    logging.info("download vad model")
    vad_model_dir = Path(os.path.expanduser("~/.cache/dolphin/speech_fsmn_vad"))
    vad_model_dir.mkdir(parents=True, exist_ok=True)
    if not (vad_model_dir / "model.pt").exists():
        modelscope.snapshot_download(
            model_id=VAD_MODEL,
            local_dir=vad_model_dir,
            allow_file_pattern=None,
            repo_type="model",
        )

    logger.info("loading vad model")
    vad_model = GenericFunASR(
        vad_model_dir,
        max_single_segment_time=SPEECH_LENGTH*1000,
        device="cpu",
        disable_update=True,
    )

    # convert audio to sample rate 16k Mono channel audio
    tmp_audio = f"{audio}.wav"
    convert_audio(audio, tmp_audio)

    logger.info("run vad model")
    segments = vad_model(input=tmp_audio, disable_pbar=True)[0]["value"]

    logger.info("decoding...")
    tokenizer = init_tokenizer(model.model_configs)

    # Process hotwords if provided
    hotword_token_ids = None
    hotword_encoder = None
    if hotwords:
        # Convert hotword strings to token ID sequences
        hotword_token_ids = []
        for hw in hotwords:
            # For hotwords, always use char-level splitting
            # BPE doesn't work well for Chinese hotwords
            tokens = list(hw)
            ids = tokenizer.tokens2ids(tokens)
            if ids:
                hotword_token_ids.append(ids)

        logger.info(f"Using hotwords with {len(hotword_token_ids)} hotwords")

    audio_segment = pydub.AudioSegment.from_wav(tmp_audio)
    for seg in segments:
        s, e = seg
        raw_data = audio_segment[s:e].set_channels(1).raw_data
        waveform: torch.Tensor = torch.frombuffer(raw_data, dtype=torch.int16).to(torch.float32) / 32768.0
        batch = extract_feats([waveform.unsqueeze(0)], model.model_configs)
        batch["feats"] = batch["feats"].to(model.device)
        batch["feats_lengths"] = batch["feats_lengths"].to(model.device)

        lang_tokens = [f"<{lang_sym}>"] if lang_sym is not None else None
        region_tokens = [f"<{region_sym}>"] if region_sym is not None else None
        need_timestamp = True if predict_time and model.model_configs.get("support_timestamp", False) else False

        decoding_infos = {
            "tokenizer": tokenizer,
            "langs": lang_tokens,
            "regions": region_tokens,
            "need_timestamp": need_timestamp,
            "word_timestamp": word_timestamp
        }

        if hotword_token_ids and hotword_encoder:
            decoding_infos["hotwords"] = hotword_token_ids
            decoding_infos["hotword_encoder"] = hotword_encoder
            decoding_infos["use_deep_biasing"] = True
            decoding_infos["use_two_stage_filter"] = use_two_stage_filter

        if use_prompt_hotword and hotword_token_ids:
            decoding_infos["hotwords"] = hotword_token_ids
            decoding_infos["use_prompt_hotword"] = True
            decoding_infos["filter_threshold"] = prompt_filter_threshold
            decoding_infos["use_two_stage_filter"] = use_two_stage_filter
            decoding_infos["tokenizer"] = tokenizer
            if decoding_method != 'attention':
                logger.warning('[decoding_method -> attention] is much more better, maybe considering change it with --decoding_method attention')

        ret = model.decode(
            methods=[decoding_method],
            speech=batch["feats"],
            speech_lengths=batch["feats_lengths"],
            beam_size=beam_size,
            infos=decoding_infos
        )
        tokens = ret[decoding_method][0].tokens
        word_ts = ret[decoding_method][0].times

        if use_prompt_hotword:
            hotwords_text, tokens = _filter_prompt_tokens(tokens, tokenizer)
            logger.info(f"Detected hotwords: {hotwords_text}")
            
        nonspecial_tokens = _filter_nonspecial_tokens(tokens, tokenizer)
        lang = tokenizer.ids2tokens([tokens[0]])[0][1:-1]
        region = tokenizer.ids2tokens([tokens[1]])[0][1:-1]
        result = TranscribeResult(
            text=tokenizer.detokenize(tokens)[0],
            text_nospecial=tokenizer.detokenize(nonspecial_tokens)[0],
            language=lang,
            region=region,
            word_timestamps=word_ts,
        )

        st = seconds_to_hms(s/1000)
        et = seconds_to_hms(e/1000)
        logger.info(f"segment: {st} - {et}, lang: {result.language}, region: {result.region}, text: {result.text_nospecial} Timestamp: {result.word_timestamps}")
        result_json = dataclasses.asdict(result)
        result_json.update({
            "start": round(s/1000, 2),
            "end": round(e/1000, 2)
        })
        segment_result = TranscribeSegmentResult(**result_json)
        results.append(segment_result)

    # clean tmp audio file
    Path(tmp_audio).unlink(missing_ok=True)

    return results


def _parse_hotwords(
    hotword_str: Optional[str] = None,
    hotword_list_path: Optional[Path] = None,
) -> Optional[List[str]]:
    """Parse hotwords from string or file.

    Args:
        hotword_str: comma-separated hotwords (e.g. '张三,李四')
        hotword_list_path: path to hotword list file

    Returns:
        List of hotword strings, or None if no hotwords provided
    """
    if hotword_str is None and hotword_list_path is None:
        return None

    hotwords = []
    if hotword_str:
        hotwords.extend(h.strip() for h in hotword_str.split(",") if h.strip())

    if hotword_list_path:
        if hotword_list_path.exists():
            with open(hotword_list_path, "r", encoding="utf-8") as f:
                hotwords.extend(line.strip() for line in f if line.strip())
        else:
            logger.warning(f"hotword list file not found: {hotword_list_path}")

    return hotwords if hotwords else None


def _prepare_prompt_hotwords(
    batch_size: int,
    ctc_probs_list: List[torch.Tensor],
    hotword_token_ids: List[List[int]],
    tokenizer: BaseTokenizer,
    use_two_stage_filter: bool = True,
    filter_threshold: float = -4.0,
    filter_window_size: int = 64,
    max_hotwords_without_filter: int = 10
) -> List[torch.Tensor]:
    """Prepare per-sample prompt tensors from hotwords based on CTC posterior filtering.

    For each sample in the batch, applies two-stage filtering to select relevant hotwords,
    then prepends [PROMPT_START, hotwords..., PROMPT_END] to create the prompt.

    Args:
        batch_size: Number of samples in the batch.
        ctc_probs_list: List of CTC log probability tensors, one per sample (T, vocab_size).
        hotword_token_ids: All hotword token ID sequences.
        tokenizer: Tokenizer instance for getting PROMPT_START/PROMPT_END token IDs.
        use_two_stage_filter: Whether to apply two-stage filtering (default: False).
        filter_threshold: Threshold for two-stage filtering.
        filter_window_size: Window size for PSC/SOC calculation.
        max_hotwords_without_filter: Maximum hotwords to use when filtering is disabled (default: 10).

    Returns:
        List of prompt tensors, one per sample. Each tensor contains token IDs.
    """
    device = ctc_probs_list[0].device if ctc_probs_list else torch.device('cpu')
    prompt_start_id = tokenizer.tokens2ids(["<PROMPT_START>"])[0]
    prompt_end_id = tokenizer.tokens2ids(["<PROMPT_END>"])[0]

    prompts = []
    for b in range(batch_size):
        ctc_posterior = ctc_probs_list[b]
        if ctc_posterior.dim() == 3:
            ctc_posterior = ctc_posterior.squeeze(0)

        # Apply two-stage filtering for this sample
        if use_two_stage_filter:
            filtered = two_stage_filtering(
                hotword_token_ids,
                ctc_posterior,
                filter_threshold=filter_threshold,
                filter_window_size=filter_window_size
            )
        else:
            # Without filtering, use very limited hotwords to avoid OOM
            # WARNING: This is not recommended for production use
            logger.warning(
                f"Prompt hotword without two-stage filtering may cause OOM or poor results. "
                f"Using only first {max_hotwords_without_filter} hotwords. "
                f"Recommend using use_two_stage_filter=True."
            )
            filtered = hotword_token_ids[:max_hotwords_without_filter]

        # Build prompt: [PROMPT_START, hotword_ids..., PROMPT_END]
        prompt_tokens = [prompt_start_id]
        for hw_ids in filtered:
            prompt_tokens.extend(hw_ids)
        prompt_tokens.append(prompt_end_id)

        prompts.append(torch.tensor(prompt_tokens, dtype=torch.long, device=device))

    return prompts


def _filter_nonspecial_tokens(tokens: List[int], tokenizer: BaseTokenizer) -> List[int]:
    """
    Filter out special tokens from the token list.

    Args:
        tokens: list of token ids
        tokenizer: tokenizer instance

    Returns:
        list of non-special token ids
    """
    last_time_id = tokenizer.tokens2ids(["<30.00>"])[0]
    nonspecial_tokens = list(filter(lambda x: x > last_time_id , tokens))

    return nonspecial_tokens

def _filter_prompt_tokens(tokens: List[int], tokenizer: BaseTokenizer) -> Tuple[str, List[int]]:
    """
    Filter out prompt tokens from the token list and get the detected hotwords text.

    Args:
        tokens: list of token ids
        tokenizer: tokenizer instance

    Returns:
        detected hotwords text;
        list of non-special token ids
    """
    prompt_start_id = tokenizer.tokens2ids(["<PROMPT_START>"])[0]
    prompt_end_id = tokenizer.tokens2ids(["<PROMPT_END>"])[0]
    prompt_start_pos, prompt_end_pos = tokens.index(prompt_start_id), tokens.index(prompt_end_id)
    hotword_tokens = tokens[prompt_start_pos+1:prompt_end_pos]
    hotwords_text = tokenizer.detokenize(hotword_tokens)[0]
    tokens = tokens[prompt_end_pos+1:]

    return hotwords_text, tokens

def detect_language(model: ASRModel, audio: str) -> Tuple[str, str]:
    """
    Detect language and dialect.
    """
    batch = extract_feats([audio], model.model_configs)
    batch["feats"] = batch["feats"].to(model.device)
    batch["feats_lengths"] = batch["feats_lengths"].to(model.device)

    ret = model.detect_language(batch["feats"], batch["feats_lengths"])
    tokenizer = init_tokenizer(model.model_configs)
    lang, dialect = tokenizer.ids2tokens(ret.tolist()[0])
    lang = lang[1:-1]
    dialect = dialect[1:-1]

    return (lang, dialect)


def _format_cli_output(
    result: Union[TranscribeResult, List[TranscribeSegmentResult]],
    output_format: str = "txt",
) -> str:
    if output_format == "json":
        if isinstance(result, list):
            payload = [dataclasses.asdict(item) for item in result]
        else:
            payload = dataclasses.asdict(result)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if output_format == "srt":
        return _format_srt_output(result)

    if isinstance(result, list):
        return "\n".join(item.text_nospecial for item in result)

    return result.text_nospecial


def _seconds_to_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_srt_output(result: Union[TranscribeResult, List[TranscribeSegmentResult]]) -> str:
    if isinstance(result, list):
        cues = [
            (segment.start, segment.end, segment.text_nospecial)
            for segment in result
            if segment.text_nospecial
        ]
    else:
        timestamps = result.word_timestamps or []
        if timestamps:
            start = float(timestamps[0].get("start", 0.0))
            end = float(timestamps[-1].get("end", start))
        else:
            start = 0.0
            end = 0.0
        cues = [(start, end, result.text_nospecial)] if result.text_nospecial else []

    blocks = []
    for index, (start, end, text) in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n"
            f"{_seconds_to_srt_time(start)} --> {_seconds_to_srt_time(end)}\n"
            f"{text}"
        )

    return "\n\n".join(blocks)


def _emit_cli_output(
    result: Union[TranscribeResult, List[TranscribeSegmentResult]],
    output_format: str,
    output: Optional[Path],
):
    text = _format_cli_output(result, output_format)
    if output is None:
        print(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


def transcribe(
    model: ASRModel,
    audio: str,
    lang_sym: str = None,
    region_sym: str = None,
    predict_time: bool = False,
    word_timestamp: bool =False,
    padding_speech: bool = False,
    decoding_method: str = "attention_rescoring",
    beam_size: int = 10,
    hotwords: Optional[List[str]] = None,
    use_deep_biasing: bool = False,
    use_two_stage_filter: bool = False,
    use_prompt_hotword: bool = False,
    prompt_filter_threshold: float = -4.0,
    **kwargs,
) -> TranscribeResult:
    """
    Transcribe audio to text.

    Args:
        model: model instance
        audio: audio path
        lang_sym: language symbol (e.g. zh)
        region_sym: region symbol (e.g. CN)
        predict_time: whether predict timestamp (default: true)
        word_timestamp: whether predict word timestamp (default: true)
        padding_speech: deprecated, whether padding speech to 30 seconds (default: false)
        decoding_method: decoding methods, supports: attention, attention_rescoring (default: attention_rescoring)
        hotwords: list of hotword strings (default: None)
        use_deep_biasing: whether use deep biasing for hotwords (default: true)
        use_two_stage_filter: whether use two-stage filtering (default: false)
        use_prompt_hotword: whether use prompt-based hotword (default: false)
        prompt_filter_threshold: filter threshold for prompt hotwords (default: -4.0)

    Returns:
        TranscribeResult
    """
    validate_lang_region(lang_sym, region_sym)

    logger.info("decoding...")
    batch = extract_feats([audio], model.model_configs)
    batch["feats"] = batch["feats"].to(model.device)
    batch["feats_lengths"] = batch["feats_lengths"].to(model.device)

    lang_tokens = [f"<{lang_sym}>"] if lang_sym is not None else None
    region_tokens = [f"<{region_sym}>"] if region_sym is not None else None
    need_timestamp = True if predict_time and model.model_configs.get("support_timestamp", False) else False

    tokenizer = init_tokenizer(model.model_configs)
    
    decoding_infos = {
        "tokenizer": tokenizer,
        "langs": lang_tokens,
        "regions": region_tokens,
        "need_timestamp": need_timestamp,
        "word_timestamp": word_timestamp
    }

    # Process hotwords if provided
    if hotwords and use_deep_biasing:
        # Convert hotword strings to token ID sequences
        hotword_token_ids = []
        for hw in hotwords:
            # For hotwords, always use char-level splitting
            # BPE doesn't work well for Chinese hotwords
            tokens = list(hw)
            ids = tokenizer.tokens2ids(tokens)
            if ids:
                hotword_token_ids.append(ids)

        if hotword_token_ids:
            # Initialize HotwordEncoder
            encoder_conf = model.model_configs.get("encoder_conf", {})
            encoder_d_model = encoder_conf.get("output_size", 256)
            hotword_encoder = HotwordEncoder(
                vocab_size=tokenizer.vocab_size(),
                embedding_size=encoder_d_model,
                encoder_layers=2,
                attention_heads=4,
                dropout_rate=0.0
            ).to(model.device)

            decoding_infos["hotwords"] = hotword_token_ids
            decoding_infos["hotword_encoder"] = hotword_encoder
            decoding_infos["use_deep_biasing"] = True
            decoding_infos["use_two_stage_filter"] = use_two_stage_filter
            logger.info(f"Using deep biasing with {len(hotword_token_ids)} hotwords")

    # Also handle prompt-based hotword if requested
    if hotwords and use_prompt_hotword:
        # Convert hotword strings to token ID sequences
        hotword_token_ids = []
        for hw in hotwords:
            tokens = list(hw)
            ids = tokenizer.tokens2ids(tokens)
            if ids:
                hotword_token_ids.append(ids)

        if hotword_token_ids:
            decoding_infos["hotwords"] = hotword_token_ids
            decoding_infos["use_prompt_hotword"] = True
            decoding_infos["filter_threshold"] = prompt_filter_threshold
            decoding_infos["use_two_stage_filter"] = use_two_stage_filter
            logger.info(f"Using prompt hotword with {len(hotword_token_ids)} hotwords")
            if decoding_method != 'attention':
                logger.warning('[decoding_method -> attention] is much more better, maybe considering change it with --decoding_method attention')


    ret = model.decode(
        methods=[decoding_method],
        speech=batch["feats"],
        speech_lengths=batch["feats_lengths"],
        beam_size=beam_size,
        infos=decoding_infos
    )

    tokens = ret[decoding_method][0].tokens
    word_ts = ret[decoding_method][0].times
    if use_prompt_hotword or "<PROMPT_START>" in tokenizer.symbol_table:
        hotwords_text, tokens = _filter_prompt_tokens(tokens, tokenizer)
        if hotwords_text:
            logger.info(f"Detected hotwords: {hotwords_text}")
    nonspecial_tokens = _filter_nonspecial_tokens(tokens, tokenizer)
    lang = tokenizer.ids2tokens([tokens[0]])[0][1:-1]
    region = tokenizer.ids2tokens([tokens[1]])[0][1:-1]

    result = TranscribeResult(
        text=tokenizer.detokenize(tokens)[0],
        text_nospecial=tokenizer.detokenize(nonspecial_tokens)[0],
        language=lang,
        region=region,
        word_timestamps=word_ts,
    )

    logger.info(f"decode result, language: {result.language}, region: {result.region}, text: {result.text_nospecial}  Timestamp: {word_ts}")
    return result


def cli():
    args = parser_args()

    model = args.model
    if model not in MODELS:
        logging.error(f"Unknown model {model}, Dolphin open source {tuple(MODELS.keys())} model, please config the correct model.")
        return

    model_dir = args.model_dir if args.model_dir else os.path.expanduser(f"~/.cache/dolphin/{model}")
    model_dir = Path(model_dir)

    device = args.device if args.device else detect_device()
    logger.info(f"loading asr model, device: {device}")
    model_instance = load_model(model, model_dir, device)
    logger.info(f"model loaded successfuly, device: {device}")

    # Parse hotwords
    hotwords = _parse_hotwords(args.hotword_str, args.hotword_list_path)

    audio_duration = pydub.AudioSegment.from_file(args.audio).duration_seconds
    transcribe_fn = transcribe_long if audio_duration > SPEECH_LENGTH else transcribe
    transcribe_params = {
        "model": model_instance,
        "audio": args.audio,
        "lang_sym": args.lang_sym,
        "region_sym": args.region_sym,
        "predict_time": args.predict_time,
        "word_timestamp": args.word_timestamp,
        "padding_speech": args.padding_speech,
        "decoding_method": args.decoding_method,
        "beam_size": args.beam_size,
        "hotwords": hotwords,
        "use_deep_biasing": args.use_deep_biasing,
        "use_two_stage_filter": args.use_two_stage_filter,
        "use_prompt_hotword": args.use_prompt_hotword,
        "prompt_filter_threshold": args.prompt_filter_threshold,
    }
    result = transcribe_fn(**transcribe_params)
    _emit_cli_output(result, args.output_format, args.output)


if __name__ == "__main__":
    cli()
