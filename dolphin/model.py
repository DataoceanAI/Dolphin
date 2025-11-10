# encoding: utf8

import os
import time
import yaml
import logging
import dataclasses
import argparse
import functools
from pathlib import Path
from os.path import join, dirname, abspath
from typing import Union, Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from espnet2.tasks.lm import LMTask
from espnet2.tasks.s2t import S2TTask
from espnet2.torch_utils.device_funcs import to_device
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.bin.s2t_inference import Speech2Text, ListOfHypothesis
from dolphin.scorefilter import DolphinScoreFilter
from dolphin.beam_search import SimpleBeamSearch
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

from .constants import (SPEECH_LENGTH, SAMPLE_RATE, FIRST_TIME_SYMBOL, LAST_TIME_SYMBOL, NOTIME_SYMBOL,
                        FIRST_LANG_SYMBOL, LAST_LANG_SYMBOL, FIRST_REGION_SYMBOL, LAST_REGION_SYMBOL)


logger = logging.getLogger("dolphin")


@dataclasses.dataclass
class TranscribeResult:
    text: str
    text_nospecial: str
    language: str
    region: str
    rtf: float


@dataclasses.dataclass
class TranscribeSegmentResult:
    text: str
    text_nospecial: str
    language: str
    region: str
    rtf: float
    start: float
    end: float


class DolphinSpeech2Text(Speech2Text):

    @typechecked
    def __init__(
        self,
        s2t_train_config: Union[Path, str, Dict] = None,
        s2t_model_file: Union[Path, str, None] = None,
        lm_train_config: Union[Path, str, None] = None,
        lm_file: Union[Path, str, None] = None,
        ngram_scorer: str = "full",
        ngram_file: Union[Path, str, None] = None,
        device: str = "cpu",
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        batch_size: int = 1,
        dtype: str = "float32",
        beam_size: int = 5,
        lm_weight: float = 0.0,
        ngram_weight: float = 0.0,
        penalty: float = 0.0,
        nbest: int = 1,
        normalize_length: bool = False,
        quantize_s2t_model: bool = False,
        quantize_lm: bool = False,
        quantize_modules: List[str] = ["Linear"],
        quantize_dtype: str = "qint8",
        task_sym: str = "<asr>",
        predict_time: bool = True,
        partial_ar: bool = False,
        **kwargs,
    ):

        qconfig_spec = set([getattr(torch.nn, q) for q in quantize_modules])
        quantize_dtype: torch.dtype = getattr(torch, quantize_dtype)

        # 1. Build S2T model
        s2t_model, s2t_train_args = self.build_model_from_file(s2t_train_config, s2t_model_file, device)
        s2t_model.to(dtype=getattr(torch, dtype)).eval()

        if quantize_s2t_model:
            logger.info("Use quantized s2t model for decoding.")
            s2t_model = torch.quantization.quantize_dynamic(s2t_model, qconfig_spec=qconfig_spec, dtype=quantize_dtype)

        decoder = s2t_model.decoder
        token_list = s2t_model.token_list
        scorers = dict(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
            scorefilter=DolphinScoreFilter(
                notimestamps=token_list.index(NOTIME_SYMBOL),
                first_time=token_list.index(FIRST_TIME_SYMBOL),
                last_time=token_list.index(LAST_TIME_SYMBOL),
                sos=s2t_model.sos,
                eos=s2t_model.eos,
                vocab_size=len(token_list),
            ),
        )

        # 2. Build language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(lm_train_config, lm_file, device)

            if quantize_lm:
                logger.info("Use quantized lm for decoding.")
                lm = torch.quantization.quantize_dynamic(lm, qconfig_spec=qconfig_spec, dtype=quantize_dtype)

            scorers["lm"] = lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
            scorers["ngram"] = ngram

        # 4. Build BeamSearch object
        weights = dict(
            decoder=1.0,
            lm=lm_weight,
            ngram=ngram_weight,
            length_bonus=penalty,
            scorefilter=1.0,
        )
            
        beam_search = SimpleBeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=s2t_model.sos,
            eos=s2t_model.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key="full",
            normalize_length=normalize_length,
        )

        if batch_size == 1:
            non_batch = [
                k
                for k, v in beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                beam_search.__class__ = SimpleBeamSearch
                logger.info("BatchBeamSearch implementation is selected.")
            else:
                logger.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )

            beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logger.info(f"Decoding device={device}, dtype={dtype}")

        bpemodel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/bpe.model")
        tokenizer = build_tokenizer(token_type="bpe", bpemodel=bpemodel)
        converter = TokenIDConverter(token_list=token_list)
        logger.info(f"Text tokenizer: {tokenizer}")

        self.s2t_model = s2t_model
        self.s2t_train_args = s2t_train_args
        self.preprocessor_conf = s2t_train_args.preprocessor_conf
        self.converter = converter
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        self.task_sym = task_sym
        self.predict_time = predict_time

        self.partial_ar = partial_ar

    @staticmethod
    @typechecked
    def build_model_from_file(
        config: Optional[Union[Path, str, Dict]] = None,
        model_file: Optional[Union[Path, str]] = None,
        device: str = "cpu",
    ) -> Tuple[AbsESPnetModel, argparse.Namespace]:
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        if isinstance(config, dict):
            args = argparse.Namespace(**config)
        else:
            with open(config, "r", encoding="utf-8") as f:
                args = yaml.safe_load(f)
            args = argparse.Namespace(**args)
        args.normalize_conf["stats_file"] = os.path.join(dirname(abspath(__file__)), "assets/feats_stats.npz")

        model = S2TTask.build_model(args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}")

        if device == "mps":
            model.to(device, dtype=torch.float32)
        else:
            model.to(device)

        if model_file is not None:
            if device == "cuda":
                device = f"cuda:{torch.cuda.current_device()}"

            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict, strict=False)

        return model, args

    def install_kv_cache(self, module: nn.Module):
        def kv_linear_forward_fn(ins: nn.Linear, input: torch.Tensor) -> torch.Tensor:
            if hasattr(ins, "kv_cache") and ins.kv_cache.size() == input.size():
                return ins.kv_cache

            output = F.linear(input, ins.weight, ins.bias)
            if hasattr(ins, "kv_cache"):
                del ins.kv_cache
            ins.register_buffer("kv_cache", output)
            return output

        def rewrite_cross_attn_forward(module_name: str, module: nn.Module):
            if module_name == "src_attn":
                module.linear_k.forward = functools.partial(kv_linear_forward_fn, module.linear_k)
                module.linear_v.forward = functools.partial(kv_linear_forward_fn, module.linear_v)

        def module_apply_with_name(module: nn.Module, fn: callable, name: str = None):
            fn(name, module)
            for name, m in module.named_children():
                module_apply_with_name(m, fn, name)

        module_apply_with_name(module, rewrite_cross_attn_forward)

    def clean_cache(self, module: nn.Module):
        def clean_kv_cache(module: nn.Module):
            if hasattr(module, "kv_cache"):
                del module.kv_cache

        module.apply(clean_kv_cache)

    def detect_language(self, speech: torch.Tensor, lang_sym: str = None, with_enc_output: bool = False, **kwargs) -> Tuple[int, int]:
        """
        Detect language and region.

        Args:
            speech: waveform

        Returns:
            tuple(lang_id, region_id)
        """
        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # Only support single-channel speech
        if speech.dim() > 1:
            assert (speech.dim() == 2 and speech.size(1) == 1), f"speech of size {speech.size()} is not supported"
            speech = speech.squeeze(1)  # (nsamples, 1) --> (nsamples,)

        speech_length = int(SAMPLE_RATE * SPEECH_LENGTH)
        # Pad or trim speech to the fixed length
        if speech.size(-1) >= speech_length:
            speech = speech[:speech_length]

        # Batchify input
        # speech: (nsamples,) -> (1, nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        batch = to_device(batch, device=self.device)
        enc, enc_olens = self.s2t_model.encode(**batch)
        assert enc.size(0) == 1

        lang_first_id = self.converter.token2id[FIRST_LANG_SYMBOL]
        lang_last_id = self.converter.token2id[LAST_LANG_SYMBOL]
        region_first_id = self.converter.token2id[FIRST_REGION_SYMBOL]
        region_last_id = self.converter.token2id[LAST_REGION_SYMBOL]

        decoder = self.s2t_model.decoder
        # detect language
        lang_id = None
        if lang_sym is None:
            ys = torch.tensor([self.s2t_model.sos], dtype=torch.long, device=self.device).unsqueeze(0)
            logp, _ = decoder.batch_score(ys, [None], enc)
            mask = torch.ones(logp.size(1), dtype=torch.bool)
            mask[lang_first_id:lang_last_id+1] = False
            logp[0, mask] = -np.inf
            lang_id = logp.argmax(dim=-1).tolist()[0]
        else:
            logger.debug(f"lang symbol is given: {lang_sym}")
            lang_id = self.converter.token2id[f"<{lang_sym}>"]

        # detect region
        ys = torch.tensor([self.s2t_model.sos, lang_id], dtype=torch.long, device=self.device).unsqueeze(0)
        logp, _ = decoder.batch_score(ys, [None], enc)
        mask = torch.ones(logp.size(1), dtype=torch.bool)
        mask[region_first_id:region_last_id+1] = False
        logp[0, mask] = -np.inf
        region_id = logp.argmax(dim=-1).tolist()[0]

        lang_symbol = self.converter.ids2tokens([lang_id])[0]
        region_symbol = self.converter.ids2tokens([region_id])[0]

        logger.debug(f"detect language: {lang_symbol}, region: {region_symbol}")

        if with_enc_output:
            return lang_id, region_id, enc
        else:
            return lang_id, region_id, None

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        lang_sym: Optional[str] = None,
        region_sym: Optional[str] = None,
        predict_time: Optional[bool] = None,
        padding_speech: bool = False,
    ) -> TranscribeResult:
        """Inference for a single utterance.

        The input speech will be padded or trimmed to the fixed length,
        which is consistent with training.

        Args:
            speech: input speech of shape (nsamples,) or (nsamples, nchannels=1)
            lang_sym: language code symbol (e.g. zh)
            region_sym: region symbol (e.g. CN)
            predict_time: whether to predict timestamps
            padding_speech: whether to padding speech to 30 seconds, default is true

        Returns:
            TranscribeResult

        """

        predict_time = predict_time if predict_time is not None else self.predict_time

        start_tm = time.monotonic()
        enc = None
        if all([lang_sym, region_sym]):
            lang_id = self.converter.token2id[f"<{lang_sym}>"]
            region_id = self.converter.token2id[f"<{region_sym}>"]
        else:

            lang_id, region_id, enc = self.detect_language(speech, lang_sym, with_enc_output=True)

        task_id = self.converter.token2id["<asr>"]
        notime_id = self.converter.token2id[NOTIME_SYMBOL]

        # Prepare hyp_primer
        hyp_primer = [self.s2t_model.sos, lang_id, region_id, task_id]
        if not predict_time:
            hyp_primer.append(notime_id)

        self.beam_search.set_hyp_primer(hyp_primer)

        # Preapre speech
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # Only support single-channel speech
        if speech.dim() > 1:
            assert (speech.dim() == 2 and speech.size(1) == 1), f"speech of size {speech.size()} is not supported"
            speech = speech.squeeze(1)  # (nsamples, 1) --> (nsamples,)

        nsamples = speech.size(-1)
        speech_length = int(SAMPLE_RATE * SPEECH_LENGTH)
        # Pad or trim speech to the fixed length
        if speech.size(-1) >= speech_length:
            speech = speech[:speech_length]
        else:
            speech = F.pad(speech, (0, speech_length - speech.size(-1))) if padding_speech else speech

        # Batchify input
        # speech: (nsamples,) -> (1, nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        logger.debug("speech length: " + str(speech.size(1)))

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        if enc is None:
            enc, enc_olens = self.s2t_model.encode(**batch)
        if isinstance(enc, tuple):
            enc, _ = enc
        encode_tm = time.monotonic()
        logger.debug(f"encode time: {round(encode_tm - start_tm, 2)} seconds")

        assert len(enc) == 1, len(enc)

        try:
            self.install_kv_cache(self.s2t_model.decoder)

            # c. Pass the encoder result to the beam search
            results = self._decode_single_sample(enc[0])
            text, _, _, text_nospecial, _ = results[0]
            decode_tm = time.monotonic()
            logger.debug(f"decode time: {round(decode_tm - encode_tm, 2)} senconds")
            rtf = round((decode_tm - start_tm) / (nsamples / SAMPLE_RATE), 2)

            lang, region = self.converter.ids2tokens([lang_id, region_id])
            ret = TranscribeResult(text, text_nospecial, lang[1:-1], region[1:-1], rtf=rtf)
            return ret
        finally:
            self.clean_cache(self.s2t_model.decoder)
