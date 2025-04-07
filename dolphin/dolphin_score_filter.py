import torch
import numpy as np

from typing import Any, Tuple
from espnet2.bin.s2t_inference import ScoreFilter

import logging

LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class DolphineScoreFilter(ScoreFilter):
    """
    Dolphin Score Filter
    for once-only auto-regressive decoding
    """
    def __init__(
        self,
        notimestamps: int,
        first_time: int,
        last_time: int,
        sos: int,
        eos: int,
        vocab_size: int,
        predict_time: bool = True,
    ):
        super().__init__(
            notimestamps,
            first_time,
            last_time,
            sos,
            eos,
            vocab_size,
        )

        self.notimestamps = notimestamps
        self.first_time = first_time
        self.last_time = last_time
        self.sos = sos
        self.eos = eos
        self.vocab_size = vocab_size
        self.predict_time = predict_time

        # dummy param used to obtain the current dtype and device
        self.param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    
    
    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """
        score = torch.zeros(
            self.vocab_size, dtype=self.param.dtype, device=self.param.device
        )
        if len(y) <= 3:
            return score, None
        if not self.predict_time:
            # Suppress timestamp tokens if we don't predict time
            score[self.first_time : self.last_time + 1] = -np.inf
        else:
            if y[-4] == self.sos:
                # The first token must be a timestamp if we predict time
                score[: self.first_time] = -np.inf
                score[self.last_time + 1 :] = -np.inf
            else:
                prev_times = y[torch.logical_and(y >= self.first_time, y <= self.last_time)]
                if len(prev_times) % 2 == 1:
                    # there are an odd number of timestamps, so the sentence is incomplete
                    score[self.eos] = -np.inf
                    # timestamps are monotonic
                    score[self.first_time : prev_times[-1] + 1] = -np.inf
                else:
                    # there are an even number of timestamps (all are paired)
                    if y[-1] >= self.first_time and y[-1] <= self.last_time:
                        # the next tokon should be a timestamp or eos
                        score[: y[-1]] = -np.inf
                        score[self.last_time + 1 :] = -np.inf
                        score[self.eos] = 0.0
                    else:
                        # this is an illegal hyp
                        score[:] = -np.inf
        
        return score, None