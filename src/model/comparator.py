import torch
from torch import nn

from src.model.common import CosineDistance
from src.model.tokenizer import Tokenizer


class Comparator(nn.Module):
    def __init__(
            self,
            tokenizer: Tokenizer
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.cosine_distance = CosineDistance()

    def forward(self, x1, x2):
        # x1 (batch1, 3, w, h), x2 (batch2, 3, w, h)
        # x1_tokens (batch1, token_size), x2_tokens (batch1, token_size)
        x1_tokens: torch.Tensor = self.tokenizer.forward(x1)
        x2_tokens: torch.Tensor = self.tokenizer.forward(x2)

        distance = self.cosine_distance(x1_tokens, x2_tokens)

        return 1 - distance
