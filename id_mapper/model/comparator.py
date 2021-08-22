from typing import List

import torch
from PIL.Image import Image
from torch import nn

from id_mapper.model.tokenizer import Tokenizer


class Comparator(nn.Module):
    def __init__(
            self,
            tokenizer: Tokenizer
    ):
        super().__init__()

        self.tokenizer = tokenizer

    def to(self, device):
        super(Comparator, self).to(device)
        self.tokenizer.to(device)

    def forward(self, keys: List[Image], queries: List[Image]):
        key_tokens = self.tokenizer(keys)

        query_tokens = self.tokenizer(queries)
        query_tokens = query_tokens.transpose(0, 1)

        logits = torch.matmul(key_tokens, query_tokens)

        return logits
