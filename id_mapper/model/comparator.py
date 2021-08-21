from typing import List

import torch
from PIL.Image import Image
from torch import nn

from id_mapper.model.attention import MultiHeadAttention, FeedForward, SelfAttention
from id_mapper.model.tokenizer import Tokenizer


class Comparator(nn.Module):
    def __init__(
            self,
            tokenizer: Tokenizer,
            head_size: int,
            intermediate_size: int,
            attention_size: int,
            dropout: float,
    ):
        super().__init__()

        kernel_size = tokenizer.token_size // head_size

        self.tokenizer = tokenizer

        self.attention = MultiHeadAttention(
            d_model=kernel_size * head_size,
            n_heads=head_size,
            d_k=kernel_size,
            d_v=kernel_size
        )
        self.feed_forward = FeedForward(
            kernel_size=kernel_size * head_size,
            dropout=dropout,
            intermediate_size=intermediate_size
        )

        attentions = []
        for i in range(attention_size):
            attentions.append(SelfAttention(
                kernel_size=kernel_size,
                head_size=head_size,
                dropout=dropout,
                intermediate_size=intermediate_size
            ))

        self.attentions = nn.Sequential(*attentions)

        self.logits = nn.Linear(kernel_size * head_size, 1)

    def to(self, device):
        super(Comparator, self).to(device)
        self.tokenizer.to(device)

    def forward(self, keys: List[Image], queries: List[Image]):
        query_tokens = self.tokenizer(queries)
        key_tokens = self.tokenizer(keys)

        kernels = self.embedding(query_tokens, key_tokens)

        key_size, query_size, _ = kernels.size()
        kernels = kernels.view(key_size * query_size, -1)

        kernels = self.attentions(kernels)
        kernels = kernels.view(len(keys) * len(queries), -1)

        logits = self.logits(kernels)
        logits = logits.view(len(keys), len(queries))

        return logits

    def embedding(self, queries, keys) -> torch.Tensor:
        kernels = []
        for key_token in keys:
            current_key_tokens = key_token.repeat(len(queries), 1)
            kernel, attention = self.attention(
                queries,
                current_key_tokens,
                current_key_tokens
            )
            kernels.append(kernel)

        kernels = torch.stack(kernels)
        kernels = kernels.view(len(keys) * len(queries), -1)

        kernels = self.feed_forward(kernels)
        kernels = kernels.view(len(keys), len(queries), -1)

        return kernels
