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
            kernel_size: int,
            head_size: int,
            attention_size: int,
            dropout: float,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.kernel_size = kernel_size

        self.attention = MultiHeadAttention(
            d_model=tokenizer.token_size,
            n_heads=head_size,
            d_k=tokenizer.token_size,
            d_v=tokenizer.token_size
        )

        self.feed_forward = FeedForward(
            d_model=tokenizer.token_size,
            dropout=dropout,
            intermediate_size=tokenizer.token_size * 2
        )
        self.normalize = nn.Linear(
            tokenizer.token_size,
            tokenizer.token_size
        )

        self_attentions = []
        for i in range(attention_size):
            self_attentions.append(SelfAttention(
                d_model=kernel_size,
                kernel_size=kernel_size,
                head_size=head_size,
                dropout=dropout
            ))

        self.self_attentions = nn.Sequential(*self_attentions)

        self.logits = nn.Linear(tokenizer.token_size, 1)

    def to(self, device):
        super(Comparator, self).to(device)
        self.tokenizer.to(device)

    def forward(self, keys: List[Image], queries: List[Image]):
        key_tokens = self.tokenizer(keys)
        query_tokens = self.tokenizer(queries)

        kernels = self.embedding(query_tokens, key_tokens)

        key_size, query_size, _ = kernels.size()
        kernels = kernels.view(key_size * query_size, -1, self.kernel_size)

        kernels = self.self_attentions(kernels)
        kernels = kernels.view(key_size * query_size, -1)

        logits = self.logits(kernels)
        logits = logits.view(key_size, query_size)

        return logits

    def embedding(self, queries, keys) -> torch.Tensor:
        keys_size = keys.size(0)
        query_size = queries.size(0)
        queries = queries.view(1, query_size, -1)

        kernels = []
        for key in keys:
            key = key.repeat(query_size, 1)
            key = key.view(1, query_size, -1)

            context, attention = self.attention(
                queries,
                key,
                key
            )
            kernels.append(context)

        kernels = torch.stack(kernels)
        kernels = kernels.view(keys_size * query_size, -1)

        kernels = self.feed_forward(kernels)
        kernels = self.normalize(kernels)
        kernels = kernels.view(keys_size, query_size, -1)

        return kernels
