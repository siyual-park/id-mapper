from typing import List

import numpy as np
import torch
from PIL.Image import Image
from torch import nn

from id_mapper.model.tokenizer import Tokenizer


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        K = K.transpose(-1, -2)
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K) / np.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            d_k,
            d_v
    ):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.scaled_dot_product_attention = ScaledDotProductAttention(d_k)
        self.output = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask=None):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q)
        q_s = q_s.view(Q.size(0), -1, self.n_heads, self.d_k)
        q_s = q_s.transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]

        k_s = self.W_K(K)
        k_s = k_s.view(K.size(0), -1, self.n_heads, self.d_k)
        k_s = k_s.transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]

        v_s = self.W_V(V)
        v_s = v_s.view(V.size(0), -1, self.n_heads, self.d_v)
        v_s = v_s.transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2)
        context = context.contiguous()
        context = context.view(Q.size(0), -1, self.n_heads * self.d_v)  # context: [batch_size x len_q x n_heads * d_v]

        output = self.output(context)
        output = self.norm(output)

        return output, attn


class Comparator(nn.Module):
    def __init__(
            self,
            image_size: int,
            token_size: int,
            head_size: int
    ):
        super().__init__()

        self.tokenizer = Tokenizer(
            image_size=image_size,
            token_size=token_size
        )
        self.embedding = MultiHeadAttention(
            d_model=token_size,
            n_heads=head_size,
            d_k=token_size // head_size,
            d_v=token_size // head_size
        )
        self.decode = nn.Linear(token_size // head_size * head_size, 1)

    def forward(self, keys: List[Image], queries: List[Image]):
        key_tokens = self.tokenizer(keys)
        query_tokens = self.tokenizer(queries)
        value_tokens = query_tokens.clone()

        kernels = []
        for key_token in key_tokens:
            current_key_tokens = key_token.repeat(len(queries), 1)
            kernel, attention = self.embedding(
                query_tokens,
                current_key_tokens,
                value_tokens
            )
            kernels.append(kernel)

        kernels = torch.stack(kernels)
        kernels = kernels.view(len(keys) * len(queries), -1)

        scores = self.decode(kernels)
        scores = scores.view(len(keys), len(queries))
        scores = torch.sigmoid(scores)

        return scores
