import numpy as np
import torch
from torch import nn


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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float, intermediate_size: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, d_model)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tensor = self.linear1(inputs)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)

        tensor = self.linear2(tensor)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)

        tensor = self.norm(tensor)

        return tensor


class SelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            kernel_size: int,
            head_size: int,
            dropout: float
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=head_size,
            d_k=kernel_size,
            d_v=kernel_size
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            dropout=dropout,
            intermediate_size=d_model * 2
        )

    def forward(self, inputs: torch.Tensor, attention=None) -> torch.Tensor:
        context, attention = self.attention(
            inputs,
            inputs,
            inputs,
            attention
        )

        context = self.feed_forward(context)
        return context, attention

class SelfAttentions(nn.Module):
    def __init__(
            self,
            d_model: int,
            kernel_size: int,
            head_size: int,
            dropout: float,
            deep: int
    ):
        super().__init__()

        self_attentions = []
        for i in range(deep):
            self_attentions.append(SelfAttention(
                d_model=kernel_size,
                kernel_size=kernel_size,
                head_size=head_size,
                dropout=dropout
            ))

        self.self_attentions = nn.ModuleList(self_attentions)

    def forward(self, inputs: torch.Tensor, attention=None) -> torch.Tensor:
        context = inputs
        for self_attention in self.attentions:
            context, _ = self_attention(context, attention)

        return context