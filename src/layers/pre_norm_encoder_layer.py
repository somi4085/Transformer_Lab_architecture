
import torch
import torch.nn as nn

from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import FeedForwardNetwork
from src.layers.pre_norm import PreNorm


class PreNormEncoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True
        )

        self.pre_norm1 = PreNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.pre_norm2 = PreNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, x, mask=None):

        x = self.pre_norm1(
            x,
            lambda normalized_x: self.self_attention(
                normalized_x,
                mask=mask,
                causal=False
            )
        )

        x = self.pre_norm2(
            x,
            self.feed_forward
        )

        return x
