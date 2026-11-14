
import torch
import torch.nn as nn

from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.add_norm import AddNorm
from src.layers.feed_forward import FeedForwardNetwork


class EncoderLayer(nn.Module):

    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True
        )

        # First Add & Norm
        self.add_norm1 = AddNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # Feed Forward Network
        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Second Add & Norm
        self.add_norm2 = AddNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, x, mask=None, causal=False):

        # Multi-head self-attention
        attention_output = self.attention(
            x,
            mask=mask,
            causal=causal
        )

        # First residual connection + LayerNorm
        output = self.add_norm1(
            x,
            attention_output
        )

        # Feed Forward Network
        feed_forward_output = self.feed_forward(output)

        # Second residual connection + LayerNorm
        output = self.add_norm2(
            output,
            feed_forward_output
        )

        return output

