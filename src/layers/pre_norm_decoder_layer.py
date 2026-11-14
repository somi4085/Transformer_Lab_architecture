
import torch
import torch.nn as nn

from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.cross_attention import CrossAttention
from src.layers.feed_forward import FeedForwardNetwork
from src.layers.pre_norm import PreNorm


class PreNormDecoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        # 1. Masked self-attention
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

        # 2. Encoder-decoder cross-attention
        self.cross_attention = CrossAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.pre_norm2 = PreNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # 3. Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.pre_norm3 = PreNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(
        self,
        x,
        encoder_output,
        self_attention_mask=None,
        encoder_mask=None
    ):

        # Step 1: Masked self-attention
        x = self.pre_norm1(
            x,
            lambda normalized_x: self.self_attention(
                normalized_x,
                mask=self_attention_mask,
                causal=True
            )
        )

        # Step 2: Cross-attention
        x = self.pre_norm2(
            x,
            lambda normalized_x: self.cross_attention(
                normalized_x,
                encoder_output,
                encoder_mask=encoder_mask
            )
        )

        # Step 3: Feed-forward network
        x = self.pre_norm3(
            x,
            self.feed_forward
        )

        return x
