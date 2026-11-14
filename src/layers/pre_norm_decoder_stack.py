
import torch
import torch.nn as nn

from src.layers.pre_norm_decoder_layer import PreNormDecoderLayer


class PreNormDecoderStack(nn.Module):

    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            PreNormDecoderLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        encoder_output,
        self_attention_mask=None,
        encoder_mask=None
    ):

        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                self_attention_mask=self_attention_mask,
                encoder_mask=encoder_mask
            )

        return x
