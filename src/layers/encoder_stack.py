
import torch
import torch.nn as nn

from src.layers.encoder_layer import EncoderLayer


class EncoderStack(nn.Module):

    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.num_layers = num_layers

        # Create multiple independent Encoder Layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):

        # Pass output of each layer to the next layer
        for encoder_layer in self.layers:
            x = encoder_layer(
                x,
                mask=mask,
                causal=False
            )

        return x


