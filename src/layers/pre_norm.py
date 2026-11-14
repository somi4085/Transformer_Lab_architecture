
import torch
import torch.nn as nn


class PreNorm(nn.Module):

    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        normalized_input = self.layer_norm(x)

        sublayer_output = sublayer(normalized_input)

        sublayer_output = self.dropout(sublayer_output)

        output = x + sublayer_output

        return output
