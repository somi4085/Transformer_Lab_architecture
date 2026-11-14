import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(
            normalized_shape = embedding_dim
        )

    def forward(self,x, sublayer_output):

        # apply dropout to sublayer
        sublayer_output = self.dropout(sublayer_output)

        # residual output
        residual_output = x + sublayer_output

        # layer normalize
        output = self.layer_norm(residual_output)

        return output
        
