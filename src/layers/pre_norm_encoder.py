
import torch
import torch.nn as nn

from src.embeddings.input_embedding import InputEmbeddings
from src.layers.pre_norm_encoder_stack import PreNormEncoderStack


class PreNormTransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_sequence_length,
        num_layers,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.input_embeddings = InputEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )

        self.encoder_stack = PreNormEncoderStack(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, token_ids, mask=None):

        embeddings = self.input_embeddings(token_ids)

        encoder_output = self.encoder_stack(
            embeddings,
            mask=mask
        )

        return encoder_output
