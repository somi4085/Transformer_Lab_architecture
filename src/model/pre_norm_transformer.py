
import torch
import torch.nn as nn

from src.layers.pre_norm_encoder import PreNormTransformerEncoder
from src.layers.pre_norm_decoder import PreNormTransformerDecoder


class PreNormTransformer(nn.Module):

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

        self.encoder = PreNormTransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.decoder = PreNormTransformerDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

    def forward(
        self,
        source_token_ids,
        target_token_ids,
        source_mask=None,
        target_mask=None
    ):

        encoder_output = self.encoder(
            token_ids=source_token_ids,
            mask=source_mask
        )

        decoder_output = self.decoder(
            token_ids=target_token_ids,
            encoder_output=encoder_output,
            self_attention_mask=target_mask,
            encoder_mask=source_mask
        )

        logits = self.output_projection(decoder_output)

        return logits
