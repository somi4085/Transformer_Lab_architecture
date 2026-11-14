
import torch
import torch.nn as nn

from src.embeddings.input_embedding import InputEmbeddings
from src.layers.pre_norm_decoder_stack import PreNormDecoderStack


class PreNormTransformerDecoder(nn.Module):

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

        self.decoder_stack = PreNormDecoderStack(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        token_ids,
        encoder_output,
        self_attention_mask=None,
        encoder_mask=None
    ):

        embeddings = self.input_embeddings(token_ids)

        decoder_output = self.decoder_stack(
            x=embeddings,
            encoder_output=encoder_output,
            self_attention_mask=self_attention_mask,
            encoder_mask=encoder_mask
        )

        return decoder_output
