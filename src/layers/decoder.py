
import torch.nn as nn

from src.embeddings.input_embedding import InputEmbeddings
from src.layers.decoder_stack import DecoderStack


class TransformerDecoder(nn.Module):

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

        self.decoder_stack = DecoderStack(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        token_ids,
        encoder_output=None,
        self_attention_mask=None,
        encoder_mask=None,
        past_key_values=None,
        use_cache=False,
        position_offset=0
    ):

        # 1. Incremental token embedding
        embeddings = self.input_embeddings(
            token_ids,
            position_offset=position_offset
        )

        # 2. Process decoder layers
        if use_cache:

            decoder_output, present_key_values = self.decoder_stack(
                x=embeddings,
                encoder_output=encoder_output,
                self_attention_mask=self_attention_mask,
                encoder_mask=encoder_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            return decoder_output, present_key_values

        decoder_output = self.decoder_stack(
            x=embeddings,
            encoder_output=encoder_output,
            self_attention_mask=self_attention_mask,
            encoder_mask=encoder_mask
        )

        return decoder_output


