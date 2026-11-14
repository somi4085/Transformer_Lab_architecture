
import torch.nn as nn

from src.layers.decoder_layer import DecoderLayer


class DecoderStack(nn.Module):

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

        self.layers = nn.ModuleList([
            DecoderLayer(
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
        encoder_output=None,
        self_attention_mask=None,
        encoder_mask=None,
        past_key_values=None,
        use_cache=False
    ):

        present_key_values = []

        for layer_index, decoder_layer in enumerate(self.layers):

            # Get cache belonging to the current decoder layer
            if past_key_values is not None:
                layer_cache = past_key_values[layer_index]

                past_self_attention = layer_cache["self_attention"]
                past_cross_attention = layer_cache["cross_attention"]

            else:
                past_self_attention = None
                past_cross_attention = None

            if use_cache:

                x, present_cache = decoder_layer(
                    x=x,
                    encoder_output=encoder_output,
                    self_attention_mask=self_attention_mask,
                    encoder_mask=encoder_mask,
                    past_self_attention=past_self_attention,
                    past_cross_attention=past_cross_attention,
                    use_cache=True
                )

                present_key_values.append(present_cache)

            else:

                x = decoder_layer(
                    x=x,
                    encoder_output=encoder_output,
                    self_attention_mask=self_attention_mask,
                    encoder_mask=encoder_mask
                )

        if use_cache:
            return x, present_key_values

        return x
