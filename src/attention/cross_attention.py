
import torch
import torch.nn as nn

from src.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention
)


class CrossAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1,
        bias=True
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias
        )

        self.attention = ScaledDotProductAttention(
            dropout=dropout
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias
        )

    def forward(
        self,
        decoder_output,
        encoder_output=None,
        encoder_mask=None,
        past_key_value=None,
        use_cache=False
    ):

        batch_size, decoder_seq_length, _ = decoder_output.shape

        # 1. Calculate Query from decoder output
        query = self.query_projection(decoder_output)

        query = query.view(
            batch_size,
            decoder_seq_length,
            self.num_heads,
            self.head_dim
        )

        query = query.transpose(1, 2)

        # 2. Reuse cached Encoder Key and Value
        if past_key_value is not None:

            key, value = past_key_value

        else:

            if encoder_output is None:
                raise ValueError(
                    "encoder_output is required when cache is unavailable"
                )

            encoder_seq_length = encoder_output.size(1)

            # Calculate Key and Value only once
            key = self.key_projection(encoder_output)
            value = self.value_projection(encoder_output)

            key = key.view(
                batch_size,
                encoder_seq_length,
                self.num_heads,
                self.head_dim
            )

            value = value.view(
                batch_size,
                encoder_seq_length,
                self.num_heads,
                self.head_dim
            )

            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        # 3. Prepare Encoder padding mask
        if encoder_mask is not None and encoder_mask.dim() == 2:
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(1)

        # 4. Calculate Cross-Attention
        attention_output = self.attention(
            query,
            key,
            value,
            mask=encoder_mask,
            causal=False
        )

        # 5. Combine attention heads
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous()

        attention_output = attention_output.view(
            batch_size,
            decoder_seq_length,
            self.embedding_dim
        )

        # 6. Final projection
        output = self.output_projection(attention_output)

        # 7. Return updated cache
        if use_cache:
            return output, (key, value)

        return output




        