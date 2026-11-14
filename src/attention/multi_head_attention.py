import torch.nn as nn
import torch
from src.attention.scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads,dropout, bias):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads


        # validate head dimension 
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.head_dim = embedding_dim // num_heads

        # query projection
        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias
        )

        # key projection
        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=bias
        )

        # value projection

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
        x,
        mask=None,
        causal=False,
        past_key_value=None,
        use_cache=False
    ):
        # 1. Project input into Query, Key and Value
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        batch_size, sequence_length, _ = x.shape

        # 2. Split embedding dimension into attention heads
        query = query.view(
            batch_size, sequence_length,
            self.num_heads, self.head_dim
        )

        key = key.view(
            batch_size, sequence_length,
            self.num_heads, self.head_dim
        )

        value = value.view(
            batch_size, sequence_length,
            self.num_heads, self.head_dim
        )

        # 3. Move heads before sequence dimension
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 4. Reuse previous Key and Value
        if past_key_value is not None:
            past_key, past_value = past_key_value

            key = torch.cat(
                [past_key, key],
                dim=2
            )

            value = torch.cat(
                [past_value, value],
                dim=2
            )

        # 5. Prepare padding mask
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)

        # 6. Apply attention
        # Cached incremental decoding has no future keys available.
        attention_causal = causal and past_key_value is None

        attention_output = self.attention(
            query,
            key,
            value,
            mask=mask,
            causal=attention_causal
        )

        # 7. Combine attention heads
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # 8. Final output projection
        attention_output = self.output_projection(attention_output)

        # 9. Return updated cache when requested
        if use_cache:
            return attention_output, (key, value)

        return attention_output

