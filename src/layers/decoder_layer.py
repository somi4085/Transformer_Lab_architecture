
import torch.nn as nn

from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.add_norm import AddNorm
from src.attention.cross_attention import CrossAttention
from src.layers.feed_forward import FeedForwardNetwork


class DecoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True
        )

        self.add_norm1 = AddNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.cross_attention = CrossAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True
        )

        self.add_norm2 = AddNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.add_norm3 = AddNorm(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(
        self,
        x,
        encoder_output=None,
        self_attention_mask=None,
        encoder_mask=None,
        past_self_attention=None,
        past_cross_attention=None,
        use_cache=False
    ):

        # 1. Decoder Self-Attention
        if use_cache:
            self_attention_output, present_self_attention = (
                self.self_attention(
                    x,
                    mask=self_attention_mask,
                    causal=True,
                    past_key_value=past_self_attention,
                    use_cache=True
                )
            )
        else:
            self_attention_output = self.self_attention(
                x,
                mask=self_attention_mask,
                causal=True
            )
            present_self_attention = None

        output = self.add_norm1(x, self_attention_output)

        # 2. Encoder-Decoder Cross-Attention
        if use_cache:
            cross_attention_output, present_cross_attention = (
                self.cross_attention(
                    decoder_output=output,
                    encoder_output=encoder_output,
                    encoder_mask=encoder_mask,
                    past_key_value=past_cross_attention,
                    use_cache=True
                )
            )
        else:
            cross_attention_output = self.cross_attention(
                decoder_output=output,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask
            )
            present_cross_attention = None

        output = self.add_norm2(output, cross_attention_output)

        # 3. Feed Forward Network
        feed_forward_output = self.feed_forward(output)

        output = self.add_norm3(output, feed_forward_output)

        if use_cache:
            present_cache = {
                "self_attention": present_self_attention,
                "cross_attention": present_cross_attention
            }

            return output, present_cache

        return output
