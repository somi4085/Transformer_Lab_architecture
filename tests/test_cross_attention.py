
import pytest
import torch

from src.attention.cross_attention import CrossAttention


def create_cross_attention(
    embedding_dim=256,
    num_heads=8,
    dropout=0.0
):
    return CrossAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=True
    )


def create_inputs(
    batch_size=2,
    decoder_length=5,
    encoder_length=7,
    embedding_dim=256
):
    decoder_output = torch.randn(
        batch_size,
        decoder_length,
        embedding_dim
    )

    encoder_output = torch.randn(
        batch_size,
        encoder_length,
        embedding_dim
    )

    return decoder_output, encoder_output


# 1. Output shape
def test_output_shape():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    output = attention(decoder_output, encoder_output)

    assert output.shape == (2, 5, 256)


# 2. Query projection comes from decoder
def test_query_projection():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    query = attention.query_projection(decoder_output)

    assert query.shape == (2, 5, 256)


# 3. Key projection comes from encoder
def test_key_projection():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    key = attention.key_projection(encoder_output)

    assert key.shape == (2, 7, 256)


# 4. Value projection comes from encoder
def test_value_projection():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    value = attention.value_projection(encoder_output)

    assert value.shape == (2, 7, 256)


# 5. Head splitting
def test_head_splitting():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    batch_size, decoder_length, _ = decoder_output.shape
    _, encoder_length, _ = encoder_output.shape

    query = attention.query_projection(decoder_output)
    key = attention.key_projection(encoder_output)
    value = attention.value_projection(encoder_output)

    query = query.view(
        batch_size,
        decoder_length,
        attention.num_heads,
        attention.head_dim
    )

    key = key.view(
        batch_size,
        encoder_length,
        attention.num_heads,
        attention.head_dim
    )

    value = value.view(
        batch_size,
        encoder_length,
        attention.num_heads,
        attention.head_dim
    )

    assert query.shape == (2, 5, 8, 32)
    assert key.shape == (2, 7, 8, 32)
    assert value.shape == (2, 7, 8, 32)


# 6. Attention score dimensions
def test_attention_score_dimensions():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    query = attention.query_projection(decoder_output)
    key = attention.key_projection(encoder_output)

    query = query.view(2, 5, 8, 32).transpose(1, 2)
    key = key.view(2, 7, 8, 32).transpose(1, 2)

    scores = torch.matmul(
        query,
        key.transpose(-2, -1)
    )

    assert scores.shape == (2, 8, 5, 7)


# 7. Encoder padding mask
def test_encoder_padding_mask():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    encoder_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0]
    ])

    output = attention(
        decoder_output,
        encoder_output,
        encoder_mask=encoder_mask
    )

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


# 8. Gradient flow
def test_gradient_flow():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs()

    decoder_output.requires_grad_(True)
    encoder_output.requires_grad_(True)

    output = attention(decoder_output, encoder_output)

    loss = output.sum()
    loss.backward()

    assert decoder_output.grad is not None
    assert encoder_output.grad is not None

    assert torch.count_nonzero(decoder_output.grad).item() > 0
    assert torch.count_nonzero(encoder_output.grad).item() > 0

    projection_layers = [
        attention.query_projection,
        attention.key_projection,
        attention.value_projection,
        attention.output_projection
    ]

    for layer in projection_layers:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad).item() > 0


# 9. Dropout behavior
def test_dropout_behavior():
    attention = create_cross_attention(dropout=0.5)

    decoder_output, encoder_output = create_inputs()

    attention.train()

    train_output_1 = attention(
        decoder_output,
        encoder_output
    )

    train_output_2 = attention(
        decoder_output,
        encoder_output
    )

    assert not torch.equal(train_output_1, train_output_2)

    attention.eval()

    eval_output_1 = attention(
        decoder_output,
        encoder_output
    )

    eval_output_2 = attention(
        decoder_output,
        encoder_output
    )

    assert torch.equal(eval_output_1, eval_output_2)


# 10. Different sequence lengths
def test_different_sequence_lengths():
    attention = create_cross_attention()

    decoder_output, encoder_output = create_inputs(
        batch_size=3,
        decoder_length=4,
        encoder_length=9
    )

    output = attention(decoder_output, encoder_output)

    assert output.shape == (3, 4, 256)


# 11. Invalid head configuration
def test_invalid_head_configuration():
    with pytest.raises(ValueError):
        create_cross_attention(
            embedding_dim=250,
            num_heads=8
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
