
import pytest
import torch

from src.layers.decoder_layer import DecoderLayer
from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.cross_attention import CrossAttention
from src.layers.add_norm import AddNorm
from src.layers.feed_forward import FeedForwardNetwork


def create_decoder_layer(
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
    dropout=0.0
):
    return DecoderLayer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout
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
    model = create_decoder_layer()

    decoder_output, encoder_output = create_inputs()

    output = model(decoder_output, encoder_output)

    assert output.shape == (2, 5, 256)


# 2. Self-attention module
def test_self_attention_module():
    model = create_decoder_layer()

    assert isinstance(model.self_attention, MultiHeadAttention)


# 3. Causal masking
def test_causal_masking():
    model = create_decoder_layer()
    model.eval()

    decoder_output, encoder_output = create_inputs()

    captured = {}

    def capture_attention_input(module, args, kwargs):
        captured["causal"] = kwargs.get("causal", False)

    hook = model.self_attention.register_forward_pre_hook(
        capture_attention_input,
        with_kwargs=True
    )

    try:
        model(decoder_output, encoder_output)
    finally:
        hook.remove()

    assert captured["causal"] is True


# 4. Cross-attention module
def test_cross_attention_module():
    model = create_decoder_layer()

    assert isinstance(model.cross_attention, CrossAttention)


# 5. Three independent AddNorm modules
def test_add_norm_modules():
    model = create_decoder_layer()

    add_norm_modules = [
        module for module in model.modules()
        if isinstance(module, AddNorm)
    ]

    assert len(add_norm_modules) == 3

    assert add_norm_modules[0] is not add_norm_modules[1]
    assert add_norm_modules[1] is not add_norm_modules[2]
    assert add_norm_modules[0] is not add_norm_modules[2]


# 6. FeedForwardNetwork module
def test_feed_forward_module():
    model = create_decoder_layer()

    feed_forward_modules = [
        module for module in model.modules()
        if isinstance(module, FeedForwardNetwork)
    ]

    assert len(feed_forward_modules) == 1


# 7. Decoder self-attention padding mask
def test_self_attention_mask():
    model = create_decoder_layer()

    decoder_output, encoder_output = create_inputs()

    self_attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    output = model(
        decoder_output,
        encoder_output,
        self_attention_mask=self_attention_mask
    )

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


# 8. Encoder padding mask
def test_encoder_mask():
    model = create_decoder_layer()

    decoder_output, encoder_output = create_inputs()

    encoder_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0]
    ])

    output = model(
        decoder_output,
        encoder_output,
        encoder_mask=encoder_mask
    )

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


# 9. Gradient flow
def test_gradient_flow():
    model = create_decoder_layer()

    decoder_output, encoder_output = create_inputs()

    decoder_output.requires_grad_(True)
    encoder_output.requires_grad_(True)

    output = model(decoder_output, encoder_output)

    loss = output.sum()
    loss.backward()

    assert decoder_output.grad is not None
    assert encoder_output.grad is not None

    assert torch.count_nonzero(decoder_output.grad).item() > 0
    assert torch.count_nonzero(encoder_output.grad).item() > 0

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient for {name}"
        )


# 10. Different batch sizes
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_decoder_layer()

    decoder_output, encoder_output = create_inputs(
        batch_size=batch_size
    )

    output = model(decoder_output, encoder_output)

    assert output.shape == (batch_size, 5, 256)


# 11. Different sequence lengths
@pytest.mark.parametrize(
    "decoder_length, encoder_length",
    [
        (1, 3),
        (5, 7),
        (10, 4),
        (3, 10)
    ]
)
def test_different_sequence_lengths(
    decoder_length,
    encoder_length
):
    model = create_decoder_layer()

    decoder_output, encoder_output = create_inputs(
        decoder_length=decoder_length,
        encoder_length=encoder_length
    )

    output = model(decoder_output, encoder_output)

    assert output.shape == (2, decoder_length, 256)


# 12. Dropout behavior
def test_dropout_behavior():
    model = create_decoder_layer(dropout=0.5)

    decoder_output, encoder_output = create_inputs()

    model.train()

    train_output_1 = model(decoder_output, encoder_output)
    train_output_2 = model(decoder_output, encoder_output)

    assert not torch.equal(train_output_1, train_output_2)

    model.eval()

    eval_output_1 = model(decoder_output, encoder_output)
    eval_output_2 = model(decoder_output, encoder_output)

    assert torch.equal(eval_output_1, eval_output_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



