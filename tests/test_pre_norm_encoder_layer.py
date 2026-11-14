
import pytest
import torch
import torch.nn as nn

from src.layers.pre_norm_encoder_layer import PreNormEncoderLayer
from src.layers.pre_norm import PreNorm
from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import FeedForwardNetwork


def create_model(dropout=0.1):
    return PreNormEncoderLayer(
        embedding_dim=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=dropout
    )


def test_output_shape():
    model = create_model()
    x = torch.randn(2, 5, 256)

    output = model(x)

    assert output.shape == (2, 5, 256)


def test_self_attention_exists():
    model = create_model()

    assert isinstance(model.self_attention, MultiHeadAttention)


def test_two_independent_pre_norm_modules():
    model = create_model()

    assert isinstance(model.pre_norm1, PreNorm)
    assert isinstance(model.pre_norm2, PreNorm)
    assert model.pre_norm1 is not model.pre_norm2


def test_feed_forward_exists():
    model = create_model()

    assert isinstance(model.feed_forward, FeedForwardNetwork)


def test_attention_is_not_causal():
    model = create_model()
    model.eval()

    x = torch.randn(2, 5, 256)
    captured = {}

    def attention_hook(module, args, kwargs):
        captured["causal"] = kwargs.get("causal")

    handle = model.self_attention.register_forward_pre_hook(
        attention_hook,
        with_kwargs=True
    )

    model(x)
    handle.remove()

    assert captured["causal"] is False


def test_padding_mask():
    model = create_model()
    model.eval()

    x = torch.randn(2, 5, 256)
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    output = model(x, mask=mask)

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


def test_gradient_flow():
    model = create_model()
    x = torch.randn(2, 5, 256, requires_grad=True)

    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all()


def test_different_batch_sizes():
    model = create_model()
    model.eval()

    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 5, 256)
        output = model(x)

        assert output.shape == (batch_size, 5, 256)


def test_different_sequence_lengths():
    model = create_model()
    model.eval()

    for sequence_length in [1, 3, 5, 10]:
        x = torch.randn(2, sequence_length, 256)
        output = model(x)

        assert output.shape == (2, sequence_length, 256)


def test_dropout_train_eval():
    model = create_model(dropout=0.5)
    x = torch.randn(2, 5, 256)

    model.eval()
    output_eval1 = model(x)
    output_eval2 = model(x)

    assert torch.allclose(output_eval1, output_eval2)

    model.train()
    output_train1 = model(x)
    output_train2 = model(x)

    assert not torch.allclose(output_train1, output_train2)


def test_output_is_finite():
    model = create_model()

    x = torch.randn(2, 5, 256)
    output = model(x)

    assert torch.isfinite(output).all()


@pytest.mark.parametrize("num_heads", [3, 7, 9])
def test_invalid_head_configuration(num_heads):
    with pytest.raises(ValueError):
        PreNormEncoderLayer(
            embedding_dim=256,
            num_heads=num_heads,
            hidden_dim=1024
        )
