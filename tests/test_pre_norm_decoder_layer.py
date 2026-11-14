
import pytest
import torch
import torch.nn as nn

from src.layers.pre_norm_decoder_layer import PreNormDecoderLayer
from src.layers.pre_norm import PreNorm
from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.cross_attention import CrossAttention
from src.layers.feed_forward import FeedForwardNetwork


def create_model(dropout=0.1):
    return PreNormDecoderLayer(
        embedding_dim=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=dropout
    )


def create_inputs(batch_size=2, decoder_length=5, encoder_length=7):
    decoder_input = torch.randn(batch_size, decoder_length, 256)
    encoder_output = torch.randn(batch_size, encoder_length, 256)

    return decoder_input, encoder_output


def test_output_shape():
    model = create_model()
    decoder_input, encoder_output = create_inputs()

    output = model(decoder_input, encoder_output)

    assert output.shape == (2, 5, 256)


def test_self_attention_exists():
    model = create_model()

    assert isinstance(model.self_attention, MultiHeadAttention)


def test_cross_attention_exists():
    model = create_model()

    assert isinstance(model.cross_attention, CrossAttention)


def test_feed_forward_exists():
    model = create_model()

    assert isinstance(model.feed_forward, FeedForwardNetwork)


def test_three_independent_pre_norm_modules():
    model = create_model()

    assert isinstance(model.pre_norm1, PreNorm)
    assert isinstance(model.pre_norm2, PreNorm)
    assert isinstance(model.pre_norm3, PreNorm)

    assert len({
        id(model.pre_norm1),
        id(model.pre_norm2),
        id(model.pre_norm3)
    }) == 3


def test_self_attention_is_causal():
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs()
    captured = {}

    def attention_hook(module, args, kwargs):
        captured["causal"] = kwargs.get("causal")

    handle = model.self_attention.register_forward_pre_hook(
        attention_hook,
        with_kwargs=True
    )

    model(decoder_input, encoder_output)
    handle.remove()

    assert captured["causal"] is True


def test_cross_attention_receives_encoder_output():
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs()
    captured = {}

    def cross_attention_hook(module, args, kwargs):
        captured["encoder_output"] = (
            args[1] if len(args) > 1
            else kwargs.get("encoder_output")
        )

    handle = model.cross_attention.register_forward_pre_hook(
        cross_attention_hook,
        with_kwargs=True
    )

    model(decoder_input, encoder_output)
    handle.remove()

    assert captured["encoder_output"] is encoder_output


def test_self_attention_mask():
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs()

    self_attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    output = model(
        decoder_input,
        encoder_output,
        self_attention_mask=self_attention_mask
    )

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


def test_encoder_mask():
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs()

    encoder_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])

    output = model(
        decoder_input,
        encoder_output,
        encoder_mask=encoder_mask
    )

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


def test_both_masks_together():
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs()

    self_attention_mask = torch.ones(2, 5)
    encoder_mask = torch.ones(2, 7)

    output = model(
        decoder_input,
        encoder_output,
        self_attention_mask=self_attention_mask,
        encoder_mask=encoder_mask
    )

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


def test_gradient_flow():
    model = create_model()

    decoder_input, encoder_output = create_inputs()

    decoder_input.requires_grad_(True)
    encoder_output.requires_grad_(True)

    output = model(decoder_input, encoder_output)
    loss = output.sum()
    loss.backward()

    assert decoder_input.grad is not None
    assert encoder_output.grad is not None

    assert torch.isfinite(decoder_input.grad).all()
    assert torch.isfinite(encoder_output.grad).all()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs(
        batch_size=batch_size
    )

    output = model(decoder_input, encoder_output)

    assert output.shape == (batch_size, 5, 256)


@pytest.mark.parametrize("decoder_length", [1, 3, 5, 10])
def test_different_decoder_sequence_lengths(decoder_length):
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs(
        decoder_length=decoder_length
    )

    output = model(decoder_input, encoder_output)

    assert output.shape == (2, decoder_length, 256)


@pytest.mark.parametrize("encoder_length", [1, 4, 7, 10])
def test_different_encoder_sequence_lengths(encoder_length):
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs(
        encoder_length=encoder_length
    )

    output = model(decoder_input, encoder_output)

    assert output.shape == (2, 5, 256)


def test_dropout_train_eval():
    model = create_model(dropout=0.5)
    decoder_input, encoder_output = create_inputs()

    model.eval()
    output_eval1 = model(decoder_input, encoder_output)
    output_eval2 = model(decoder_input, encoder_output)

    assert torch.allclose(output_eval1, output_eval2)

    model.train()
    output_train1 = model(decoder_input, encoder_output)
    output_train2 = model(decoder_input, encoder_output)

    assert not torch.allclose(output_train1, output_train2)


def test_output_is_finite():
    model = create_model()

    decoder_input, encoder_output = create_inputs()
    output = model(decoder_input, encoder_output)

    assert torch.isfinite(output).all()


@pytest.mark.parametrize("num_heads", [3, 7, 9])
def test_invalid_head_configuration(num_heads):
    with pytest.raises(ValueError):
        PreNormDecoderLayer(
            embedding_dim=256,
            num_heads=num_heads,
            hidden_dim=1024
        )
