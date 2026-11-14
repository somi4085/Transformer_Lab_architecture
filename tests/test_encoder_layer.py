import torch
import pytest
import torch.nn as nn
from src.layers.encoder_layer import EncoderLayer
from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import FeedForwardNetwork
from src.layers.add_norm import AddNorm

def test_encoder_layer():
    model = EncoderLayer(
        embedding_dim = 256,
        num_heads = 8,
        hidden_dim = 1024,
        dropout = 0.1
    )

    x = torch.randn(2,5,256)
    model.eval()

    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2,5,256)


    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


def create_layer(
    embedding_dim=256,
    num_head = 8,
    hidden_dim = 1024,
    dropout = 0.0
):
    return EncoderLayer(
        embedding_dim=embedding_dim,
        num_heads=num_head,
        hidden_dim= hidden_dim,
        dropout=dropout
    )
def create_input(
    batch_size=2,
    sequence_length=5,
    embedding_dim=256
):
    return torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )
def test_output_shape():
    model = create_layer()

    x = create_input()

    output = model(x)

    assert output.shape == (2,5,256)

def test_self_attention_module():
    model = create_layer()

    attention_modules = [
        module for module in model.modules()
        if isinstance(module,MultiHeadAttention)
    ] 

    assert len(attention_modules) == 1

def test_non_causal_attention():
    model = create_layer()
    model.eval()

    x = create_input()

    attention_modules = next(
        module for module in model.modules()
        if isinstance(module, MultiHeadAttention)
    )
    captured = {}

    def capture_attention_input(module,args,kwarg):
        captured["causal"] = kwarg.get("causal",False)
    hook = attention_modules.register_forward_pre_hook(
        capture_attention_input,
        with_kwargs = True
    )
    try:
        model(x,causal=False)
    finally:
        hook.remove()

    assert captured["causal"] is False

def test_first_add_norm_module():
    model = create_layer()

    add_norm_modules = [
        module for module in model.modules()
        if isinstance(module, AddNorm)
    ]

    assert len(add_norm_modules) == 2
    assert add_norm_modules[0] is not add_norm_modules[1]

def test_feed_forward_module():
    model = create_layer()

    feed_forward_modules = [
        module for module in model.modules()
        if isinstance(module,FeedForwardNetwork)
    ]
    assert len(feed_forward_modules) == 1

def test_second_add_norm_module():
    model = create_layer()

    add_norm_modules = [
        module for module in model.modules()
        if isinstance(module,AddNorm)
    ]

    assert len(add_norm_modules) == 2
    for module in add_norm_modules:
        assert module.layer_norm.normalized_shape == (256,)

def test_gradient_flow():
    model = create_layer()
    x = create_input()
    x.requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad).item() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient for {name}"
        )

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_size(batch_size):
    model = create_layer()
    x = create_input(batch_size=batch_size)
    output = model(x)
    assert output.shape == (batch_size,5,256)


@pytest.mark.parametrize("sequence_length", [1, 5, 10])
def test_different_sequence_lengths(sequence_length):
    model = create_layer()

    x = create_input(sequence_length=sequence_length)

    output = model(x)

    assert output.shape == (2, sequence_length, 256)

def test_padding_mask():
    model = create_layer()

    x = create_input()

    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    output = model(x, mask=mask, causal=False)

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    test_encoder_layer()