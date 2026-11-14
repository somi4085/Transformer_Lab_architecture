
import pytest
import torch
import torch.nn as nn

from src.embeddings.input_embedding import InputEmbeddings
from src.layers.pre_norm_encoder import PreNormTransformerEncoder
from src.layers.pre_norm_encoder_stack import PreNormEncoderStack
def create_model(
    num_layers=3,
    dropout=0.1,
    max_sequence_length=512
):
    return PreNormEncoderStack(
        vocab_size=10000,
        embedding_dim=256,
        max_sequence_length=max_sequence_length,
        num_layers=num_layers,
        num_heads=8,
        hidden_dim=1024,
        dropout=dropout
    )


def test_output_shape():
    model = create_model()
    token_ids = torch.randint(0, 10000, (2, 5))

    output = model(token_ids)

    assert output.shape == (2, 5, 256)


def test_input_embedding_exists():
    model = create_model()

    assert isinstance(model.input_embeddings, InputEmbeddings)


def test_encoder_stack_exists():
    model = create_model()

    assert isinstance(model.encoder_stack, PreNormEncoderStack)


def test_padding_mask_propagation():
    model = create_model()
    model.eval()

    token_ids = torch.randint(0, 10000, (2, 5))

    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    captured = {}

    def stack_hook(module, args, kwargs):
        captured["mask"] = kwargs.get("mask")

    handle = model.encoder_stack.register_forward_pre_hook(
        stack_hook,
        with_kwargs=True
    )

    output = model(token_ids, mask=mask)
    handle.remove()

    assert captured["mask"] is mask
    assert output.shape == (2, 5, 256)


def test_gradient_flow():
    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))

    output = model(token_ids)
    loss = output.sum()
    loss.backward()

    embedding_weight = (
        model.input_embeddings
        .token_embedding
        .embedding.weight
    )

    assert embedding_weight.grad is not None

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_model()
    model.eval()

    token_ids = torch.randint(0, 10000, (batch_size, 5))

    output = model(token_ids)

    assert output.shape == (batch_size, 5, 256)


@pytest.mark.parametrize("sequence_length", [1, 3, 5, 10])
def test_different_sequence_lengths(sequence_length):
    model = create_model()
    model.eval()

    token_ids = torch.randint(
        0,
        10000,
        (2, sequence_length)
    )

    output = model(token_ids)

    assert output.shape == (2, sequence_length, 256)


@pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
def test_different_layer_counts(num_layers):
    model = create_model(num_layers=num_layers)
    model.eval()

    token_ids = torch.randint(0, 10000, (2, 5))

    output = model(token_ids)

    assert len(model.encoder_stack.layers) == num_layers
    assert output.shape == (2, 5, 256)


def test_output_is_finite():
    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))

    output = model(token_ids)

    assert torch.isfinite(output).all()


def test_max_sequence_length():
    model = create_model(max_sequence_length=10)
    model.eval()

    token_ids = torch.randint(0, 10000, (2, 10))

    output = model(token_ids)

    assert output.shape == (2, 10, 256)


def test_exceeding_max_sequence_length():
    model = create_model(max_sequence_length=5)

    token_ids = torch.randint(0, 10000, (2, 6))

    with pytest.raises(ValueError):
        model(token_ids)


def test_train_eval_mode():
    model = create_model(dropout=0.5)
    token_ids = torch.randint(0, 10000, (2, 5))

    model.eval()

    output_eval1 = model(token_ids)
    output_eval2 = model(token_ids)

    assert torch.allclose(output_eval1, output_eval2)

    model.train()

    output_train1 = model(token_ids)
    output_train2 = model(token_ids)

    assert not torch.allclose(output_train1, output_train2)


def test_parameter_registration():
    model = create_model()

    registered_parameters = dict(model.named_parameters())

    assert any(
        "input_embeddings" in name
        for name in registered_parameters
    )

    assert any(
        "encoder_stack" in name
        for name in registered_parameters
    )


def test_encoder_output_is_not_logits():
    model = create_model()
    token_ids = torch.randint(0, 10000, (2, 5))

    output = model(token_ids)

    # Encoder returns contextual representations,
    # not vocabulary logits.
    assert output.shape[-1] == 256
    assert output.shape[-1] != 10000
