
import pytest
import torch
import torch.nn as nn

from src.layers.pre_norm_encoder_stack import PreNormEncoderStack
from src.layers.pre_norm_encoder_layer import PreNormEncoderLayer


def create_model(num_layers=3, dropout=0.1):
    return PreNormEncoderStack(
        num_layers=num_layers,
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


def test_correct_number_of_layers():
    model = create_model(num_layers=4)

    assert len(model.layers) == 4


def test_layers_are_pre_norm_encoder_layers():
    model = create_model()

    for layer in model.layers:
        assert isinstance(layer, PreNormEncoderLayer)


def test_layers_have_independent_parameters():
    model = create_model(num_layers=3)

    first_layer_params = list(model.layers[0].parameters())
    second_layer_params = list(model.layers[1].parameters())

    assert len(first_layer_params) > 0
    assert len(second_layer_params) > 0

    for param1, param2 in zip(
        first_layer_params,
        second_layer_params
    ):
        assert param1 is not param2


def test_layers_execute_sequentially():
    model = create_model(num_layers=3)
    model.eval()

    x = torch.randn(2, 5, 256)
    execution_order = []

    handles = []

    for index, layer in enumerate(model.layers):
        handle = layer.register_forward_hook(
            lambda module, inputs, output, i=index:
                execution_order.append(i)
        )
        handles.append(handle)

    model(x)

    for handle in handles:
        handle.remove()

    assert execution_order == [0, 1, 2]


def test_padding_mask_propagation():
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


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_model()
    model.eval()

    x = torch.randn(batch_size, 5, 256)

    output = model(x)

    assert output.shape == (batch_size, 5, 256)


@pytest.mark.parametrize("sequence_length", [1, 3, 5, 10])
def test_different_sequence_lengths(sequence_length):
    model = create_model()
    model.eval()

    x = torch.randn(2, sequence_length, 256)

    output = model(x)

    assert output.shape == (2, sequence_length, 256)


@pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
def test_different_layer_counts(num_layers):
    model = create_model(num_layers=num_layers)
    model.eval()

    x = torch.randn(2, 5, 256)

    output = model(x)

    assert len(model.layers) == num_layers
    assert output.shape == (2, 5, 256)


def test_output_is_finite():
    model = create_model()

    x = torch.randn(2, 5, 256)

    output = model(x)

    assert torch.isfinite(output).all()


def test_invalid_layer_count():
    with pytest.raises(ValueError):
        create_model(num_layers=0)

    with pytest.raises(ValueError):
        create_model(num_layers=-1)


def test_module_registration():
    model = create_model()

    registered_modules = dict(model.named_modules())

    for index in range(3):
        assert f"layers.{index}" in registered_modules


def test_train_eval_mode():
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
