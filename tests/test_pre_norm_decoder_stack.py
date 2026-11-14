
import pytest
import torch

from src.layers.pre_norm_decoder_stack import PreNormDecoderStack
from src.layers.pre_norm_decoder_layer import PreNormDecoderLayer


def create_model(num_layers=3, dropout=0.1):
    return PreNormDecoderStack(
        num_layers=num_layers,
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


def test_correct_number_of_layers():
    model = create_model(num_layers=4)

    assert len(model.layers) == 4


def test_layers_are_pre_norm_decoder_layers():
    model = create_model()

    for layer in model.layers:
        assert isinstance(layer, PreNormDecoderLayer)


def test_layers_have_independent_parameters():
    model = create_model(num_layers=3)

    first_layer_params = list(model.layers[0].parameters())
    second_layer_params = list(model.layers[1].parameters())

    assert len(first_layer_params) > 0
    assert len(second_layer_params) > 0

    for param1, param2 in zip(first_layer_params, second_layer_params):
        assert param1 is not param2


def test_layers_execute_sequentially():
    model = create_model(num_layers=3)
    model.eval()

    decoder_input, encoder_output = create_inputs()
    execution_order = []
    handles = []

    for index, layer in enumerate(model.layers):
        handle = layer.register_forward_hook(
            lambda module, inputs, output, i=index:
                execution_order.append(i)
        )
        handles.append(handle)

    model(decoder_input, encoder_output)

    for handle in handles:
        handle.remove()

    assert execution_order == [0, 1, 2]


def test_self_attention_mask_propagation():
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs()

    self_attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    captured_masks = []
    handles = []

    for layer in model.layers:
        handle = layer.register_forward_pre_hook(
            lambda module, args, kwargs:
                captured_masks.append(kwargs.get("self_attention_mask")),
            with_kwargs=True
        )
        handles.append(handle)

    output = model(
        decoder_input,
        encoder_output,
        self_attention_mask=self_attention_mask
    )

    for handle in handles:
        handle.remove()

    assert output.shape == (2, 5, 256)
    assert len(captured_masks) == 3

    for captured_mask in captured_masks:
        assert captured_mask is self_attention_mask


def test_encoder_mask_propagation():
    model = create_model()
    model.eval()

    decoder_input, encoder_output = create_inputs()

    encoder_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])

    captured_masks = []
    handles = []

    for layer in model.layers:
        handle = layer.register_forward_pre_hook(
            lambda module, args, kwargs:
                captured_masks.append(kwargs.get("encoder_mask")),
            with_kwargs=True
        )
        handles.append(handle)

    output = model(
        decoder_input,
        encoder_output,
        encoder_mask=encoder_mask
    )

    for handle in handles:
        handle.remove()

    assert output.shape == (2, 5, 256)
    assert len(captured_masks) == 3

    for captured_mask in captured_masks:
        assert captured_mask is encoder_mask


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


def test_same_encoder_output_passed_to_every_layer():
    model = create_model(num_layers=3)
    model.eval()

    decoder_input, encoder_output = create_inputs()

    captured_encoder_outputs = []
    handles = []

    for layer in model.layers:
        handle = layer.register_forward_pre_hook(
            lambda module, args, kwargs:
                captured_encoder_outputs.append(
                    args[1] if len(args) > 1
                    else kwargs.get("encoder_output")
                ),
            with_kwargs=True
        )
        handles.append(handle)

    model(decoder_input, encoder_output)

    for handle in handles:
        handle.remove()

    assert len(captured_encoder_outputs) == 3

    for captured in captured_encoder_outputs:
        assert captured is encoder_output


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


@pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
def test_different_layer_counts(num_layers):
    model = create_model(num_layers=num_layers)
    model.eval()

    decoder_input, encoder_output = create_inputs()

    output = model(decoder_input, encoder_output)

    assert len(model.layers) == num_layers
    assert output.shape == (2, 5, 256)


def test_output_is_finite():
    model = create_model()

    decoder_input, encoder_output = create_inputs()
    output = model(decoder_input, encoder_output)

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
    decoder_input, encoder_output = create_inputs()

    model.eval()

    output_eval1 = model(decoder_input, encoder_output)
    output_eval2 = model(decoder_input, encoder_output)

    assert torch.allclose(output_eval1, output_eval2)

    model.train()

    output_train1 = model(decoder_input, encoder_output)
    output_train2 = model(decoder_input, encoder_output)

    assert not torch.allclose(output_train1, output_train2)
