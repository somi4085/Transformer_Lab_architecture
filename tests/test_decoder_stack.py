
import pytest
import torch
import torch.nn as nn

from src.layers.decoder_stack import DecoderStack
from src.layers.decoder_layer import DecoderLayer
from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.cross_attention import CrossAttention


def create_decoder_stack(num_layers=3, dropout=0.1):
    return DecoderStack(
        num_layers=num_layers,
        embedding_dim=256,
        num_heads=4,
        hidden_dim=1024,
        dropout=dropout
    )


def create_inputs(batch_size=2, decoder_seq_len=5, encoder_seq_len=7):
    decoder_input = torch.randn(batch_size, decoder_seq_len, 256)
    encoder_output = torch.randn(batch_size, encoder_seq_len, 256)

    return decoder_input, encoder_output


def test_output_shape():
    model = create_decoder_stack()
    decoder_input, encoder_output = create_inputs()

    output = model(decoder_input, encoder_output)

    assert output.shape == (2, 5, 256)


def test_correct_number_of_layers():
    model = create_decoder_stack(num_layers=4)

    assert len(model.layers) == 4

    assert all(
        isinstance(layer, DecoderLayer)
        for layer in model.layers
    )


def test_layers_have_independent_parameters():
    model = create_decoder_stack(num_layers=3)

    parameter_groups = [
        list(layer.parameters())
        for layer in model.layers
    ]

    assert all(
        len(group) == len(parameter_groups[0])
        for group in parameter_groups
    )

    for i in range(len(parameter_groups)):
        for j in range(i + 1, len(parameter_groups)):
            for param_i, param_j in zip(
                parameter_groups[i],
                parameter_groups[j]
            ):
                assert param_i is not param_j
                assert param_i.data_ptr() != param_j.data_ptr()


def test_sequential_layer_execution():
    model = create_decoder_stack(num_layers=3)
    decoder_input, encoder_output = create_inputs()

    execution_order = []
    hooks = []

    for index, layer in enumerate(model.layers):
        hook = layer.register_forward_hook(
            lambda module, inputs, output, idx=index:
            execution_order.append(idx)
        )
        hooks.append(hook)

    try:
        model(decoder_input, encoder_output)
    finally:
        for hook in hooks:
            hook.remove()

    assert execution_order == [0, 1, 2]


def test_causal_masking_in_every_layer():
    model = create_decoder_stack(num_layers=3)
    model.eval()

    decoder_input, encoder_output = create_inputs()

    captured_values = []
    hooks = []

    for layer in model.layers:
        attention_module = next(
            module for module in layer.modules()
            if isinstance(module, MultiHeadAttention)
        )

        def capture_causal(module, args, kwargs):
            captured_values.append(kwargs.get("causal", False))

        hook = attention_module.register_forward_pre_hook(
            capture_causal,
            with_kwargs=True
        )
        hooks.append(hook)

    try:
        model(decoder_input, encoder_output)
    finally:
        for hook in hooks:
            hook.remove()

    assert captured_values == [True, True, True]


def test_self_attention_mask_propagation():
    model = create_decoder_stack(num_layers=3)
    decoder_input, encoder_output = create_inputs()

    self_attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    captured_masks = []
    hooks = []

    for layer in model.layers:
        attention_module = next(
            module for module in layer.modules()
            if isinstance(module, MultiHeadAttention)
        )

        def capture_mask(module, args, kwargs):
            captured_masks.append(kwargs.get("mask"))

        hook = attention_module.register_forward_pre_hook(
            capture_mask,
            with_kwargs=True
        )
        hooks.append(hook)

    try:
        model(
            decoder_input,
            encoder_output,
            self_attention_mask=self_attention_mask
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_masks) == 3

    for captured_mask in captured_masks:
        assert torch.equal(captured_mask, self_attention_mask)


def test_encoder_mask_propagation():
    model = create_decoder_stack(num_layers=3)
    decoder_input, encoder_output = create_inputs()

    encoder_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0]
    ])

    captured_masks = []
    hooks = []

    for layer in model.layers:
        cross_attention_module = next(
            module for module in layer.modules()
            if isinstance(module, CrossAttention)
        )

        def capture_mask(module, args, kwargs):
            captured_masks.append(kwargs.get("encoder_mask"))

        hook = cross_attention_module.register_forward_pre_hook(
            capture_mask,
            with_kwargs=True
        )
        hooks.append(hook)

    try:
        model(
            decoder_input,
            encoder_output,
            encoder_mask=encoder_mask
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_masks) == 3

    for captured_mask in captured_masks:
        assert torch.equal(captured_mask, encoder_mask)


def test_gradient_flow():
    model = create_decoder_stack(num_layers=3)

    decoder_input, encoder_output = create_inputs()

    decoder_input.requires_grad_(True)
    encoder_output.requires_grad_(True)

    output = model(decoder_input, encoder_output)
    loss = output.sum()
    loss.backward()

    assert decoder_input.grad is not None
    assert encoder_output.grad is not None

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient for parameter: {name}"
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_decoder_stack()

    decoder_input, encoder_output = create_inputs(
        batch_size=batch_size
    )

    output = model(decoder_input, encoder_output)

    assert output.shape == (batch_size, 5, 256)


@pytest.mark.parametrize(
    "decoder_seq_len,encoder_seq_len",
    [(3, 7), (5, 10), (8, 4), (12, 15)]
)
def test_different_sequence_lengths(
    decoder_seq_len,
    encoder_seq_len
):
    model = create_decoder_stack()

    decoder_input, encoder_output = create_inputs(
        decoder_seq_len=decoder_seq_len,
        encoder_seq_len=encoder_seq_len
    )

    output = model(decoder_input, encoder_output)

    assert output.shape == (2, decoder_seq_len, 256)


@pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
def test_different_layer_counts(num_layers):
    model = create_decoder_stack(num_layers=num_layers)

    decoder_input, encoder_output = create_inputs()

    output = model(decoder_input, encoder_output)

    assert len(model.layers) == num_layers
    assert output.shape == (2, 5, 256)


def test_encoder_output_remains_unchanged():
    model = create_decoder_stack(num_layers=3)
    decoder_input, encoder_output = create_inputs()

    original_encoder_output = encoder_output.clone()

    model(decoder_input, encoder_output)

    assert torch.equal(
        encoder_output,
        original_encoder_output
    )


def test_dropout_train_eval():
    model = create_decoder_stack(dropout=0.5)

    decoder_input, encoder_output = create_inputs()

    model.train()
    train_output_1 = model(decoder_input, encoder_output)
    train_output_2 = model(decoder_input, encoder_output)

    model.eval()
    eval_output_1 = model(decoder_input, encoder_output)
    eval_output_2 = model(decoder_input, encoder_output)

    assert not torch.equal(train_output_1, train_output_2)
    assert torch.allclose(eval_output_1, eval_output_2)


def test_output_is_finite():
    model = create_decoder_stack()
    decoder_input, encoder_output = create_inputs()

    output = model(decoder_input, encoder_output)

    assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

