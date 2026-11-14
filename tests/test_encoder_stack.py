
import pytest
import torch

from src.layers.encoder_stack import EncoderStack
from src.layers.encoder_layer import EncoderLayer
from src.attention.multi_head_attention import MultiHeadAttention

def create_encoder_stack(
    num_layers=6,
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
    dropout=0.0
):
    return EncoderStack(
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
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


# 1. Output shape
def test_output_shape():
    model = create_encoder_stack()

    x = create_input()

    output = model(x)

    assert output.shape == (2, 5, 256)


# 2. Layer count
def test_layer_count():
    model = create_encoder_stack(num_layers=6)

    assert len(model.layers) == 6

    for layer in model.layers:
        assert isinstance(layer, EncoderLayer)


# 3. Independent parameters
def test_independent_parameters():
    model = create_encoder_stack(num_layers=3)

    first_layer_parameters = list(
        model.layers[0].parameters()
    )

    second_layer_parameters = list(
        model.layers[1].parameters()
    )

    assert len(first_layer_parameters) > 0
    assert len(second_layer_parameters) > 0

    for parameter_1, parameter_2 in zip(
        first_layer_parameters,
        second_layer_parameters
    ):
        assert parameter_1 is not parameter_2
        assert parameter_1.data_ptr() != parameter_2.data_ptr()


# 4. Sequential execution
def test_sequential_execution():
    model = create_encoder_stack(num_layers=3)
    model.eval()

    x = create_input()

    execution_order = []

    hooks = []

    for index, layer in enumerate(model.layers):

        def capture_execution(
            module,
            args,
            output,
            layer_index=index
        ):
            execution_order.append(layer_index)

        hooks.append(
            layer.register_forward_hook(capture_execution)
        )

    try:
        output = model(x)
    finally:
        for hook in hooks:
            hook.remove()

    assert execution_order == [0, 1, 2]
    assert output.shape == (2, 5, 256)


# 5. Non-causal attention
def test_non_causal_attention():
    model = create_encoder_stack(num_layers=3)
    model.eval()

    x = create_input()

    captured_causal_values = []
    hooks = []

    for layer in model.layers:

        attention_module = next(
            module
            for module in layer.modules()
            if isinstance(module, MultiHeadAttention)
        )

        def capture_causal(module, args, kwargs):
            captured_causal_values.append(
                kwargs.get("causal", False)
            )

        hooks.append(
            attention_module.register_forward_pre_hook(
                capture_causal,
                with_kwargs=True
            )
        )

    try:
        model(x)
    finally:
        for hook in hooks:
            hook.remove()

    assert captured_causal_values == [False, False, False]


# 6. Padding mask propagation
def test_padding_mask():
    model = create_encoder_stack(num_layers=3)

    x = create_input()

    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    captured_masks = []
    hooks = []

    for layer in model.layers:

        def capture_mask(
            module,
            args,
            kwargs
        ):
            captured_masks.append(kwargs.get("mask"))

        hooks.append(
            layer.register_forward_pre_hook(
                capture_mask,
                with_kwargs=True
            )
        )

    try:
        output = model(x, mask=mask)
    finally:
        for hook in hooks:
            hook.remove()

    assert output.shape == (2, 5, 256)
    assert len(captured_masks) == 3

    for captured_mask in captured_masks:
        assert torch.equal(captured_mask, mask)


# 7. Gradient flow
def test_gradient_flow():
    model = create_encoder_stack(num_layers=3)

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


# 8. Different batch sizes
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_encoder_stack(num_layers=2)

    x = create_input(batch_size=batch_size)

    output = model(x)

    assert output.shape == (batch_size, 5, 256)


# 9. Different sequence lengths
@pytest.mark.parametrize("sequence_length", [1, 5, 10])
def test_different_sequence_lengths(sequence_length):
    model = create_encoder_stack(num_layers=2)

    x = create_input(sequence_length=sequence_length)

    output = model(x)

    assert output.shape == (2, sequence_length, 256)


# 10. Different layer counts
@pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
def test_different_layer_counts(num_layers):
    model = create_encoder_stack(num_layers=num_layers)

    x = create_input()

    output = model(x)

    assert len(model.layers) == num_layers
    assert output.shape == (2, 5, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
