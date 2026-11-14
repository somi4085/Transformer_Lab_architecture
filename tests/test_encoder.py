
import pytest
import torch
import torch.nn as nn

from src.layers.encoder import TransformerEncoder
from src.layers.encoder_stack import EncoderStack
from src.embeddings.input_embedding import InputEmbeddings
from src.attention.multi_head_attention import MultiHeadAttention

def create_encoder(
    num_layers=3,
    dropout=0.1
):
    return TransformerEncoder(
        vocab_size=10000,
        embedding_dim=256,
        max_sequence_length=128,
        num_layers=num_layers,
        num_heads=4,
        hidden_dim=1024,
        dropout=dropout
    )


def create_input(
    batch_size=2,
    sequence_length=5
):
    return torch.randint(
        0,
        10000,
        (batch_size, sequence_length)
    )


# 1. Output shape
def test_encoder_output_shape():
    model = create_encoder()
    token_ids = create_input()

    output = model(token_ids)

    assert output.shape == (2, 5, 256)


# 2. InputEmbedding module exists
def test_input_embedding_exists():
    model = create_encoder()

    assert any(
        isinstance(module, InputEmbeddings)
        for module in model.modules()
    )


# 3. EncoderStack module exists
def test_encoder_stack_exists():
    model = create_encoder()

    assert any(
        isinstance(module, EncoderStack)
        for module in model.modules()
    )


# 4. Padding mask propagation
def test_padding_mask_propagation():
    model = create_encoder(num_layers=3)

    token_ids = create_input()

    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    encoder_stack = next(
        module for module in model.modules()
        if isinstance(module, EncoderStack)
    )

    captured_masks = []

    def capture_mask(module, args, kwargs):
        captured_masks.append(kwargs.get("mask"))

    hook = encoder_stack.register_forward_pre_hook(
        capture_mask,
        with_kwargs=True
    )

    try:
        model(token_ids, mask=mask)
    finally:
        hook.remove()

    assert len(captured_masks) == 1
    assert torch.equal(captured_masks[0], mask)


# 5. Non-causal attention
def test_encoder_attention_is_non_causal():
    model = create_encoder(num_layers=3)
    model.eval()

    token_ids = create_input()

    attention_modules = [
        module
        for module in model.modules()
        if isinstance(module, MultiHeadAttention)
    ]

    assert len(attention_modules) == 3

    captured_values = []
    hooks = []

    for attention_module in attention_modules:

        def capture_causal(module, args, kwargs):
            captured_values.append(
                kwargs.get("causal", False)
            )

        hook = attention_module.register_forward_pre_hook(
            capture_causal,
            with_kwargs=True
        )

        hooks.append(hook)

    try:
        model(token_ids)
    finally:
        for hook in hooks:
            hook.remove()

    assert captured_values == [False, False, False]


# 6. Gradient flow
def test_encoder_gradient_flow():
    model = create_encoder()

    token_ids = create_input()

    output = model(token_ids)
    loss = output.sum()

    loss.backward()

    parameters_with_grad = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(parameters_with_grad) > 0

    for parameter in parameters_with_grad:
        assert parameter.grad is not None


# 7. Different batch sizes
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4]
)
def test_different_batch_sizes(batch_size):
    model = create_encoder()

    token_ids = create_input(
        batch_size=batch_size
    )

    output = model(token_ids)

    assert output.shape == (
        batch_size,
        5,
        256
    )


# 8. Different sequence lengths
@pytest.mark.parametrize(
    "sequence_length",
    [3, 5, 10, 20]
)
def test_different_sequence_lengths(sequence_length):
    model = create_encoder()

    token_ids = create_input(
        sequence_length=sequence_length
    )

    output = model(token_ids)

    assert output.shape == (
        2,
        sequence_length,
        256
    )


# 9. Different layer counts
@pytest.mark.parametrize(
    "num_layers",
    [1, 2, 4, 6]
)
def test_different_layer_counts(num_layers):
    model = create_encoder(
        num_layers=num_layers
    )

    token_ids = create_input()

    output = model(token_ids)

    encoder_stack = next(
        module for module in model.modules()
        if isinstance(module, EncoderStack)
    )

    assert len(encoder_stack.layers) == num_layers
    assert output.shape == (2, 5, 256)


# 10. Train and evaluation mode
def test_encoder_train_eval_mode():
    model = create_encoder(dropout=0.5)

    token_ids = create_input()

    model.train()

    assert model.training

    train_output_1 = model(token_ids)
    train_output_2 = model(token_ids)

    model.eval()

    assert not model.training

    eval_output_1 = model(token_ids)
    eval_output_2 = model(token_ids)

    assert not torch.equal(
        train_output_1,
        train_output_2
    )

    assert torch.allclose(
        eval_output_1,
        eval_output_2
    )


# 11. Output should contain finite values
def test_encoder_output_is_finite():
    model = create_encoder()

    token_ids = create_input()

    output = model(token_ids)

    assert torch.isfinite(output).all()


# 12. Parameter registration
def test_encoder_parameter_registration():
    model = create_encoder()

    parameters = list(model.parameters())

    assert len(parameters) > 0

    assert all(
        isinstance(parameter, nn.Parameter)
        for parameter in parameters
    )


# 13. Model has trainable parameters
def test_encoder_has_trainable_parameters():
    model = create_encoder()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
