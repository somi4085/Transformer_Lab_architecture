
import pytest
import torch
import torch.nn as nn

from src.layers.decoder import TransformerDecoder
from src.layers.decoder_stack import DecoderStack
from src.layers.decoder_layer import DecoderLayer
from src.embeddings.input_embedding import InputEmbeddings
from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.cross_attention import CrossAttention


def create_decoder(num_layers=3, dropout=0.1):
    return TransformerDecoder(
        vocab_size=10000,
        embedding_dim=256,
        max_sequence_length=128,
        num_layers=num_layers,
        num_heads=4,
        hidden_dim=1024,
        dropout=dropout
    )


def create_inputs(
    batch_size=2,
    decoder_seq_len=5,
    encoder_seq_len=7
):
    decoder_token_ids = torch.randint(
        0, 10000, (batch_size, decoder_seq_len)
    )

    encoder_output = torch.randn(
        batch_size, encoder_seq_len, 256
    )

    return decoder_token_ids, encoder_output


# 1. Output shape
def test_decoder_output_shape():
    model = create_decoder()
    decoder_token_ids, encoder_output = create_inputs()

    output = model(decoder_token_ids, encoder_output)

    assert output.shape == (2, 5, 256)


# 2. InputEmbedding exists
def test_input_embedding_exists():
    model = create_decoder()

    assert any(
        isinstance(module, InputEmbeddings)
        for module in model.modules()
    )


# 3. DecoderStack exists
def test_decoder_stack_exists():
    model = create_decoder()

    assert any(
        isinstance(module, DecoderStack)
        for module in model.modules()
    )


# 4. Encoder output propagation
def test_encoder_output_propagation():
    model = create_decoder()
    decoder_token_ids, encoder_output = create_inputs()

    decoder_stack = next(
        module for module in model.modules()
        if isinstance(module, DecoderStack)
    )

    captured_encoder_outputs = []

    def capture_encoder_output(module, args, kwargs):
        captured_encoder_outputs.append(
            args[1] if len(args) > 1
            else kwargs.get("encoder_output")
        )

    hook = decoder_stack.register_forward_pre_hook(
        capture_encoder_output,
        with_kwargs=True
    )

    try:
        model(decoder_token_ids, encoder_output)
    finally:
        hook.remove()

    assert len(captured_encoder_outputs) == 1
    assert torch.equal(
        captured_encoder_outputs[0],
        encoder_output
    )


# 5. Self-attention mask propagation
def test_self_attention_mask_propagation():
    model = create_decoder()
    decoder_token_ids, encoder_output = create_inputs()

    self_attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    attention_modules = [
        module for module in model.modules()
        if isinstance(module, MultiHeadAttention)
    ]

    captured_masks = []
    hooks = []

    for attention_module in attention_modules:

        def capture_mask(module, args, kwargs):
            captured_masks.append(kwargs.get("mask"))

        hooks.append(
            attention_module.register_forward_pre_hook(
                capture_mask,
                with_kwargs=True
            )
        )

    try:
        model(
            decoder_token_ids,
            encoder_output,
            self_attention_mask=self_attention_mask
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_masks) == 3

    for captured_mask in captured_masks:
        assert torch.equal(captured_mask, self_attention_mask)


# 6. Encoder mask propagation
def test_encoder_mask_propagation():
    model = create_decoder()
    decoder_token_ids, encoder_output = create_inputs()

    encoder_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0]
    ])

    cross_attention_modules = [
        module for module in model.modules()
        if isinstance(module, CrossAttention)
    ]

    captured_masks = []
    hooks = []

    for cross_attention_module in cross_attention_modules:

        def capture_mask(module, args, kwargs):
            captured_masks.append(
                kwargs.get("encoder_mask")
            )

        hooks.append(
            cross_attention_module.register_forward_pre_hook(
                capture_mask,
                with_kwargs=True
            )
        )

    try:
        model(
            decoder_token_ids,
            encoder_output,
            encoder_mask=encoder_mask
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_masks) == 3

    for captured_mask in captured_masks:
        assert torch.equal(captured_mask, encoder_mask)


# 7. Causal masking
def test_causal_masking_in_every_layer():
    model = create_decoder(num_layers=3)
    model.eval()

    decoder_token_ids, encoder_output = create_inputs()

    attention_modules = [
        module for module in model.modules()
        if isinstance(module, MultiHeadAttention)
    ]

    captured_values = []
    hooks = []

    for attention_module in attention_modules:

        def capture_causal(module, args, kwargs):
            captured_values.append(
                kwargs.get("causal", False)
            )

        hooks.append(
            attention_module.register_forward_pre_hook(
                capture_causal,
                with_kwargs=True
            )
        )

    try:
        model(decoder_token_ids, encoder_output)
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_values) == 3
    assert captured_values == [True, True, True]


# 8. Cross-attention receives encoder output

def test_cross_attention_receives_encoder_output():
    model = create_decoder()
    decoder_token_ids, encoder_output = create_inputs()

    cross_attention_modules = [
        module for module in model.modules()
        if isinstance(module, CrossAttention)
    ]

    captured_encoder_outputs = []
    hooks = []

    for cross_attention_module in cross_attention_modules:

        def capture_encoder(module, args, kwargs):
            if len(args) > 1:
                captured_encoder_outputs.append(args[1])
            else:
                captured_encoder_outputs.append(
                    kwargs.get("encoder_output")
                )

        hooks.append(
            cross_attention_module.register_forward_pre_hook(
                capture_encoder,
                with_kwargs=True
            )
        )

    try:
        model(decoder_token_ids, encoder_output)
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_encoder_outputs) == 3

    for captured_output in captured_encoder_outputs:
        assert captured_output is not None
        assert torch.equal(captured_output, encoder_output)


# 9. Gradient flow
def test_decoder_gradient_flow():
    model = create_decoder()

    decoder_token_ids, encoder_output = create_inputs()
    encoder_output.requires_grad_(True)

    output = model(decoder_token_ids, encoder_output)
    loss = output.sum()

    loss.backward()

    assert encoder_output.grad is not None

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Missing gradient: {name}"
            )


# 10. Different batch sizes
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_decoder()

    decoder_token_ids, encoder_output = create_inputs(
        batch_size=batch_size
    )

    output = model(decoder_token_ids, encoder_output)

    assert output.shape == (batch_size, 5, 256)


# 11. Different sequence lengths
@pytest.mark.parametrize(
    "decoder_seq_len,encoder_seq_len",
    [(3, 7), (5, 10), (8, 4), (12, 15)]
)
def test_different_sequence_lengths(
    decoder_seq_len,
    encoder_seq_len
):
    model = create_decoder()

    decoder_token_ids, encoder_output = create_inputs(
        decoder_seq_len=decoder_seq_len,
        encoder_seq_len=encoder_seq_len
    )

    output = model(decoder_token_ids, encoder_output)

    assert output.shape == (
        2, decoder_seq_len, 256
    )


# 12. Train and evaluation mode
def test_decoder_train_eval_mode():
    model = create_decoder(dropout=0.5)

    decoder_token_ids, encoder_output = create_inputs()

    model.train()

    train_output_1 = model(
        decoder_token_ids, encoder_output
    )
    train_output_2 = model(
        decoder_token_ids, encoder_output
    )

    model.eval()

    eval_output_1 = model(
        decoder_token_ids, encoder_output
    )
    eval_output_2 = model(
        decoder_token_ids, encoder_output
    )

    assert not torch.equal(
        train_output_1, train_output_2
    )

    assert torch.allclose(
        eval_output_1, eval_output_2
    )


# 13. Output finiteness
def test_decoder_output_is_finite():
    model = create_decoder()

    decoder_token_ids, encoder_output = create_inputs()

    output = model(decoder_token_ids, encoder_output)

    assert torch.isfinite(output).all()


# 14. Parameter registration
def test_decoder_parameter_registration():
    model = create_decoder()

    parameters = list(model.parameters())

    assert len(parameters) > 0

    assert all(
        isinstance(parameter, nn.Parameter)
        for parameter in parameters
    )


# 15. Different layer counts
@pytest.mark.parametrize("num_layers", [1, 2, 4, 6])
def test_different_layer_counts(num_layers):
    model = create_decoder(num_layers=num_layers)

    decoder_token_ids, encoder_output = create_inputs()

    output = model(decoder_token_ids, encoder_output)

    decoder_stack = next(
        module for module in model.modules()
        if isinstance(module, DecoderStack)
    )

    assert len(decoder_stack.layers) == num_layers

    assert output.shape == (2, 5, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
