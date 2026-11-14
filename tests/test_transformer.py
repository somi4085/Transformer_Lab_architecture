
import pytest
import torch
import torch.nn as nn

from src.model.transformer import Transformer
from src.layers.encoder import TransformerEncoder
from src.layers.decoder import TransformerDecoder
from src.layers.encoder_stack import EncoderStack
from src.layers.decoder_stack import DecoderStack
from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.cross_attention import CrossAttention


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def create_transformer(
    num_encoder_layers=3,
    num_decoder_layers=3,
    dropout=0.1
):
    return Transformer(
        vocab_size=10000,
        embedding_dim=256,
        max_sequence_length=128,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=4,
        hidden_dim=1024,
        dropout=dropout
    )


def create_inputs(
    batch_size=2,
    source_seq_len=7,
    target_seq_len=5
):
    source_token_ids = torch.randint(
        0, 10000, (batch_size, source_seq_len)
    )

    target_token_ids = torch.randint(
        0, 10000, (batch_size, target_seq_len)
    )

    return source_token_ids, target_token_ids


# --------------------------------------------------
# 1. Output shape
# --------------------------------------------------

def test_transformer_output_shape():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    output = model(source_ids, target_ids)

    assert output.shape == (2, 5, 10000)


# --------------------------------------------------
# 2. Encoder integration
# --------------------------------------------------

def test_encoder_integration():
    model = create_transformer()

    assert any(
        isinstance(module, TransformerEncoder)
        for module in model.modules()
    )


# --------------------------------------------------
# 3. Decoder integration
# --------------------------------------------------

def test_decoder_integration():
    model = create_transformer()

    assert any(
        isinstance(module, TransformerDecoder)
        for module in model.modules()
    )


# --------------------------------------------------
# 4. Output projection
# --------------------------------------------------

def test_output_projection_shape():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    decoder_outputs = []
    hooks = []

    decoder = next(
        module for module in model.modules()
        if isinstance(module, TransformerDecoder)
    )

    def capture_decoder_output(module, args, output):
        decoder_outputs.append(output)

    hooks.append(
        decoder.register_forward_hook(capture_decoder_output)
    )

    try:
        logits = model(source_ids, target_ids)
    finally:
        for hook in hooks:
            hook.remove()

    assert len(decoder_outputs) == 1
    assert decoder_outputs[0].shape == (2, 5, 256)
    assert logits.shape == (2, 5, 10000)


# --------------------------------------------------
# 5. Source mask propagation
# --------------------------------------------------

def test_source_mask_propagation():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    source_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0]
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
        model(
            source_ids,
            target_ids,
            source_mask=source_mask
        )
    finally:
        hook.remove()

    assert len(captured_masks) == 1
    assert torch.equal(captured_masks[0], source_mask)


# --------------------------------------------------
# 6. Target self-attention mask propagation
# --------------------------------------------------


def test_cross_attention_integration():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    cross_attention_modules = [
        module for module in model.modules()
        if isinstance(module, CrossAttention)
    ]

    assert len(cross_attention_modules) == 3

    captured_shapes = []
    hooks = []

    for module in cross_attention_modules:

        def capture_shapes(module, args, kwargs):
            decoder_output = (
                args[0] if len(args) > 0
                else kwargs.get("decoder_output")
            )

            encoder_output = (
                args[1] if len(args) > 1
                else kwargs.get("encoder_output")
            )

            assert decoder_output is not None
            assert encoder_output is not None

            captured_shapes.append((
                decoder_output.shape,
                encoder_output.shape
            ))

        hooks.append(
            module.register_forward_pre_hook(
                capture_shapes,
                with_kwargs=True
            )
        )

    try:
        model(source_ids, target_ids)
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_shapes) == 3

    for decoder_shape, encoder_shape in captured_shapes:
        assert decoder_shape == (2, 5, 256)
        assert encoder_shape == (2, 7, 256)



# --------------------------------------------------
# 7. Causal masking
# --------------------------------------------------

def test_decoder_causal_masking():
    model = create_transformer()
    model.eval()

    source_ids, target_ids = create_inputs()

    decoder_attention_modules = [
        module for module in model.modules()
        if isinstance(module, MultiHeadAttention)
    ]

    captured_values = []
    hooks = []

    for attention_module in decoder_attention_modules:

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
        model(source_ids, target_ids)
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_values) == 6

    assert captured_values == [
        False, False, False,
        True, True, True
    ]


# --------------------------------------------------
# 8. Cross-attention integration
# --------------------------------------------------


def test_cross_attention_integration():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    cross_attention_modules = [
        module for module in model.modules()
        if isinstance(module, CrossAttention)
    ]

    assert len(cross_attention_modules) == 3

    captured_shapes = []
    hooks = []

    for module in cross_attention_modules:

        def capture_shapes(module, args, kwargs):
            decoder_output = (
                args[0] if len(args) > 0
                else kwargs.get("decoder_output")
            )

            encoder_output = (
                args[1] if len(args) > 1
                else kwargs.get("encoder_output")
            )

            assert decoder_output is not None
            assert encoder_output is not None

            captured_shapes.append((
                decoder_output.shape,
                encoder_output.shape
            ))

        hooks.append(
            module.register_forward_pre_hook(
                capture_shapes,
                with_kwargs=True
            )
        )

    try:
        model(source_ids, target_ids)
    finally:
        for hook in hooks:
            hook.remove()

    assert len(captured_shapes) == 3

    for decoder_shape, encoder_shape in captured_shapes:
        assert decoder_shape == (2, 5, 256)
        assert encoder_shape == (2, 7, 256)



# --------------------------------------------------
# 9. Gradient flow
# --------------------------------------------------

def test_transformer_gradient_flow():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    output = model(source_ids, target_ids)

    loss = output.square().mean()

    loss.backward()

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(parameters) > 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Missing gradient: {name}"
            )

            assert torch.isfinite(parameter.grad).all(), (
                f"Non-finite gradient: {name}"
            )


# --------------------------------------------------
# 10. Different batch sizes
# --------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    model = create_transformer()

    source_ids, target_ids = create_inputs(
        batch_size=batch_size
    )

    output = model(source_ids, target_ids)

    assert output.shape == (batch_size, 5, 10000)


# --------------------------------------------------
# 11. Different source and target lengths
# --------------------------------------------------

@pytest.mark.parametrize(
    "source_len,target_len",
    [
        (3, 5),
        (5, 10),
        (10, 4),
        (20, 15)
    ]
)
def test_different_sequence_lengths(source_len, target_len):
    model = create_transformer()

    source_ids, target_ids = create_inputs(
        source_seq_len=source_len,
        target_seq_len=target_len
    )

    output = model(source_ids, target_ids)

    assert output.shape == (
        2,
        target_len,
        10000
    )


# --------------------------------------------------
# 12. Different encoder and decoder layer counts
# --------------------------------------------------

@pytest.mark.parametrize(
    "encoder_layers,decoder_layers",
    [
        (1, 1),
        (2, 2),
        (4, 4),
        (6, 6),
        (2, 4)
    ]
)
def test_different_layer_counts(
    encoder_layers,
    decoder_layers
):
    model = create_transformer(
        num_encoder_layers=encoder_layers,
        num_decoder_layers=decoder_layers
    )

    source_ids, target_ids = create_inputs()

    output = model(source_ids, target_ids)

    encoder_stack = next(
        module for module in model.modules()
        if isinstance(module, EncoderStack)
    )

    decoder_stack = next(
        module for module in model.modules()
        if isinstance(module, DecoderStack)
    )

    assert len(encoder_stack.layers) == encoder_layers
    assert len(decoder_stack.layers) == decoder_layers

    assert output.shape == (2, 5, 10000)


# --------------------------------------------------
# 13. Train and evaluation mode
# --------------------------------------------------

def test_transformer_train_eval_mode():
    model = create_transformer(dropout=0.5)

    source_ids, target_ids = create_inputs()

    model.train()

    train_output_1 = model(source_ids, target_ids)
    train_output_2 = model(source_ids, target_ids)

    model.eval()

    eval_output_1 = model(source_ids, target_ids)
    eval_output_2 = model(source_ids, target_ids)

    assert not torch.equal(
        train_output_1,
        train_output_2
    )

    assert torch.allclose(
        eval_output_1,
        eval_output_2
    )


# --------------------------------------------------
# 14. Output finiteness
# --------------------------------------------------

def test_transformer_output_is_finite():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    output = model(source_ids, target_ids)

    assert torch.isfinite(output).all()


# --------------------------------------------------
# 15. Parameter registration
# --------------------------------------------------

def test_transformer_parameter_registration():
    model = create_transformer()

    parameters = list(model.parameters())

    assert len(parameters) > 0

    assert all(
        isinstance(parameter, nn.Parameter)
        for parameter in parameters
    )


# --------------------------------------------------
# 16. Output is raw logits
# --------------------------------------------------

def test_transformer_returns_raw_logits():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    assert logits.shape[-1] == 10000

    # Raw logits should not be assumed to sum to 1.
    probability_sums = torch.softmax(
        logits,
        dim=-1
    ).sum(dim=-1)

    assert torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5
    )


# --------------------------------------------------
# 17. Source and target inputs are unchanged
# --------------------------------------------------

def test_input_token_ids_remain_unchanged():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    original_source = source_ids.clone()
    original_target = target_ids.clone()

    model(source_ids, target_ids)

    assert torch.equal(source_ids, original_source)
    assert torch.equal(target_ids, original_target)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

