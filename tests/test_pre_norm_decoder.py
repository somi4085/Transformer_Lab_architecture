
import pytest
import torch
import torch.nn as nn

from src.layers.pre_norm_decoder import PreNormTransformerDecoder
from src.embeddings.input_embedding import InputEmbeddings
from src.layers.pre_norm_decoder_stack import PreNormDecoderStack
from src.layers.pre_norm_decoder_layer import PreNormDecoderLayer
from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.cross_attention import CrossAttention


def create_model(num_layers=3, dropout=0.1, max_sequence_length=512):

    return PreNormTransformerDecoder(
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
    encoder_output = torch.randn(2, 7, 256)

    output = model(token_ids, encoder_output)

    assert output.shape == (2, 5, 256)


def test_input_embeddings_exists():

    model = create_model()

    assert isinstance(model.input_embeddings, InputEmbeddings)


def test_decoder_stack_exists():

    model = create_model()

    assert isinstance(model.decoder_stack, PreNormDecoderStack)


def test_encoder_output_reaches_decoder_stack():

    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256)

    captured = {}

    def hook(module, args, kwargs):
        captured["encoder_output"] = kwargs.get(
            "encoder_output",
            args[1] if len(args) > 1 else None
        )

    handle = model.decoder_stack.register_forward_pre_hook(
        hook, with_kwargs=True
    )

    model(token_ids, encoder_output)

    handle.remove()

    assert captured["encoder_output"] is encoder_output


def test_self_attention_mask_propagation():

    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256)
    self_attention_mask = torch.ones(2, 5)

    captured = []

    def hook(module, args, kwargs):
        captured.append(kwargs.get("self_attention_mask"))

    handle = model.decoder_stack.register_forward_pre_hook(
        hook, with_kwargs=True
    )

    model(
        token_ids,
        encoder_output,
        self_attention_mask=self_attention_mask
    )

    handle.remove()

    assert captured[0] is self_attention_mask


def test_encoder_mask_propagation():

    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256)
    encoder_mask = torch.ones(2, 7)

    captured = []

    def hook(module, args, kwargs):
        captured.append(kwargs.get("encoder_mask"))

    handle = model.decoder_stack.register_forward_pre_hook(
        hook, with_kwargs=True
    )

    model(
        token_ids,
        encoder_output,
        encoder_mask=encoder_mask
    )

    handle.remove()

    assert captured[0] is encoder_mask


def test_all_decoder_layers_use_causal_self_attention():

    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256)

    captured = []

    def hook(module, args, kwargs):
        captured.append(kwargs.get("causal"))

    handles = []

    for layer in model.decoder_stack.layers:

        attention = next(
            module for module in layer.modules()
            if isinstance(module, MultiHeadAttention)
        )

        handles.append(
            attention.register_forward_pre_hook(
                hook, with_kwargs=True
            )
        )

    model(token_ids, encoder_output)

    for handle in handles:
        handle.remove()

    assert len(captured) == 3
    assert all(value is True for value in captured)


def test_cross_attention_receives_encoder_output():

    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256)

    captured = []

    def hook(module, args, kwargs):
        received = kwargs.get(
            "encoder_output",
            args[1] if len(args) > 1 else None
        )
        captured.append(received)

    handles = []

    for layer in model.decoder_stack.layers:

        cross_attention = next(
            module for module in layer.modules()
            if isinstance(module, CrossAttention)
        )

        handles.append(
            cross_attention.register_forward_pre_hook(
                hook, with_kwargs=True
            )
        )

    model(token_ids, encoder_output)

    for handle in handles:
        handle.remove()

    assert len(captured) == 3
    assert all(value is encoder_output for value in captured)


def test_gradient_flow():

    model = create_model()
    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256, requires_grad=True)

    output = model(token_ids, encoder_output)
    loss = output.mean()
    loss.backward()

    assert encoder_output.grad is not None

    for name, parameter in model.named_parameters():

        assert parameter.grad is not None, (
            f"Missing gradient for parameter: {name}"
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):

    model = create_model()

    token_ids = torch.randint(0, 10000, (batch_size, 5))
    encoder_output = torch.randn(batch_size, 7, 256)

    output = model(token_ids, encoder_output)

    assert output.shape == (batch_size, 5, 256)


@pytest.mark.parametrize(
    "decoder_length, encoder_length",
    [(3, 5), (5, 7), (8, 10)]
)
def test_different_sequence_lengths(decoder_length, encoder_length):

    model = create_model()

    token_ids = torch.randint(0, 10000, (2, decoder_length))
    encoder_output = torch.randn(2, encoder_length, 256)

    output = model(token_ids, encoder_output)

    assert output.shape == (2, decoder_length, 256)


def test_dropout_train_eval():

    model = create_model(dropout=0.5)

    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256)

    model.train()
    train_output = model(token_ids, encoder_output)

    model.eval()
    eval_output = model(token_ids, encoder_output)

    assert not torch.allclose(train_output, eval_output)


def test_output_contains_finite_values():

    model = create_model()

    token_ids = torch.randint(0, 10000, (2, 5))
    encoder_output = torch.randn(2, 7, 256)

    output = model(token_ids, encoder_output)

    assert torch.isfinite(output).all()


def test_parameters_registered():

    model = create_model()

    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert all(isinstance(parameter, nn.Parameter) for parameter in parameters)


def test_exceeding_max_sequence_length():

    model = create_model(max_sequence_length=5)

    token_ids = torch.randint(0, 10000, (2, 6))
    encoder_output = torch.randn(2, 7, 256)

    with pytest.raises(ValueError):
        model(token_ids, encoder_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
