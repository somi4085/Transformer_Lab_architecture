
import pytest
import torch
import torch.nn as nn

from src.model.pre_norm_transformer import PreNormTransformer
from src.layers.pre_norm_encoder import PreNormTransformerEncoder
from src.layers.pre_norm_decoder import PreNormTransformerDecoder


def create_model(num_layers=3, dropout=0.1, max_sequence_length=512):

    return PreNormTransformer(
        vocab_size=10000,
        embedding_dim=256,
        max_sequence_length=max_sequence_length,
        num_layers=num_layers,
        num_heads=8,
        hidden_dim=1024,
        dropout=dropout
    )


def test_output_logits_shape():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    logits = model(source_token_ids, target_token_ids)

    assert logits.shape == (2, 5, 10000)


def test_encoder_component_exists():

    model = create_model()

    assert isinstance(model.encoder, PreNormTransformerEncoder)


def test_decoder_component_exists():

    model = create_model()

    assert isinstance(model.decoder, PreNormTransformerDecoder)


def test_output_projection_exists():

    model = create_model()

    assert isinstance(model.output_projection, nn.Linear)
    assert model.output_projection.in_features == 256
    assert model.output_projection.out_features == 10000


def test_encoder_receives_source_tokens():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    captured = {}

    def hook(module, args, kwargs):
        captured["token_ids"] = kwargs.get(
            "token_ids",
            args[0] if args else None
        )

    handle = model.encoder.register_forward_pre_hook(
        hook, with_kwargs=True
    )

    model(source_token_ids, target_token_ids)

    handle.remove()

    assert captured["token_ids"] is source_token_ids


def test_decoder_receives_target_tokens():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    captured = {}

    def hook(module, args, kwargs):
        captured["token_ids"] = kwargs.get(
            "token_ids",
            args[0] if args else None
        )

    handle = model.decoder.register_forward_pre_hook(
        hook, with_kwargs=True
    )

    model(source_token_ids, target_token_ids)

    handle.remove()

    assert captured["token_ids"] is target_token_ids


def test_source_mask_propagation():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))
    source_mask = torch.ones(2, 7)

    captured_encoder = {}
    captured_decoder = {}

    def encoder_hook(module, args, kwargs):
        captured_encoder["mask"] = kwargs.get("mask")

    def decoder_hook(module, args, kwargs):
        captured_decoder["mask"] = kwargs.get("encoder_mask")

    encoder_handle = model.encoder.register_forward_pre_hook(
        encoder_hook, with_kwargs=True
    )

    decoder_handle = model.decoder.register_forward_pre_hook(
        decoder_hook, with_kwargs=True
    )

    model(
        source_token_ids,
        target_token_ids,
        source_mask=source_mask
    )

    encoder_handle.remove()
    decoder_handle.remove()

    assert captured_encoder["mask"] is source_mask
    assert captured_decoder["mask"] is source_mask


def test_target_mask_propagation():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))
    target_mask = torch.ones(2, 5)

    captured = {}

    def hook(module, args, kwargs):
        captured["mask"] = kwargs.get("self_attention_mask")

    handle = model.decoder.register_forward_pre_hook(
        hook, with_kwargs=True
    )

    model(
        source_token_ids,
        target_token_ids,
        target_mask=target_mask
    )

    handle.remove()

    assert captured["mask"] is target_mask


def test_encoder_output_reaches_decoder():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    captured = {}

    def encoder_hook(module, args, output):
        captured["encoder_output"] = output

    def decoder_hook(module, args, kwargs):
        captured["received_output"] = kwargs.get(
            "encoder_output",
            args[1] if len(args) > 1 else None
        )

    encoder_handle = model.encoder.register_forward_hook(
        encoder_hook
    )

    decoder_handle = model.decoder.register_forward_pre_hook(
        decoder_hook, with_kwargs=True
    )

    model(source_token_ids, target_token_ids)

    encoder_handle.remove()
    decoder_handle.remove()

    assert captured["received_output"] is captured["encoder_output"]


def test_gradient_flow():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    logits = model(source_token_ids, target_token_ids)

    loss = logits.mean()
    loss.backward()

    for name, parameter in model.named_parameters():

        assert parameter.grad is not None, (
            f"Missing gradient for parameter: {name}"
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (batch_size, 7))
    target_token_ids = torch.randint(0, 10000, (batch_size, 5))

    logits = model(source_token_ids, target_token_ids)

    assert logits.shape == (batch_size, 5, 10000)


@pytest.mark.parametrize(
    "source_length, target_length",
    [(3, 5), (7, 4), (10, 8)]
)
def test_different_sequence_lengths(source_length, target_length):

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, source_length))
    target_token_ids = torch.randint(0, 10000, (2, target_length))

    logits = model(source_token_ids, target_token_ids)

    assert logits.shape == (2, target_length, 10000)


def test_different_layer_counts():

    model = create_model(num_layers=2)

    assert len(model.encoder.encoder_stack.layers) == 2
    assert len(model.decoder.decoder_stack.layers) == 2


def test_train_eval_mode():

    model = create_model(dropout=0.5)

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    model.train()
    train_output = model(source_token_ids, target_token_ids)

    model.eval()
    eval_output = model(source_token_ids, target_token_ids)

    assert not torch.allclose(train_output, eval_output)


def test_output_contains_finite_values():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    logits = model(source_token_ids, target_token_ids)

    assert torch.isfinite(logits).all()


def test_parameters_registered():

    model = create_model()

    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert all(
        isinstance(parameter, nn.Parameter)
        for parameter in parameters
    )


def test_exceeding_source_max_sequence_length():

    model = create_model(max_sequence_length=5)

    source_token_ids = torch.randint(0, 10000, (2, 6))
    target_token_ids = torch.randint(0, 10000, (2, 4))

    with pytest.raises(ValueError):
        model(source_token_ids, target_token_ids)


def test_exceeding_target_max_sequence_length():

    model = create_model(max_sequence_length=5)

    source_token_ids = torch.randint(0, 10000, (2, 4))
    target_token_ids = torch.randint(0, 10000, (2, 6))

    with pytest.raises(ValueError):
        model(source_token_ids, target_token_ids)


def test_output_is_raw_logits():

    model = create_model()

    source_token_ids = torch.randint(0, 10000, (2, 7))
    target_token_ids = torch.randint(0, 10000, (2, 5))

    logits = model(source_token_ids, target_token_ids)

    assert logits.shape[-1] == 10000
    assert model.output_projection.out_features == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
