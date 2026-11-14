
import pytest
import torch

from src.model.transformer import Transformer


def create_model():
    torch.manual_seed(42)

    model = Transformer(
        vocab_size=266,
        embedding_dim=64,
        max_sequence_length=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        hidden_dim=256,
        dropout=0.0
    )

    model.eval()

    return model


def create_inputs():
    source_token_ids = torch.tensor([
        [2, 10, 20, 30, 3],
        [2, 15, 25, 35, 3]
    ])

    target_token_ids = torch.tensor([
        [2, 40, 50, 60],
        [2, 45, 55, 65]
    ])

    return source_token_ids, target_token_ids


# 1. Encoder output equivalence
def test_encoder_output_equivalence():

    model = create_model()

    source_token_ids, _ = create_inputs()

    output_1 = model.encoder(source_token_ids)
    output_2 = model.encode(source_token_ids)

    assert torch.allclose(
        output_1,
        output_2,
        atol=1e-5,
        rtol=1e-4
    )


# 2. First token logits equivalence
def test_first_token_logits():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    # Non-cached decoding
    full_output = model.decoder(
        token_ids=target_token_ids[:, :1],
        encoder_output=encoder_output
    )

    expected_logits = model.output_projection(full_output)

    # Cached decoding
    actual_logits, cache = model.decode_step(
        target_token_ids=target_token_ids[:, :1],
        encoder_output=encoder_output,
        position_offset=0
    )

    assert torch.allclose(
        expected_logits,
        actual_logits,
        atol=1e-5,
        rtol=1e-4
    )

    assert len(cache) == 2


# 3. Second token logits equivalence
def test_second_token_logits():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    # Full prefix
    full_output = model.decoder(
        token_ids=target_token_ids[:, :2],
        encoder_output=encoder_output
    )

    expected_logits = model.output_projection(
        full_output[:, -1:, :]
    )

    # First cached step
    _, cache = model.decode_step(
        target_token_ids=target_token_ids[:, :1],
        encoder_output=encoder_output,
        position_offset=0
    )

    # Second cached step
    actual_logits, cache = model.decode_step(
        target_token_ids=target_token_ids[:, 1:2],
        encoder_output=encoder_output,
        past_key_values=cache,
        position_offset=1
    )

    assert torch.allclose(
        expected_logits,
        actual_logits,
        atol=1e-5,
        rtol=1e-4
    )


# 4. Complete incremental decoding equivalence
def test_complete_sequence_equivalence():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    # Non-cached full sequence
    full_output = model.decoder(
        token_ids=target_token_ids,
        encoder_output=encoder_output
    )

    expected_logits = model.output_projection(full_output)

    # Cached incremental decoding
    past_key_values = None
    cached_logits = []

    for position in range(target_token_ids.size(1)):

        current_token = target_token_ids[:, position:position + 1]

        logits, past_key_values = model.decode_step(
            target_token_ids=current_token,
            encoder_output=encoder_output,
            past_key_values=past_key_values,
            position_offset=position
        )

        cached_logits.append(logits)

    actual_logits = torch.cat(
        cached_logits,
        dim=1
    )

    assert torch.allclose(
        expected_logits,
        actual_logits,
        atol=1e-5,
        rtol=1e-4
    )


# 5. Self-attention cache length
def test_self_attention_cache_length():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    past_key_values = None

    for position in range(3):

        current_token = target_token_ids[:, position:position + 1]

        _, past_key_values = model.decode_step(
            target_token_ids=current_token,
            encoder_output=encoder_output,
            past_key_values=past_key_values,
            position_offset=position
        )

        for layer_cache in past_key_values:

            key, value = layer_cache["self_attention"]

            assert key.size(2) == position + 1
            assert value.size(2) == position + 1


# 6. Cross-attention cache length remains constant
def test_cross_attention_cache_length():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    encoder_sequence_length = encoder_output.size(1)

    past_key_values = None

    for position in range(3):

        current_token = target_token_ids[:, position:position + 1]

        _, past_key_values = model.decode_step(
            target_token_ids=current_token,
            encoder_output=encoder_output,
            past_key_values=past_key_values,
            position_offset=position
        )

        for layer_cache in past_key_values:

            key, value = layer_cache["cross_attention"]

            assert key.size(2) == encoder_sequence_length
            assert value.size(2) == encoder_sequence_length


# 7. Cache shapes
def test_cache_shapes():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    _, cache = model.decode_step(
        target_token_ids=target_token_ids[:, :1],
        encoder_output=encoder_output,
        position_offset=0
    )

    assert len(cache) == 2

    for layer_cache in cache:

        self_key, self_value = layer_cache["self_attention"]

        cross_key, cross_value = layer_cache["cross_attention"]

        assert self_key.shape == (2, 4, 1, 16)
        assert self_value.shape == (2, 4, 1, 16)

        assert cross_key.shape == (2, 4, 5, 16)
        assert cross_value.shape == (2, 4, 5, 16)


# 8. Greedy prediction equivalence
def test_greedy_prediction_equivalence():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    full_output = model.decoder(
        token_ids=target_token_ids,
        encoder_output=encoder_output
    )

    expected_logits = model.output_projection(full_output)

    expected_tokens = expected_logits.argmax(dim=-1)

    past_key_values = None
    cached_logits = []

    for position in range(target_token_ids.size(1)):

        current_token = target_token_ids[:, position:position + 1]

        logits, past_key_values = model.decode_step(
            target_token_ids=current_token,
            encoder_output=encoder_output,
            past_key_values=past_key_values,
            position_offset=position
        )

        cached_logits.append(logits)

    actual_logits = torch.cat(cached_logits, dim=1)

    actual_tokens = actual_logits.argmax(dim=-1)

    assert torch.equal(
        expected_tokens,
        actual_tokens
    )


# 9. Output remains finite
def test_cached_output_is_finite():

    model = create_model()

    source_token_ids, target_token_ids = create_inputs()

    encoder_output = model.encode(source_token_ids)

    past_key_values = None

    for position in range(target_token_ids.size(1)):

        current_token = target_token_ids[:, position:position + 1]

        logits, past_key_values = model.decode_step(
            target_token_ids=current_token,
            encoder_output=encoder_output,
            past_key_values=past_key_values,
            position_offset=position
        )

        assert torch.isfinite(logits).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
