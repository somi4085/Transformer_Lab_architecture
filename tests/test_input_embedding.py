
import math

import torch

from src.embeddings.input_embedding import InputEmbeddings


def create_input_embedding(dropout=0.1):
    return InputEmbeddings(
        vocab_size=1000,
        embedding_dim=256,
        max_sequence_length=100,
        dropout=dropout
    )


def test_input_embedding_output_shape():
    model = create_input_embedding()

    token_ids = torch.randint(0, 1000, (2, 5))

    output = model(token_ids)

    assert output.shape == (2, 5, 256)


def test_embedding_scaling():
    model = create_input_embedding(dropout=0.0)
    model.eval()

    token_ids = torch.tensor([[10, 20, 30]])

    token_embeddings = model.token_embedding(token_ids)

    scaled_embeddings = (
        token_embeddings * math.sqrt(model.embedding_dim)
    )

    expected_output = model.positional_encoding(scaled_embeddings)

    actual_output = model(token_ids)

    assert torch.allclose(actual_output, expected_output)


def test_positional_encoding_addition():
    model = create_input_embedding(dropout=0.0)
    model.eval()

    token_ids = torch.tensor([[10, 20, 30]])

    token_embeddings = model.token_embedding(token_ids)

    scaled_embeddings = (
        token_embeddings * math.sqrt(model.embedding_dim)
    )

    positional_output = model.positional_encoding(scaled_embeddings)

    actual_output = model(token_ids)

    assert torch.allclose(actual_output, positional_output)


def test_dropout_behavior():
    model = create_input_embedding(dropout=0.5)

    token_ids = torch.randint(0, 1000, (2, 10))

    # Training mode: dropout active
    model.train()

    train_output_1 = model(token_ids)
    train_output_2 = model(token_ids)

    assert not torch.equal(train_output_1, train_output_2)

    # Evaluation mode: dropout inactive
    model.eval()

    eval_output_1 = model(token_ids)
    eval_output_2 = model(token_ids)

    assert torch.equal(eval_output_1, eval_output_2)


def test_gradient_flow():
    model = create_input_embedding(dropout=0.0)

    token_ids = torch.tensor([[10, 20, 10]])

    output = model(token_ids)

    loss = output.sum()

    loss.backward()

    gradients = model.token_embedding.embedding.weight.grad

    assert gradients is not None
    assert torch.count_nonzero(gradients).item() > 0


def test_device_and_dtype_compatibility():
    model = create_input_embedding(dropout=0.0)

    token_ids = torch.randint(0, 1000, (2, 5))

    output = model(token_ids)

    assert output.device.type == "cpu"
    assert output.dtype == torch.float32

    # Test float64 compatibility
    model = model.to(dtype=torch.float64)

    output_float64 = model(token_ids)

    assert output_float64.dtype == torch.float64
    assert output_float64.shape == (2, 5, 256)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
