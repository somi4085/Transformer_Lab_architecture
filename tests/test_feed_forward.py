
import pytest
import torch

from src.layers.feed_forward import FeedForwardNetwork


def create_feed_forward(
    embedding_dim=256,
    hidden_dim=1024,
    dropout=0.0
):
    return FeedForwardNetwork(
        embedding_dim=embedding_dim,
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
    model = create_feed_forward()

    x = create_input()

    output = model(x)

    assert output.shape == (2, 5, 256)


# 2. First linear layer dimensions
def test_first_linear_layer_dimensions():
    model = create_feed_forward()

    assert model.linear1.in_features == 256
    assert model.linear1.out_features == 1024


# 3. ReLU activation
def test_relu_activation():
    model = create_feed_forward()

    x = torch.tensor([
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    ])

    output = model.activation(x)

    expected = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 2.0]
    ])

    assert torch.equal(output, expected)


# 4. Second linear layer dimensions
def test_second_linear_layer_dimensions():
    model = create_feed_forward()

    assert model.linear2.in_features == 1024
    assert model.linear2.out_features == 256


# 5. Dropout in training mode
def test_dropout_training_mode():
    model = create_feed_forward(dropout=0.5)

    x = torch.ones(100, 100)

    model.train()

    output_1 = model.dropout(x)
    output_2 = model.dropout(x)

    assert not torch.equal(output_1, output_2)


# 6. Dropout in evaluation mode
def test_dropout_evaluation_mode():
    model = create_feed_forward(dropout=0.5)

    x = torch.randn(2, 5, 256)

    model.eval()

    output_1 = model(x)
    output_2 = model(x)

    assert torch.equal(output_1, output_2)


# 7. Gradient flow
def test_gradient_flow():
    model = create_feed_forward()

    x = create_input()
    x.requires_grad_(True)

    output = model(x)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad).item() > 0

    for layer in [model.linear1, model.linear2]:

        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

        assert torch.count_nonzero(
            layer.weight.grad
        ).item() > 0

        assert torch.count_nonzero(
            layer.bias.grad
        ).item() > 0


# 8. Different batch sizes
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_different_batch_sizes(batch_size):
    model = create_feed_forward()

    x = create_input(batch_size=batch_size)

    output = model(x)

    assert output.shape == (batch_size, 5, 256)


# 9. Different sequence lengths
@pytest.mark.parametrize("sequence_length", [1, 3, 5, 10])
def test_different_sequence_lengths(sequence_length):
    model = create_feed_forward()

    x = create_input(sequence_length=sequence_length)

    output = model(x)

    assert output.shape == (2, sequence_length, 256)


# 10. Invalid embedding dimension
def test_invalid_embedding_dimension():
    model = create_feed_forward()

    x = torch.randn(2, 5, 128)

    with pytest.raises(RuntimeError):
        model(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

