
import math

import torch

from src.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention
)


def create_attention(dropout=0.0):
    return ScaledDotProductAttention(dropout=dropout)


def test_output_shape():
    attention = create_attention()

    query = torch.randn(2, 4, 5, 64)
    key = torch.randn(2, 4, 5, 64)
    value = torch.randn(2, 4, 5, 64)

    output = attention(query, key, value)

    assert output.shape == (2, 4, 5, 64)


def test_scaling_factor():
    attention = create_attention()

    query = torch.ones(1, 1, 2, 64)
    key = torch.ones(1, 1, 2, 64)
    value = torch.tensor(
        [[[[1.0] * 64, [3.0] * 64]]]
    )

    output = attention(query, key, value)

    # Each score is 64 / sqrt(64) = 8.
    # Equal scores produce equal attention weights.
    expected = torch.full((1, 1, 2, 64), 2.0)

    assert torch.allclose(output, expected, atol=1e-5)


def test_padding_mask():
    attention = create_attention()

    query = torch.ones(1, 1, 1, 2)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 4.0], [100.0, 200.0]]]])

    mask = torch.tensor([[1, 0]])

    output = attention(query, key, value, mask=mask)

    expected = torch.tensor([[[[2.0, 4.0]]]])

    assert torch.allclose(output, expected, atol=1e-5)


def test_causal_mask():
    attention = create_attention()

    query = torch.ones(1, 1, 3, 2)
    key = torch.ones(1, 1, 3, 2)

    value = torch.tensor(
        [[[
            [1.0, 1.0],
            [2.0, 2.0],
            [100.0, 100.0]
        ]]]
    )

    output = attention(
        query,
        key,
        value,
        causal=True
    )

    expected = torch.tensor(
        [[[
            [1.0, 1.0],
            [1.5, 1.5],
            [103.0 / 3.0, 103.0 / 3.0]
        ]]]
    )

    assert torch.allclose(output, expected, atol=1e-5)


def test_combined_masks():
    attention = create_attention()

    query = torch.ones(1, 1, 3, 2)
    key = torch.ones(1, 1, 3, 2)

    value = torch.tensor(
        [[[
            [2.0, 2.0],
            [100.0, 100.0],
            [200.0, 200.0]
        ]]]
    )

    # First and third positions are valid.
    # Middle position is padding.
    mask = torch.tensor([[1, 0, 1]])

    output = attention(
        query,
        key,
        value,
        mask=mask,
        causal=True
    )

    expected = torch.tensor(
        [[[
            [2.0, 2.0],
            [2.0, 2.0],
            [101.0, 101.0]
        ]]]
    )

    assert torch.allclose(output, expected, atol=1e-5)


def test_gradient_flow():
    attention = create_attention()

    query = torch.randn(
        2, 4, 5, 64,
        requires_grad=True
    )

    key = torch.randn(
        2, 4, 5, 64,
        requires_grad=True
    )

    value = torch.randn(
        2, 4, 5, 64,
        requires_grad=True
    )

    output = attention(query, key, value)

    loss = output.sum()

    loss.backward()

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None

    assert torch.count_nonzero(query.grad).item() > 0
    assert torch.count_nonzero(key.grad).item() > 0
    assert torch.count_nonzero(value.grad).item() > 0


def test_dropout_behavior():
    attention = create_attention(dropout=0.5)

    query = torch.randn(2, 4, 5, 64)
    key = torch.randn(2, 4, 5, 64)
    value = torch.randn(2, 4, 5, 64)

    attention.train()

    train_output_1 = attention(query, key, value)
    train_output_2 = attention(query, key, value)

    assert not torch.equal(train_output_1, train_output_2)

    attention.eval()

    eval_output_1 = attention(query, key, value)
    eval_output_2 = attention(query, key, value)

    assert torch.equal(eval_output_1, eval_output_2)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

