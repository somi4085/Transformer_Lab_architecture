
import torch
import pytest
from src.attention.multi_head_attention import MultiHeadAttention


def test_multi_head_attention():

    # 1. Create MHA model
    model = MultiHeadAttention(
        embedding_dim=256,
        num_heads=4,
        dropout=0.1,
        bias=True
    )

    # 2. Create input embeddings
    x = torch.randn(2, 5, 256)

    # 3. Pass input through MHA
    output = model(x)

    # 4. Print tensor information
    print("Output type:", type(output))
    print("Input shape:", x.shape)
    print("Attention output shape:", output.shape)

    # 5. Verify output
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 5, 256)



def test_dropout_train_eval_mode():
    model = MultiHeadAttention(
        embedding_dim=256,
        num_heads=4,
        dropout=0.5,
        bias=True
    )

    # create fixed input
    torch.manual_seed(42)
    x = torch.randn(2,5,256)

    # Training mode: Dropout active
    model.train()

    output_train1 = model(x)
    output_train2 = model(x)

        # Outputs should generally be different
    assert not torch.allclose(
        output_train1,
        output_train2
    )

    # 4. Evaluation mode: Dropout inactive
    model.eval()

    with torch.no_grad():
        output_eval1 = model(x)
        output_eval2 = model(x)


    assert torch.allclose(
        output_eval1,
        output_eval2
    )

    assert output_eval1.shape == (2,5,256)


def test_multi_head_attention_padding_mask():
    model = MultiHeadAttention(
        embedding_dim=256,
        num_heads=4,
        dropout=0.1,
        bias=True
    )
    model.eval()

    # create embeddings mask
    x = torch.randn(2,5,256)

    # create padding mask
    mask = torch.tensor([
        [1,1,1,0,0],
        [1,1,1,1,0]
    ])

    # apply mha with padding mask
    with torch.no_grad():
        output = model(x,mask=mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2,5,256)
    assert not torch.isnan(output).any()

    print("Padding mask test passed")
    print("Output shape: ",output.shape)


def test_multi_head_attention_causal_mask():
    model = MultiHeadAttention(
        embedding_dim=256,
        num_heads=4,
        dropout=0.1,
        bias=True
    )

    model.eval()

    x = torch.randn(2,5,256)

    with torch.no_grad():
        output = model(x,causal=True)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2,5,256)
    assert not torch.isnan(output).any()
    print("Causal mask test passed!")
    print("Output shape:", output.shape)


import pytest
import torch

from src.attention.multi_head_attention import MultiHeadAttention


def create_mha(
    embedding_dim=256,
    num_heads=8,
    dropout=0.0
):
    return MultiHeadAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=True
    )


def test_output_shape():
    attention = create_mha()

    x = torch.randn(2, 5, 256)

    output = attention(x)

    assert output.shape == (2, 5, 256)


def test_head_dimension():
    attention = create_mha()

    assert attention.num_heads == 8
    assert attention.embedding_dim == 256
    assert attention.head_dim == 32


def test_qkv_projections():
    attention = create_mha()

    x = torch.randn(2, 5, 256)

    query = attention.query_projection(x)
    key = attention.key_projection(x)
    value = attention.value_projection(x)

    assert query.shape == (2, 5, 256)
    assert key.shape == (2, 5, 256)
    assert value.shape == (2, 5, 256)


def test_attention_heads_shape():
    attention = create_mha()

    x = torch.randn(2, 5, 256)

    batch_size, sequence_length, _ = x.shape

    query = attention.query_projection(x)

    query = query.view(
        batch_size,
        sequence_length,
        attention.num_heads,
        attention.head_dim
    )

    assert query.shape == (2, 5, 8, 32)

    query = query.transpose(1, 2)

    assert query.shape == (2, 8, 5, 32)


def test_output_projection():
    attention = create_mha()

    x = torch.randn(2, 5, 256)

    output = attention(x)

    assert output.shape == (2, 5, 256)
    assert attention.output_projection.weight.shape == (256, 256)


def test_causal_masking():
    attention = create_mha()

    x = torch.randn(2, 5, 256)

    output = attention(x, causal=True)

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


def test_padding_masking():
    attention = create_mha()

    x = torch.randn(2, 5, 256)

    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    output = attention(x, mask=mask)

    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()


def test_gradient_flow():
    attention = create_mha()

    x = torch.randn(
        2, 5, 256,
        requires_grad=True
    )

    output = attention(x)

    loss = output.sum()

    loss.backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad).item() > 0

    projection_layers = [
        attention.query_projection,
        attention.key_projection,
        attention.value_projection,
        attention.output_projection
    ]

    for layer in projection_layers:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(
            layer.weight.grad
        ).item() > 0


def test_dropout_behavior():
    attention = create_mha(dropout=0.5)

    x = torch.randn(2, 5, 256)

    attention.train()

    train_output_1 = attention(x)
    train_output_2 = attention(x)

    assert not torch.equal(train_output_1, train_output_2)

    attention.eval()

    eval_output_1 = attention(x)
    eval_output_2 = attention(x)

    assert torch.equal(eval_output_1, eval_output_2)


def test_invalid_head_configuration():
    with pytest.raises(ValueError):
        create_mha(
            embedding_dim=250,
            num_heads=8
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
