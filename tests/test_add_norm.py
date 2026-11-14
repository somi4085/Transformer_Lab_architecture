import torch
import pytest
from src.layers.add_norm import AddNorm

def test_add_norm():
    model = AddNorm(
        embedding_dim=256,
        dropout=0.1
    )

    x = torch.rand(2,5,256)
    sublayer_output = torch.randn(2,5,256)

    model.eval()
    output = model(x, sublayer_output)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2,5,256)

    print("Input shape: ", x.shape)
    print("output shape: ", sublayer_output.shape)
    print("Final output: ",output.shape)


def create_add_norm(
        embedding_dim=256,
        dropout=0.0
):
    return AddNorm(
        embedding_dim=embedding_dim,
        dropout=dropout
    )

def create_inputs(
    batch_size = 2,
    sequence_length=5,
    embedding_dim=256
):
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )
    sublayer_output = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )
    return x, sublayer_output


def test_output_shape():
    model = create_add_norm()

    x, sublayer_output = create_inputs()

    output = model(x,sublayer_output)

    assert output.shape == (2,5,256)

def test_residual_connection():
    model = create_add_norm()

    model.eval()
    x = torch.randn(2,5,256)
    sublayer_output = torch.randn(2,5,256)
    output = model(x,sublayer_output)
    expected = model.layer_norm(x+sublayer_output)
    assert torch.allclose(output,expected)

def test_layer_norm_statastic():
    model = create_add_norm(dropout=0.0)
    model.eval()

    x, sublayer_output = create_inputs()

    output = model(x,sublayer_output)
    mean = output.mean(dim=-1)
    std = output.std(dim=-1,unbiased=False)

    assert torch.allclose(mean,torch.zeros_like(mean),atol=1e-5)

    assert torch.allclose(std,torch.ones_like(std),atol=1e-5)

def test_layer_norm_dimension():
    model = create_add_norm(embedding_dim=256)

    assert model.layer_norm.normalized_shape == (256,)

def test_dropout_training_mode():
    model = create_add_norm(
        dropout=0.5
    )
    x, sublayer_output = create_inputs()

    model.train()

    output_1 = model(x,sublayer_output)
    output_2 = model(x,sublayer_output)

    assert not torch.equal(output_1,output_2)

def test_dropout_evaluation_model():
    model = create_add_norm(dropout=0.5)

    x, sublayer_output = create_inputs()

    model.eval()

    output_1 = model(x,sublayer_output)
    output_2 = model(x,sublayer_output)
    assert torch.equal(output_1,output_2)

def test_gradient_flow():
    model = create_add_norm(dropout=0.5)

    x,sublayer_output = create_inputs()
    x.requires_grad_(True)
    sublayer_output.requires_grad_(True)
    output = model(x, sublayer_output)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert sublayer_output.grad is not None

    assert torch.count_nonzero(x.grad).item() > 0
    assert torch.count_nonzero(x.grad).item() > 0

    assert model.layer_norm.weight.grad is not None
    assert model.layer_norm.weight.grad is not None

@pytest.mark.parametrize("sequence_length", [1, 5, 10])
def test_different_sequence_lengths(sequence_length):
    model = create_add_norm()

    x,sublayer_output = create_inputs(
        sequence_length=sequence_length
    )
    output = model(x,sublayer_output)

    assert output.shape == (2,sequence_length,256)

def test_invalid_dimension():
    model = create_add_norm()

    x = torch.randn(2,5,256)
    sublayer_output = torch.randn(2,5,128)

    with pytest.raises(RuntimeError):
        model(x,sublayer_output)



if __name__ == "__main__":
    test_add_norm()