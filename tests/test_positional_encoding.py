
import pytest
import torch

from src.embeddings.positional_encoding import PositionalEncoding


@pytest.fixture
def positional_encoding():
    return PositionalEncoding(
        embedding_dim=4,
        max_sequence_length=10
    )


# 1. Output shape
def test_output_shape(positional_encoding):
    token_embeddings = torch.randn(2, 5, 4)

    output = positional_encoding(token_embeddings)

    assert output.shape == (2, 5, 4)


# 2. PE matrix shape
def test_pe_matrix_shape(positional_encoding):
    assert positional_encoding.pe.shape == (10, 4)


# 3. Sinusoidal values
def test_sinusoidal_values(positional_encoding):
    pe = positional_encoding.pe

    assert torch.allclose(pe[1, 0], torch.tensor(0.84147096), atol=1e-5)
    assert torch.allclose(pe[1, 1], torch.tensor(0.5403023), atol=1e-5)
    assert torch.allclose(pe[1, 2], torch.tensor(0.01), atol=1e-5)
    assert torch.allclose(pe[1, 3], torch.tensor(0.99995), atol=1e-5)


# 4. Position zero values
def test_position_zero_values(positional_encoding):
    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])

    assert torch.allclose(positional_encoding.pe[0], expected)


# 5. Embedding addition
def test_embedding_addition(positional_encoding):
    token_embeddings = torch.ones(2, 5, 4)

    original_embeddings = token_embeddings.clone()

    output = positional_encoding(token_embeddings)

    expected = original_embeddings + positional_encoding.pe[:5, :]

    print("\nActual Output:")
    print(output)

    print("\nExpected Output:")
    print(expected)

    print("\nMaximum Difference:")
    print(torch.max(torch.abs(output - expected)))

    print("\nInput Changed:", not torch.equal(
        token_embeddings, original_embeddings
    ))

    assert torch.allclose(output, expected)


# 6. Batch consistency
def test_batch_consistency(positional_encoding):
    token_embeddings = torch.zeros(2, 5, 4)

    output = positional_encoding(token_embeddings)

    assert torch.allclose(output[0], output[1])


# 7. Buffer registration
def test_buffer_registration(positional_encoding):
    buffers = dict(positional_encoding.named_buffers())

    assert "pe" in buffers


# 8. PE is not a trainable parameter
def test_pe_not_trainable_parameter(positional_encoding):
    parameters = dict(positional_encoding.named_parameters())

    assert "pe" not in parameters


# 9. Maximum sequence length
def test_max_sequence_length(positional_encoding):
    token_embeddings = torch.randn(2, 10, 4)

    print("PE matrix shape:", positional_encoding.pe.shape)
    print("Token embeddings shape:", token_embeddings.shape)

    output = positional_encoding(token_embeddings)

    assert output.shape == (2, 10, 4)


# 10. Sequence length exceeds maximum
def test_sequence_length_exceeds_max(positional_encoding):
    token_embeddings = torch.randn(2, 11, 4)

    with pytest.raises(ValueError):
        positional_encoding(token_embeddings)


# odds
def test_odd_embedding_dimension():
    positional_encoding = PositionalEncoding(
        embedding_dim=5,
        max_sequence_length=10
    )

    assert positional_encoding.pe.shape == (10, 5)


def test_odd_embedding_dimension_forward():
    positional_encoding = PositionalEncoding(
        embedding_dim=5,
        max_sequence_length=10
    )

    token_embeddings = torch.randn(2, 5, 5)

    output = positional_encoding(token_embeddings)

    assert output.shape == (2, 5, 5)



def test_odd_embedding_dimension_values():
    positional_encoding = PositionalEncoding(
        embedding_dim=5,
        max_sequence_length=10
    )

    pe = positional_encoding.pe

    expected = torch.tensor([
        0.84147096,
        0.5403023,
        0.02511622,
        0.99968454,
        0.00063096
    ])

    assert torch.allclose(pe[1], expected, atol=1e-5)


@pytest.mark.parametrize("embedding_dim", [5, 7, 9])
def test_multiple_odd_embedding_dimensions(embedding_dim):
    positional_encoding = PositionalEncoding(
        embedding_dim=embedding_dim,
        max_sequence_length=10
    )

    assert positional_encoding.pe.shape == (10, embedding_dim)




def test_zero_embedding_dim():
    with pytest.raises(ValueError):
        PositionalEncoding(
            embedding_dim=0,
            max_sequence_length=10
        )


def test_negative_embedding_dim():
    with pytest.raises(ValueError):
        PositionalEncoding(
            embedding_dim=-5,
            max_sequence_length=10
        )


def test_zero_max_sequence_length():
    with pytest.raises(ValueError):
        PositionalEncoding(
            embedding_dim=4,
            max_sequence_length=0
        )


def test_negative_max_sequence_length():
    with pytest.raises(ValueError):
        PositionalEncoding(
            embedding_dim=4,
            max_sequence_length=-10
        )


def test_invalid_input_dimensions(positional_encoding):
    token_embeddings = torch.randn(5, 4)

    with pytest.raises(ValueError):
        positional_encoding(token_embeddings)


def test_invalid_input_type(positional_encoding):
    with pytest.raises(TypeError):
        positional_encoding("hello")


def test_embedding_dimension_mismatch(positional_encoding):
    token_embeddings = torch.randn(2, 5, 5)

    with pytest.raises(ValueError):
        positional_encoding(token_embeddings)




# TASK -1 CPU compatibility

def test_cpu_device_compatibility(positional_encoding):
    positional_encoding = positional_encoding.to("cpu")

    token_embedding = torch.rand(2,5,4, device="cpu")

    output = positional_encoding(token_embedding)

    assert output.device.type == "cpu"

def test_positional_encoding_buffer_device(positional_encoding):
    positional_encoding = positional_encoding.to("cpu")

    assert positional_encoding.pe.device.type == "cpu"

def test_cuda_device_compatibility():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not avaiable")

    positional_encoding  = PositionalEncoding(
        embedding_dim=4,
        max_sequence_length=10
    ).to("cuda")

    token_embedding = torch.rand(
        2,5,4,
        device = "cuda"
        
    )

    output = positional_encoding(token_embedding)

    assert output.device.type == "cuda"


def test_float32_compatibility(positional_encoding):
    token_embedding = torch.rand(
        (2,5,4),
        dtype = torch.float32
    )

    output = positional_encoding(token_embedding)

    assert output.dtype == torch.float32

def test_float16_compatibility(positional_encoding):
    token_embedding = torch.randn(
        (2,5,4),
        dtype = torch.float16
    )
    output = positional_encoding(token_embedding)

    assert output.dtype == torch.float16

def test_bfloat16_compatibility(positional_encoding):
    token_embedding = torch.rand(
        (2, 5, 4),
        dtype=torch.bfloat16
    )

    output = positional_encoding(token_embedding)

    assert output.dtype == torch.bfloat16

    