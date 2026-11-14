
import torch
import pytest
from src.embeddings.token_embedding import TokenEmbedding


def test_token_embedding():

    # 1. Define vocabulary and embedding dimensions
    vocab_size = 10000
    embedding_dim = 256

    # 2. Create embedding layer
    embedding_layer = TokenEmbedding(
        vocab_size,
        embedding_dim
    )

    # 3. Create SGD optimizer
    optimizer = torch.optim.SGD(
        embedding_layer.parameters(),
        lr=0.01
    )

    # 4. Input token IDs
    token_ids = torch.tensor([10, 25, 10])

    # 5. Save original weights before training
    original_weights_10 = (
        embedding_layer.embedding.weight[10].detach().clone()
    )

    original_weights_25 = (
        embedding_layer.embedding.weight[25].detach().clone()
    )

    original_weights_30 = (
        embedding_layer.embedding.weight[30].detach().clone()
    )

    # 6. Forward pass
    output = embedding_layer(token_ids)

    # 7. Calculate loss
    loss = output.sum()

    # 8. Clear previous gradients
    optimizer.zero_grad()

    # 9. Backpropagation
    loss.backward()

    # 10. Inspect gradients
    gradients = embedding_layer.embedding.weight.grad

    print("Output shape:", output.shape)
    print("Loss:", loss.item())
    print("Gradient shape:", gradients.shape)

    print("Token 10 gradient:", gradients[10][0].item())
    print("Token 25 gradient:", gradients[25][0].item())
    print("Token 30 gradient:", gradients[30][0].item())

    # 11. Update embedding weights
    optimizer.step()

    # 12. Capture updated weights
    updated_weights_10 = (
        embedding_layer.embedding.weight[10].detach().clone()
    )

    updated_weights_25 = (
        embedding_layer.embedding.weight[25].detach().clone()
    )

    updated_weights_30 = (
        embedding_layer.embedding.weight[30].detach().clone()
    )

    # 13. Compare original and updated weights
    print("\nToken 10 weights changed:",
          not torch.equal(original_weights_10, updated_weights_10))

    print("Token 25 weights changed:",
          not torch.equal(original_weights_25, updated_weights_25))

    print("Token 30 weights changed:",
          not torch.equal(original_weights_30, updated_weights_30))

    # 14. Verify expected behavior
    assert not torch.equal(original_weights_10, updated_weights_10)
    assert not torch.equal(original_weights_25, updated_weights_25)
    assert torch.equal(original_weights_30, updated_weights_30)



# test output shape

def test_out_shape():
    embedding_layer = TokenEmbedding(1000,256)

    token_ids = torch.randint(0,1000,(2,5))

    output = embedding_layer(token_ids)

    assert output.shape == (2,5,256)

# test single token
def test_single_token():
    embedding_layer = TokenEmbedding(1000,256)
    token_ids = torch.tensor([10])
    output = embedding_layer(token_ids)

    assert output.shape == (1,256)

# test repeated token
def test_repeated_tokens():
    embedding_layer = TokenEmbedding(1000,256)
    token_ids = torch.tensor([10,25,10])
    output = embedding_layer(token_ids)

    assert torch.equal(output[0],output[2])


# test batch input

def test_batch_input():
    embedding_layer = TokenEmbedding(1000,256)
    token_ids = torch.randint(0,1000,(3,5))
    output = embedding_layer(token_ids)

    assert output.shape == (3,5,256)

# test negative token id
def test_negative_id():
    embedding_layer = TokenEmbedding(1000,256)
    token_id = torch.tensor([-1,10,20])
    with pytest.raises((IndexError, RuntimeError)):
        embedding_layer(token_id)

# test out-of-range

def test_out_of_range():
    embedding_layer = TokenEmbedding(1000,256)
    token_id = torch.tensor([10,1000,20])
    with pytest.raises((IndexError, RuntimeError)):
        embedding_layer(token_id)

# test empty sequence

def test_empty_sequence():
    embedding_layer = TokenEmbedding(1000,256)
    token_id = torch.empty((2,0), dtype=torch.long)
    output = embedding_layer(token_id)
    assert output.shape == (2,0,256)

# Test invalid float point
def test_invalid_float_token_ids():
    embedding_layer = TokenEmbedding(1000,256)
    token_id = torch.tensor([10.0,20.0,30.0])
    with pytest.raises((RuntimeError,TypeError)):
        embedding_layer(token_id)
        

if __name__ == "__main__":
    test_token_embedding()



