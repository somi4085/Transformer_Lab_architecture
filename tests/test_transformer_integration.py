
import pytest
import torch

from src.model.transformer import Transformer
from src.layers.encoder import TransformerEncoder
from src.layers.decoder import TransformerDecoder
from src.layers.encoder_stack import EncoderStack
from src.layers.decoder_stack import DecoderStack
from src.attention.multi_head_attention import MultiHeadAttention
from src.embeddings.input_embedding import InputEmbeddings

def create_transformer():
    return Transformer(
        vocab_size=1000,
        embedding_dim=256,
        max_sequence_length=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1
    )


def create_inputs():
    source_ids = torch.randint(0, 1000, (2, 7))
    target_ids = torch.randint(0, 1000, (2, 5))

    return source_ids, target_ids


# 1. Complete Transformer forward pass
def test_complete_transformer_forward_pass():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    output = model(source_ids, target_ids)

    assert output.shape == (2, 5, 1000)
    assert torch.isfinite(output).all()


# 2. Encoder integration
def test_encoder_integration():
    model = create_transformer()
    source_ids, _ = create_inputs()

    encoder_output = model.encoder(
        source_ids,
        mask=None
    )

    assert encoder_output.shape == (2, 7, 256)
    assert torch.isfinite(encoder_output).all()


# 3. Decoder integration
def test_decoder_integration():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    encoder_output = model.encoder(source_ids)

    decoder_output = model.decoder(
        target_ids,
        encoder_output
    )

    assert decoder_output.shape == (2, 5, 256)
    assert torch.isfinite(decoder_output).all()


# 4. Vocabulary projection integration
def test_vocabulary_projection_integration():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    encoder_output = model.encoder(source_ids)
    decoder_output = model.decoder(
        target_ids,
        encoder_output
    )

    logits = model.output_projection(decoder_output)

    assert logits.shape == (2, 5, 1000)


# 5. Source mask integration
def test_source_mask_integration():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    source_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])

    output = model(
        source_ids,
        target_ids,
        source_mask=source_mask
    )

    assert output.shape == (2, 5, 1000)
    assert torch.isfinite(output).all()


# 6. Target mask integration
def test_target_mask_integration():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    target_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0]
    ])

    output = model(
        source_ids,
        target_ids,
        target_mask=target_mask
    )

    assert output.shape == (2, 5, 1000)
    assert torch.isfinite(output).all()


# 7. Different source and target sequence lengths
@pytest.mark.parametrize(
    "source_length,target_length",
    [
        (3, 5),
        (7, 4),
        (10, 8),
        (15, 3)
    ]
)
def test_different_sequence_lengths(
    source_length,
    target_length
):
    model = create_transformer()

    source_ids = torch.randint(
        0, 1000, (2, source_length)
    )
    target_ids = torch.randint(
        0, 1000, (2, target_length)
    )

    output = model(source_ids, target_ids)

    assert output.shape == (
        2,
        target_length,
        1000
    )


# 8. Gradient flow through complete architecture
def test_complete_gradient_flow():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    output = model(source_ids, target_ids)

    loss = output.mean()
    loss.backward()

    assert model.encoder.input_embedding.token_embedding.embedding.weight.grad is not None
    assert model.decoder.input_embeddings.token_embedding.embedding.weight.grad is not None   
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None


# 9. Encoder and decoder parameters are independent
def test_encoder_decoder_parameters_independent():
    model = create_transformer()

    encoder_parameters = {
        id(parameter)
        for parameter in model.encoder.parameters()
    }

    decoder_parameters = {
        id(parameter)
        for parameter in model.decoder.parameters()
    }

    assert encoder_parameters.isdisjoint(
        decoder_parameters
    )


# 10. Training and evaluation mode integration
def test_train_eval_mode_integration():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    model.train()
    train_output = model(source_ids, target_ids)

    model.eval()

    with torch.no_grad():
        eval_output = model(source_ids, target_ids)

    assert train_output.shape == eval_output.shape
    assert torch.isfinite(train_output).all()
    assert torch.isfinite(eval_output).all()


# 11. Batch size integration
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4]
)
def test_batch_size_integration(batch_size):
    model = create_transformer()

    source_ids = torch.randint(
        0, 1000, (batch_size, 7)
    )
    target_ids = torch.randint(
        0, 1000, (batch_size, 5)
    )

    output = model(source_ids, target_ids)

    assert output.shape == (
        batch_size,
        5,
        1000
    )


# 12. Model parameters receive finite gradients
def test_gradients_are_finite():
    model = create_transformer()

    source_ids, target_ids = create_inputs()

    output = model(source_ids, target_ids)
    loss = output.square().mean()
    loss.backward()

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
