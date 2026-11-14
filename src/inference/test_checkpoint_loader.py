
from pathlib import Path

import pytest
import torch

from src.model.transformer import Transformer
from src.inference.checkpoint_loader import load_model_checkpoint


CHECKPOINT_PATH = Path(
    "checkpoints/tinystories_smoke/best.pt"
)

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 64
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
NUM_HEADS = 4
HIDDEN_DIM = 256
DROPOUT = 0.1


def get_checkpoint_state(checkpoint):

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "model_state" in checkpoint:
        return checkpoint["model_state"]

    raise KeyError(
        "Model state dictionary not found in checkpoint"
    )


def get_vocab_size(model_state_dict):

    projection_key = "output_projection.weight"

    if projection_key not in model_state_dict:
        raise KeyError(
            f"Missing checkpoint parameter: {projection_key}"
        )

    projection_weight = model_state_dict[projection_key]

    return projection_weight.shape[0]


def create_model(vocab_size):

    return Transformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )


@pytest.fixture
def loaded_model():

    assert CHECKPOINT_PATH.is_file(), (
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model_state_dict = get_checkpoint_state(checkpoint)

    vocab_size = get_vocab_size(model_state_dict)

    model = create_model(vocab_size)

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu"
    )

    return model


def test_checkpoint_file_exists():

    assert CHECKPOINT_PATH.is_file()


def test_checkpoint_loading(loaded_model):

    assert isinstance(loaded_model, Transformer)


def test_model_evaluation_mode(loaded_model):

    assert loaded_model.training is False


def test_model_device(loaded_model):

    for parameter in loaded_model.parameters():
        assert parameter.device.type == "cpu"


def test_model_parameters_are_finite(loaded_model):

    for parameter in loaded_model.parameters():
        assert torch.isfinite(parameter).all()


def test_model_forward_pass(loaded_model):

    vocab_size = loaded_model.output_projection.out_features

    source_token_ids = torch.tensor([
        [2, 10, 15, 20, 3],
        [2, 25, 30, 35, 3]
    ])

    target_token_ids = torch.tensor([
        [2, 40, 45, 50],
        [2, 55, 60, 65]
    ])

    source_mask = torch.ones(
        2, 5,
        dtype=torch.long
    )

    target_mask = torch.ones(
        2, 4,
        dtype=torch.long
    )

    assert source_token_ids.max().item() < vocab_size
    assert target_token_ids.max().item() < vocab_size

    with torch.no_grad():

        logits = loaded_model(
            source_token_ids=source_token_ids,
            target_token_ids=target_token_ids,
            source_mask=source_mask,
            target_mask=target_mask
        )

    assert logits.shape == (2, 4, vocab_size)

    assert torch.isfinite(logits).all()


if __name__ == "__main__":

    pytest.main([
        __file__,
        "-v"
    ])
