
import pickle
from pathlib import Path

import pytest
import torch

from src.model.transformer import Transformer
from src.inference.generate import generate_text


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "tokenizer.pkl"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "tinystories_smoke"
    / "best.pt"
)


# ==========================================
# MODEL CONFIGURATION
# Must match run_training.py
# ==========================================

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 64
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
NUM_HEADS = 4
HIDDEN_DIM = 256
DROPOUT = 0.1


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def tokenizer():

    if not TOKENIZER_PATH.exists():
        pytest.skip(
            "Tokenizer file not found. Run training first."
        )

    if TOKENIZER_PATH.stat().st_size == 0:
        pytest.fail("Tokenizer file is empty.")

    with open(TOKENIZER_PATH, "rb") as file:
        loaded_tokenizer = pickle.load(file)

    assert loaded_tokenizer.token_to_id
    assert loaded_tokenizer.id_to_token

    return loaded_tokenizer


@pytest.fixture(scope="module")
def model(tokenizer):

    if not CHECKPOINT_PATH.exists():
        pytest.skip(
            "Checkpoint not found. Run training first."
        )

    vocab_size = len(tokenizer.token_to_id)

    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            "Model state dictionary not found in checkpoint."
        )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ==========================================
# 1. TOKENIZER LOADING
# ==========================================

def test_tokenizer_loading(tokenizer):

    assert tokenizer is not None

    assert len(tokenizer.token_to_id) > 0
    assert len(tokenizer.id_to_token) > 0

    assert tokenizer.pad_token_id is not None
    assert tokenizer.unk_token_id is not None
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None


# ==========================================
# 2. CHECKPOINT LOADING
# ==========================================

def test_checkpoint_loading(model):

    assert model is not None

    assert isinstance(model, torch.nn.Module)

    assert not model.training


# ==========================================
# 3. VOCABULARY CONSISTENCY
# ==========================================

def test_vocabulary_consistency(model, tokenizer):

    model_vocab_size = model.output_projection.out_features

    tokenizer_vocab_size = len(tokenizer.token_to_id)

    assert model_vocab_size == tokenizer_vocab_size


# ==========================================
# 4. END-TO-END TEXT GENERATION
# ==========================================

def test_end_to_end_generation(model, tokenizer):

    input_text = "Once upon a time"

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=10,
        device="cpu"
    )

    assert isinstance(generated_text, str)


# ==========================================
# 5. MAXIMUM TOKEN LIMIT
# ==========================================

def test_maximum_token_limit(model, tokenizer):

    max_new_tokens = 5

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Once upon a time",
        max_new_tokens=max_new_tokens,
        device="cpu"
    )

    generated_ids = tokenizer.encode_to_ids(
        generated_text,
        add_special_tokens=False
    )

    assert len(generated_ids) <= max_new_tokens


# ==========================================
# 6. REPEATED INFERENCE
# ==========================================

def test_repeated_inference(model, tokenizer):

    input_text = "Once upon a time"

    output_1 = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=10,
        device="cpu"
    )

    output_2 = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        max_new_tokens=10,
        device="cpu"
    )

    assert output_1 == output_2


# ==========================================
# 7. CPU INFERENCE
# ==========================================

def test_cpu_inference(model, tokenizer):

    model = model.to("cpu")

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="The little boy",
        max_new_tokens=5,
        device="cpu"
    )

    assert isinstance(output, str)

    assert next(model.parameters()).device.type == "cpu"


# ==========================================
# 8. GENERATED TEXT SPECIAL TOKENS
# ==========================================

def test_generated_text_special_tokens(model, tokenizer):

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Once upon a time",
        max_new_tokens=10,
        device="cpu"
    )

    assert tokenizer.bos_token not in output
    assert tokenizer.eos_token not in output
    assert tokenizer.pad_token not in output


# ==========================================
# 9. MODEL REMAINS IN EVALUATION MODE
# ==========================================

def test_model_evaluation_mode(model, tokenizer):

    model.eval()

    generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Once upon a time",
        max_new_tokens=5,
        device="cpu"
    )

    assert model.training is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
