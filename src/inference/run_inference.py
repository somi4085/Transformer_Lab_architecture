
import pickle
from pathlib import Path

import torch

from src.model.transformer import Transformer
from src.inference.generate import generate_text


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "tokenizer.pkl"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "tinystories_smoke" / "best.pt"


# Model configuration
EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 64
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
NUM_HEADS = 4
HIDDEN_DIM = 256
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(tokenizer_path):

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}"
        )

    if tokenizer_path.stat().st_size == 0:
        raise ValueError(
            "Tokenizer file is empty. "
            "Save the trained tokenizer again."
        )

    try:
        with open(tokenizer_path, "rb") as file:
            tokenizer = pickle.load(file)

    except (EOFError, pickle.UnpicklingError) as error:
        raise ValueError(
            "Tokenizer file is corrupted or incomplete. "
            "Retrain and save the tokenizer."
        ) from error

    if not hasattr(tokenizer, "token_to_id"):
        raise ValueError(
            "Loaded tokenizer does not contain token_to_id."
        )

    print("Tokenizer loaded successfully.")
    print("Vocabulary size:", len(tokenizer.token_to_id))

    return tokenizer


def load_model(tokenizer):

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

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model


def main():

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    print("\nLoading Transformer model...")

    model = load_model(tokenizer)

    prompts = [
        "Once upon a time",
        "The little girl",
        "One beautiful morning"
    ]

    print("\nGenerating text...\n")

    for prompt in prompts:

        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                input_text=prompt,
                max_new_tokens=30,
                device=DEVICE
            )

            print("Prompt:", prompt)
            print("Generated:", generated_text)
            print("-" * 60)

        except Exception as error:
            print(f"Generation failed for '{prompt}': {error}")


if __name__ == "__main__":
    main()
