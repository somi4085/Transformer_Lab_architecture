
import pickle
from pathlib import Path

import torch

from src.data.prepare_dataset import prepare_dataset
from src.data.trainer import Trainer
from src.data.fit import fit
from src.model.transformer import Transformer


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TOKENIZER_PATH = ARTIFACT_DIR / "tokenizer.pkl"

CHECKPOINT_DIR = (
    PROJECT_ROOT / "checkpoints" / "tinystories_smoke"
)


# ==========================================
# TOKENIZER PERSISTENCE
# ==========================================

def save_tokenizer(tokenizer, tokenizer_path):
    tokenizer_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = tokenizer_path.with_suffix(".tmp")

    try:
        with open(temporary_path, "wb") as file:
            pickle.dump(
                tokenizer,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        if not temporary_path.exists():
            raise RuntimeError(
                "Temporary tokenizer file was not created."
            )

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(
                "Temporary tokenizer file is empty."
            )

        # Verify that the saved file can be loaded.
        with open(temporary_path, "rb") as file:
            loaded_tokenizer = pickle.load(file)

        if not loaded_tokenizer.token_to_id:
            raise ValueError(
                "Saved tokenizer has an empty token_to_id mapping."
            )

        # Replace the existing file atomically.
        temporary_path.replace(tokenizer_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print("\n--- TOKENIZER PERSISTENCE ---")
    print("Tokenizer path:", tokenizer_path)
    print("Tokenizer file size:", tokenizer_path.stat().st_size)
    print("Vocabulary size:", len(tokenizer.token_to_id))
    print("Tokenizer successfully saved and verified.")


# ==========================================
# TRAINING PIPELINE
# ==========================================

def run_training():

    torch.manual_seed(42)

    # --------------------------------------
    # 1. Training configuration
    # --------------------------------------

    train_samples = 100
    validation_samples = 20

    num_merges = 200
    max_sequence_length = 64
    batch_size = 4

    embedding_dim = 64
    num_encoder_layers = 2
    num_decoder_layers = 2
    num_heads = 4
    hidden_dim = 256

    dropout = 0.1

    learning_rate = 1e-3
    epochs = 3

    device = torch.device("cpu")

    # --------------------------------------
    # 2. Prepare datasets and tokenizer
    # --------------------------------------

    print("\nPreparing datasets...")

    (
        tokenizer,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    ) = prepare_dataset(
        train_samples=train_samples,
        validation_samples=validation_samples,
        num_merges=num_merges,
        max_length=max_sequence_length,
        batch_size=batch_size,
        num_workers=0
    )

    # --------------------------------------
    # 3. Validate tokenizer vocabulary
    # --------------------------------------

    if not tokenizer.token_to_id:
        raise ValueError(
            "Tokenizer vocabulary has not been built."
        )

    vocab_size = len(tokenizer.token_to_id)

    if vocab_size == 0:
        raise ValueError(
            "Tokenizer vocabulary is empty."
        )

    # Every token ID must be unique.
    token_ids = list(tokenizer.token_to_id.values())

    if len(token_ids) != len(set(token_ids)):
        raise ValueError(
            "Duplicate token IDs found in tokenizer."
        )

    # IDs should form a continuous range from 0.
    expected_ids = set(range(vocab_size))

    if set(token_ids) != expected_ids:
        raise ValueError(
            "Tokenizer IDs are not continuous."
        )

    print("\n--- VOCABULARY VALIDATION ---")
    print("Vocabulary size:", vocab_size)
    print("Token-to-ID entries:", len(tokenizer.token_to_id))
    print("ID-to-token entries:", len(tokenizer.id_to_token))

    # --------------------------------------
    # 4. Save tokenizer
    # --------------------------------------

    save_tokenizer(
        tokenizer=tokenizer,
        tokenizer_path=TOKENIZER_PATH
    )

    # --------------------------------------
    # 5. Dataset information
    # --------------------------------------

    print("\n--- DATASET INFORMATION ---")

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))

    print("Vocabulary size:", vocab_size)
    print("PAD token ID:", tokenizer.pad_token_id)
    print("UNK token ID:", tokenizer.unk_token_id)
    print("BOS token ID:", tokenizer.bos_token_id)
    print("EOS token ID:", tokenizer.eos_token_id)

    # --------------------------------------
    # 6. Initialize Transformer
    # --------------------------------------

    print("\nInitializing Transformer...")

    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Trainable parameters:", total_parameters)

    # --------------------------------------
    # 7. Initialize optimizer
    # --------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------
    # 8. Initialize Trainer
    # --------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_grad_norm=1.0,
        device=device
    )

    # --------------------------------------
    # 9. Checkpoint directory
    # --------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # 10. Start training
    # --------------------------------------

    print("\nStarting training...")

    history = fit(
        trainer=trainer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=epochs,
        checkpoint_dir=CHECKPOINT_DIR
    )

    # --------------------------------------
    # 11. Training summary
    # --------------------------------------

    print("\n--- TRAINING SUMMARY ---")

    print("Training completed.")

    print("Tokenizer:", TOKENIZER_PATH)
    print("Tokenizer exists:", TOKENIZER_PATH.exists())

    if TOKENIZER_PATH.exists():
        print(
            "Tokenizer size:",
            TOKENIZER_PATH.stat().st_size,
            "bytes"
        )

    print("Checkpoint directory:", CHECKPOINT_DIR)

    best_checkpoint = CHECKPOINT_DIR / "best.pt"
    latest_checkpoint = CHECKPOINT_DIR / "latest.pt"

    print("Best checkpoint exists:", best_checkpoint.exists())
    print("Latest checkpoint exists:", latest_checkpoint.exists())

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "checkpoint_dir": CHECKPOINT_DIR,
        "tokenizer_path": TOKENIZER_PATH
    }


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    run_training()

