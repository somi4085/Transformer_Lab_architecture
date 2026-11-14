
import pickle
from pathlib import Path

import torch

from src.data.prepare_dataset import prepare_dataset
from src.model.transformer import Transformer

from evalaution.generation_evaluator import (
    evaluate_generation,
    display_generation_metrics,
    save_generation_metrics
)


# 1. Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "tokenizer.pkl"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "tinystories_smoke"
    / "best.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "generation_metrics.json"
)

# 2. Model configuration

MODEL_CONFIG = {
    "embedding_dim": 64,
    "max_sequence_length": 64,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "num_heads": 4,
    "hidden_dim": 256,
    "dropout": 0.1
}


# 3. Load saved tokenizer

def load_tokenizer():

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    if not tokenizer.token_to_id:
        raise ValueError(
            "Loaded tokenizer has an empty vocabulary."
        )

    print("Tokenizer loaded successfully.")
    print("Vocabulary size:", len(tokenizer.token_to_id))

    return tokenizer


# 4. Prepare validation dataset

def prepare_validation_data():

    (
        prepared_tokenizer,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    ) = prepare_dataset(
        train_samples=100,
        validation_samples=20,
        num_merges=200,
        max_length=64,
        batch_size=4,
        num_workers=0
    )

    return prepared_tokenizer, validation_dataset


# 5. Verify tokenizer consistency

def verify_tokenizer(saved_tokenizer, prepared_tokenizer):

    if saved_tokenizer.token_to_id != prepared_tokenizer.token_to_id:
        raise ValueError(
            "Tokenizer vocabulary mismatch. "
            "Evaluation tokenizer must match training tokenizer."
        )

    if saved_tokenizer.merges != prepared_tokenizer.merges:
        raise ValueError(
            "Tokenizer merges mismatch."
        )

    print("Tokenizer consistency verified.")


# 6. Load trained Transformer

def load_trained_model(tokenizer, device):

    model = Transformer(
        vocab_size=len(tokenizer.token_to_id),
        **MODEL_CONFIG
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
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

    model.to(device)
    model.eval()

    print("Trained Transformer loaded successfully.")

    return model


# 7. Extract prompts and references

def create_validation_prompts(
    validation_dataset,
    tokenizer,
    num_prompts=20
):

    prompts = []
    references = []

    total_samples = min(
        len(validation_dataset),
        num_prompts
    )

    for index in range(total_samples):

        example = validation_dataset[index]

        encoder_ids = example["encoder_input_ids"]
        label_ids = example["labels"]

        # Remove ignored labels, if present.
        label_ids = [
            token_id
            for token_id in label_ids
            if token_id != -100
        ]

        # Decode source prompt.
        prompt = tokenizer.decode(
            encoder_ids,
            skip_special_tokens=True
        )

        # Decode expected target text.
        reference = tokenizer.decode(
            label_ids,
            skip_special_tokens=True
        )

        prompts.append(prompt)
        references.append(reference)

    if not prompts:
        raise ValueError(
            "No validation prompts were created."
        )

    if len(prompts) != len(references):
        raise ValueError(
            "Prompt and reference counts do not match."
        )

    print("\nValidation samples prepared.")
    print("Total prompts:", len(prompts))
    print("Total references:", len(references))

    print("\nSample prompt:")
    print(prompts[0])

    print("\nSample reference:")
    print(references[0])

    return prompts, references


# 8. Run complete evaluation

def run_evaluation():

    device = torch.device("cpu")

    print("\n========== TRANSFORMER EVALUATION ==========")

    # Load saved tokenizer.
    tokenizer = load_tokenizer()

    # Prepare validation dataset.
    prepared_tokenizer, validation_dataset = (
        prepare_validation_data()
    )

    # Verify tokenizer consistency.
    verify_tokenizer(
        tokenizer,
        prepared_tokenizer
    )

    # Load trained model.
    model = load_trained_model(
        tokenizer,
        device
    )

    # Extract prompts and references.
    prompts, references = create_validation_prompts(
        validation_dataset,
        tokenizer,
        num_prompts=20
    )

    # Evaluate generation, BLEU and ROUGE.
    metrics = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        references=references,
        max_new_tokens=20,
        device=device
    )

    # Display evaluation results.
    display_generation_metrics(metrics)

    # Save all metrics.
    save_generation_metrics(
        metrics=metrics,
        output_path=METRICS_PATH
    )

    print("\nEvaluation completed successfully.")
    print("Metrics saved at:", METRICS_PATH)

    return metrics




# 9. Entry point

if __name__ == "__main__":
    run_evaluation()
