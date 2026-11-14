
from pathlib import Path

import torch
import torch.nn as nn


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path,
    device="cpu"
):
    """
    Load trained model weights from a checkpoint.

    Args:
        model: Initialized Transformer model.
        checkpoint_path: Path to saved checkpoint.
        device: Device on which the model should run.

    Returns:
        Loaded model in evaluation mode.
    """

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)

    # Step 1: Validate checkpoint path.
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise ValueError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    # Step 2: Load checkpoint onto the requested device.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must contain a dictionary"
        )

    # Step 3: Extract model weights.
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]

    elif "model_state" in checkpoint:
        model_state_dict = checkpoint["model_state"]

    else:
        raise KeyError(
            "Checkpoint does not contain model_state_dict "
            "or model_state"
        )

    # Step 4: Restore model parameters.
    model.load_state_dict(model_state_dict)

    # Step 5: Move model to requested device.
    model.to(device)

    # Step 6: Switch to evaluation mode.
    model.eval()

    print("\n========== CHECKPOINT LOADED ==========")
    print("Checkpoint:", checkpoint_path)
    print("Device:", device)
    print("Model mode: Evaluation")

    if "epoch" in checkpoint:
        print("Saved epoch:", checkpoint["epoch"])

    if "validation_loss" in checkpoint:
        print(
            "Validation loss:",
            checkpoint["validation_loss"]
        )

    return model


if __name__ == "__main__":
    print(
        "Checkpoint loader module ready. "
        "Initialize your Transformer before loading."
    )
