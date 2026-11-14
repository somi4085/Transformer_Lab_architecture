
from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    history: list,
    checkpoint_path,
    scheduler=None,
    early_stopping=None
):
    """Save model, optimizer, scheduler, early stopping and history."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module")

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a PyTorch Optimizer")

    if (
        not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
    ):
        raise ValueError("epoch must be a non-negative integer")

    if not isinstance(history, list):
        raise TypeError("history must be a list")

    for name, value in (
        ("train_loss", train_loss),
        ("validation_loss", validation_loss)
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not torch.isfinite(torch.tensor(float(value))).item()
        ):
            raise ValueError(f"{name} must be a finite number")

    if scheduler is not None:
        if not hasattr(scheduler, "state_dict"):
            raise TypeError("scheduler must support state_dict()")

    if early_stopping is not None:
        if not hasattr(early_stopping, "state_dict"):
            raise TypeError(
                "early_stopping must support state_dict()"
            )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "early_stopping_state_dict": (
            early_stopping.state_dict()
            if early_stopping is not None
            else None
        ),
        "epoch": epoch,
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "history": history
    }

    temporary_path = checkpoint_path.with_name(
        checkpoint_path.name + ".tmp"
    )

    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path,
    device="cpu",
    scheduler=None,
    early_stopping=None
):
    """Restore model, optimizer, scheduler and early stopping."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch nn.Module")

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a PyTorch Optimizer")

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "train_loss",
        "validation_loss",
        "history"
    }

    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise ValueError(
            f"Invalid checkpoint. Missing keys: {missing_keys}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler_state = checkpoint.get("scheduler_state_dict")

    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    early_stopping_state = checkpoint.get(
        "early_stopping_state_dict"
    )

    if (
        early_stopping is not None
        and early_stopping_state is not None
    ):
        early_stopping.load_state_dict(
            early_stopping_state
        )

    return {
        "epoch": checkpoint["epoch"],
        "train_loss": checkpoint["train_loss"],
        "validation_loss": checkpoint["validation_loss"],
        "history": checkpoint["history"]
    }
