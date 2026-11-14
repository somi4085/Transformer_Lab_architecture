
from pathlib import Path

from src.data.trainer import Trainer
from src.data.train import train_model
from src.data.validate import validate_model
from src.data.metrics import calculate_perplexity
from src.data.checkpoint import (
    save_checkpoint,
    load_checkpoint
)
from src.data.scheduler import create_scheduler
from src.data.early_stopping import EarlyStopping


def fit(
    trainer: Trainer,
    train_loader,
    validation_loader,
    epochs: int,
    checkpoint_dir="checkpoints",
    resume_from=None,
    scheduler_factor=0.5,
    scheduler_patience=2,
    scheduler_min_lr=1e-6,
    scheduler_threshold=1e-4,
    early_stopping_patience=3,
    early_stopping_min_delta=0.0
):
    """Train, validate, schedule LR, early-stop and resume."""

    if not isinstance(trainer, Trainer):
        raise TypeError("trainer must be a Trainer instance")

    if (
        not isinstance(epochs, int)
        or isinstance(epochs, bool)
        or epochs <= 0
    ):
        raise ValueError("epochs must be a positive integer")

    if len(train_loader) == 0:
        raise ValueError("train_loader cannot be empty")

    if len(validation_loader) == 0:
        raise ValueError("validation_loader cannot be empty")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = checkpoint_dir / "latest.pt"
    best_checkpoint = checkpoint_dir / "best.pt"

    history = []
    start_epoch = 0
    best_validation_loss = float("inf")

    scheduler = create_scheduler(
        optimizer=trainer.optimizer,
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
        threshold=scheduler_threshold
    )

    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta
    )

    # Restore previous training state if requested.
    if resume_from is not None:

        progress = load_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            checkpoint_path=resume_from,
            device=trainer.device,
            scheduler=scheduler,
            early_stopping=early_stopping
        )

        start_epoch = progress["epoch"] + 1
        history = progress["history"]

        if history:
            best_validation_loss = min(
                item["validation_loss"]
                for item in history
            )

        print(
            f"Resuming training from epoch {start_epoch + 1}"
        )

    if early_stopping.early_stop:
        print("Early stopping was already triggered.")
        return history

    if start_epoch >= epochs:
        print("Training is already complete.")
        return history

    for epoch in range(start_epoch, epochs):

        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        learning_rate_used = (
            trainer.optimizer.param_groups[0]["lr"]
        )

        # Training.
        training_history = train_model(
            trainer=trainer,
            train_loader=train_loader,
            epochs=1
        )

        train_loss = training_history[0]["train_loss"]

        # Validation.
        validation_metrics = validate_model(
            trainer=trainer,
            validation_loader=validation_loader
        )

        validation_loss = validation_metrics["validation_loss"]

        train_perplexity = calculate_perplexity(train_loss)

        validation_perplexity = calculate_perplexity(
            validation_loss
        )

        # Update scheduler.
        scheduler.step(validation_loss)

        next_learning_rate = (
            trainer.optimizer.param_groups[0]["lr"]
        )

        # Update early stopping.
        should_stop = early_stopping(validation_loss)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_perplexity": train_perplexity,
            "validation_loss": validation_loss,
            "validation_perplexity": validation_perplexity,
            "valid_tokens": validation_metrics["valid_tokens"],
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": next_learning_rate,
            "early_stopping_counter": early_stopping.counter
        }

        history.append(epoch_metrics)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train PPL: {train_perplexity:.4f} "
            f"| Val Loss: {validation_loss:.4f} "
            f"| Val PPL: {validation_perplexity:.4f} "
            f"| LR: {learning_rate_used:.8f} "
            f"| Next LR: {next_learning_rate:.8f} "
            f"| Early Stop Counter: {early_stopping.counter}"
        )

        # Save best checkpoint when validation improves.
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            save_checkpoint(
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=scheduler,
                early_stopping=early_stopping,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                history=history,
                checkpoint_path=best_checkpoint
            )

            print("Best checkpoint updated.")

        # Save latest checkpoint after every epoch.
        save_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            history=history,
            checkpoint_path=latest_checkpoint
        )

        # Stop after saving the final state of this epoch.
        if should_stop:
            print(
                f"Early stopping triggered at epoch {epoch + 1}."
            )
            break

    return history

