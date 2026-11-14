
from src.data.trainer import Trainer


def train_model(
    trainer: Trainer,
    train_loader,
    epochs: int
):
    """
    Train the Transformer model for multiple epochs.

    Returns:
        history: List containing average training loss per epoch.
    """

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

    history = []

    for epoch in range(epochs):

        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:

            batch_metrics = trainer.train_step(batch)

            batch_loss = batch_metrics["loss"]

            total_loss += batch_loss
            total_batches += 1

        average_loss = total_loss / total_batches

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss
        })

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"| Training Loss: {average_loss:.4f}"
        )

    return history
