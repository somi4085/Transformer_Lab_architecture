
import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 2,
    min_lr: float = 1e-6,
    threshold: float = 1e-4
):
    """
    Create a ReduceLROnPlateau learning rate scheduler.

    The scheduler monitors validation loss and reduces
    the learning rate when improvement stops.
    """

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "optimizer must be a PyTorch Optimizer"
        )

    if not isinstance(factor, (int, float)) or isinstance(factor, bool):
        raise TypeError("factor must be a number")

    if not 0 < factor < 1:
        raise ValueError("factor must be between 0 and 1")

    if (
        not isinstance(patience, int)
        or isinstance(patience, bool)
        or patience < 0
    ):
        raise ValueError(
            "patience must be a non-negative integer"
        )

    if not isinstance(min_lr, (int, float)) or isinstance(min_lr, bool):
        raise TypeError("min_lr must be a number")

    if min_lr < 0:
        raise ValueError("min_lr cannot be negative")

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a number")

    if threshold < 0:
        raise ValueError("threshold cannot be negative")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode="rel",
        min_lr=min_lr
    )

    return scheduler
