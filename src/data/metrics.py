
import math

import torch


def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.

    Args:
        loss: Python float or scalar PyTorch tensor.

    Returns:
        Perplexity as a Python float.
    """

    if isinstance(loss, torch.Tensor):

        if loss.numel() != 1:
            raise ValueError("Loss tensor must contain exactly one value")

        if not torch.is_floating_point(loss):
            raise TypeError("Loss tensor must have a floating-point dtype")

        loss = loss.detach().item()

    elif isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise TypeError("Loss must be a number or scalar tensor")

    loss = float(loss)

    if not math.isfinite(loss):
        raise ValueError("Loss must be finite")

    if loss < 0:
        raise ValueError("Loss cannot be negative")

    # Prevent exponential overflow.
    max_safe_loss = math.log(float.fromhex("0x1.fffffffffffffp+1023"))

    if loss > max_safe_loss:
        raise OverflowError("Loss is too large to calculate perplexity")

    return math.exp(loss)


def calculate_metrics(loss):
    """
    Calculate training metrics from cross-entropy loss.
    """

    perplexity = calculate_perplexity(loss)

    if isinstance(loss, torch.Tensor):
        loss_value = loss.detach().item()
    else:
        loss_value = float(loss)

    return {
        "loss": loss_value,
        "perplexity": perplexity
    }
