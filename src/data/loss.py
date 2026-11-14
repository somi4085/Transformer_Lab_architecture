
import torch
import torch.nn as nn


class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-100):
        super().__init__()

        if (
            not isinstance(ignore_index, int)
            or isinstance(ignore_index, bool)
        ):
            raise TypeError("ignore_index must be an integer")

        self.ignore_index = ignore_index

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean"
        )

    def forward(self, logits, labels):

        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")

        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor")

        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape (batch, sequence, vocab_size)"
            )

        if labels.ndim != 2:
            raise ValueError(
                "labels must have shape (batch, sequence)"
            )

        if logits.shape[:2] != labels.shape:
            raise ValueError(
                "Logits and labels batch/sequence dimensions must match"
            )

        if logits.shape[-1] <= 0:
            raise ValueError("Vocabulary dimension cannot be empty")

        if labels.dtype != torch.long:
            raise TypeError("labels must have torch.long dtype")

        if not torch.isfinite(logits).all():
            raise ValueError("logits contain non-finite values")

        valid_labels = labels != self.ignore_index

        if not valid_labels.any():
            raise ValueError("All labels are ignored; loss cannot be calculated")

        if (
            labels[valid_labels].min() < 0
            or labels[valid_labels].max() >= logits.shape[-1]
        ):
            raise ValueError("Target token ID is outside vocabulary range")

        batch_size, sequence_length, vocab_size = logits.shape

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size
        )

        labels = labels.reshape(
            batch_size * sequence_length
        )

        loss = self.criterion(logits, labels)

        return loss
