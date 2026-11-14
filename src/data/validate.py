

import torch

from src.data.trainer import Trainer


def validate_model(
    trainer: Trainer,
    validation_loader
):
    """
    Evaluate the Transformer on validation data.

    No gradients are calculated and no optimizer updates occur.
    """

    if not isinstance(trainer, Trainer):
        raise TypeError(
            "trainer must be a Trainer instance"
        )

    if len(validation_loader) == 0:
        raise ValueError(
            "validation_loader cannot be empty"
        )

    model = trainer.model
    loss_fn = trainer.loss_fn
    device = trainer.device

    model.eval()

    total_loss = 0.0
    total_valid_tokens = 0

    with torch.no_grad():

        for batch in validation_loader:

            source_token_ids = batch[
                "encoder_input_ids"
            ].to(device)

            source_mask = batch[
                "encoder_attention_mask"
            ].to(device)

            target_token_ids = batch[
                "decoder_input_ids"
            ].to(device)

            target_mask = batch[
                "decoder_attention_mask"
            ].to(device)

            labels = batch["labels"].to(device)

            logits = model(
                source_token_ids=source_token_ids,
                target_token_ids=target_token_ids,
                source_mask=source_mask,
                target_mask=target_mask
            )

            loss = loss_fn(logits, labels)

            valid_tokens = (
                labels != -100
            ).sum().item()

            total_loss += (
                loss.item() * valid_tokens
            )

            total_valid_tokens += valid_tokens

    if total_valid_tokens == 0:
        raise ValueError(
            "Validation data contains no valid target tokens"
        )

    average_loss = (
        total_loss / total_valid_tokens
    )

    return {
        "validation_loss": average_loss,
        "valid_tokens": total_valid_tokens
    }
