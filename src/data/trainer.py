
import torch
import torch.nn as nn

from src.data.loss import TokenCrossEntropyLoss


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn=None,
        max_grad_norm=1.0,
        device="cpu"
    ):
        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be a PyTorch nn.Module"
            )

        if not isinstance(
            optimizer,
            torch.optim.Optimizer
        ):
            raise TypeError(
                "optimizer must be a PyTorch optimizer"
            )

        if (
            not isinstance(max_grad_norm, (int, float))
            or isinstance(max_grad_norm, bool)
            or max_grad_norm <= 0
        ):
            raise ValueError(
                "max_grad_norm must be positive"
            )

        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else TokenCrossEntropyLoss(ignore_index=-100)
        )

        self.max_grad_norm = float(max_grad_norm)

    def train_step(self, batch):

        self.model.train()

        source_token_ids = batch[
            "encoder_input_ids"
        ].to(self.device)

        source_mask = batch[
            "encoder_attention_mask"
        ].to(self.device)

        target_token_ids = batch[
            "decoder_input_ids"
        ].to(self.device)

        target_mask = batch[
            "decoder_attention_mask"
        ].to(self.device)

        labels = batch["labels"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(
            source_token_ids=source_token_ids,
            target_token_ids=target_token_ids,
            source_mask=source_mask,
            target_mask=target_mask
        )

        loss = self.loss_fn(logits, labels)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.max_grad_norm
        )

        self.optimizer.step()

        return {
            "loss": loss.detach().item(),
            "grad_norm": grad_norm.detach().item()
        }
