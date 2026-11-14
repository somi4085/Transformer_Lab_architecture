
import torch


def greedy_decode(
    model,
    source_token_ids,
    bos_token_id,
    eos_token_id,
    max_new_tokens,
    source_mask=None,
    pad_token_id=0
):
    """
    Generate tokens using greedy decoding.

    Args:
        model: Trained Encoder-Decoder Transformer.
        source_token_ids: Source tokens, shape (B, S).
        bos_token_id: Beginning-of-sequence token ID.
        eos_token_id: End-of-sequence token ID.
        max_new_tokens: Maximum number of tokens to generate.
        source_mask: Optional source padding mask.
        pad_token_id: Padding token ID.

    Returns:
        Generated token IDs, shape (B, generated_length).
    """

    if source_token_ids.ndim != 2:
        raise ValueError(
            "source_token_ids must have shape (B, S)"
        )

    if source_token_ids.size(1) == 0:
        raise ValueError("Source sequence cannot be empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    if max_new_tokens == 0:
        return torch.full(
            (source_token_ids.size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=source_token_ids.device
        )

    if source_mask is not None:
        if source_mask.shape != source_token_ids.shape:
            raise ValueError(
                "source_mask must have the same shape "
                "as source_token_ids"
            )

    original_training_mode = model.training

    try:
        model.eval()

        generated = torch.full(
            (source_token_ids.size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=source_token_ids.device
        )

        finished = torch.zeros(
            source_token_ids.size(0),
            dtype=torch.bool,
            device=source_token_ids.device
        )

        with torch.inference_mode():

            for _ in range(max_new_tokens):

                logits = model(
                    source_token_ids=source_token_ids,
                    target_token_ids=generated,
                    source_mask=source_mask,
                    target_mask=None
                )

                # Select logits from the last decoder position.
                next_token_logits = logits[:, -1, :]

                # Select the token with the highest logit.
                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1
                )

                # Finished sequences should only receive PAD.
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, pad_token_id),
                    next_token
                )

                # Append the selected token.
                generated = torch.cat(
                    [
                        generated,
                        next_token.unsqueeze(1)
                    ],
                    dim=1
                )

                # Mark sequences that generated EOS.
                finished = finished | (
                    next_token == eos_token_id
                )

                # Stop when every sequence is finished.
                if finished.all():
                    break

        return generated

    finally:
        model.train(original_training_mode)
