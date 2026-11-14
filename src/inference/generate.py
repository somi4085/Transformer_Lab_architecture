

import torch

from src.inference.greedy_decoder import greedy_decode


def generate_text(
    model,
    tokenizer,
    input_text,
    max_new_tokens=50,
    device=None
):
    """
    Generate text using a trained Encoder-Decoder Transformer.
    """

    if not isinstance(input_text, str):
        raise TypeError("input_text must be a string")

    if not input_text.strip():
        raise ValueError("input_text cannot be empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model = model.to(device)

    # Retrieve special token IDs.
    token_to_id = tokenizer.token_to_id

    bos_token_id = token_to_id["[BOS]"]
    eos_token_id = token_to_id["[EOS]"]
    pad_token_id = token_to_id["[PAD]"]

    # Encode text without adding special tokens.
    source_ids = tokenizer.encode_to_ids(
        input_text,
        add_special_tokens=False
    )

    if not source_ids:
        raise ValueError("Input text produced no tokens")

    # Add BOS and EOS exactly once.
    source_ids = [
        bos_token_id,
        *source_ids,
        eos_token_id
    ]

    # Check model's positional encoding limit.
    max_sequence_length = (
        model.encoder.input_embedding
        .positional_encoding.max_sequence_length
    )

    if len(source_ids) > max_sequence_length:
        raise ValueError(
            f"Input length {len(source_ids)} exceeds "
            f"maximum sequence length {max_sequence_length}"
        )

    # Create source tensors.
    source_token_ids = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device
    )

    source_mask = torch.ones(
        source_token_ids.shape,
        dtype=torch.long,
        device=device
    )

    # Generate output token IDs.
    generated_ids = greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        source_mask=source_mask,
        pad_token_id=pad_token_id
    )

    # Remove BOS, EOS and PAD tokens.
    output_ids = generated_ids[0].tolist()

    cleaned_ids = []

    for token_id in output_ids:

        if token_id == bos_token_id:
            continue

        if token_id == eos_token_id:
            break

        if token_id == pad_token_id:
            continue

        cleaned_ids.append(token_id)

    # Convert generated token IDs into text.
    generated_text = tokenizer.decode(
        cleaned_ids,
        skip_special_tokens=True
    )

    return generated_text
