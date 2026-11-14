
from src.data.dataset import TextDataset


def prepare_seq2seq_examples(
    tokenized_sequences,
    bos_token_id,
    eos_token_id
):
    """
    Convert tokenized stories into encoder-decoder training examples.

    Each input sequence must have this format:
        [BOS, token_1, token_2, ..., EOS]

    Each output example contains:
        encoder_input_ids
        decoder_input_ids
        labels
    """

    if not isinstance(tokenized_sequences, (list, tuple)):
        raise TypeError("tokenized_sequences must be a list or tuple")

    if not tokenized_sequences:
        raise ValueError("tokenized_sequences cannot be empty")

    if (
        not isinstance(bos_token_id, int)
        or isinstance(bos_token_id, bool)
        or bos_token_id < 0
    ):
        raise ValueError("bos_token_id must be a non-negative integer")

    if (
        not isinstance(eos_token_id, int)
        or isinstance(eos_token_id, bool)
        or eos_token_id < 0
    ):
        raise ValueError("eos_token_id must be a non-negative integer")

    if bos_token_id == eos_token_id:
        raise ValueError("BOS and EOS token IDs must be different")

    examples = []

    for sequence_index, sequence in enumerate(tokenized_sequences):

        if not isinstance(sequence, (list, tuple)):
            raise TypeError(
                f"Sequence at index {sequence_index} must be a list or tuple"
            )

        if len(sequence) < 4:
            continue

        for token_id in sequence:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
            ):
                raise ValueError(
                    f"Invalid token ID in sequence {sequence_index}"
                )

        if sequence[0] != bos_token_id:
            raise ValueError(
                f"Sequence {sequence_index} must start with BOS"
            )

        if sequence[-1] != eos_token_id:
            raise ValueError(
                f"Sequence {sequence_index} must end with EOS"
            )

        # Remove BOS and EOS before splitting the story.
        content_tokens = sequence[1:-1]

        # Both source and target need at least one content token.
        if len(content_tokens) < 2:
            continue

        # Split the story into prefix and continuation.
        split_index = len(content_tokens) // 2

        source_tokens = content_tokens[:split_index]
        target_tokens = content_tokens[split_index:]

        # Encoder reads the source prefix.
        encoder_input_ids = (
            [bos_token_id]
            + source_tokens
            + [eos_token_id]
        )

        # Target sequence includes BOS and EOS.
        target_sequence = (
            [bos_token_id]
            + target_tokens
            + [eos_token_id]
        )

        # Shift target right for decoder input.
        decoder_input_ids = target_sequence[:-1]

        # Labels contain the next token at each position.
        labels = target_sequence[1:]

        examples.append({
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        })

    if not examples:
        raise ValueError(
            "No valid examples created. "
            "Each sequence needs BOS, EOS, and at least two content tokens."
        )

    return examples


def create_seq2seq_dataset(
    tokenized_sequences,
    bos_token_id,
    eos_token_id
):
    """
    Prepare encoder-decoder examples and wrap them in TextDataset.
    """

    examples = prepare_seq2seq_examples(
        tokenized_sequences=tokenized_sequences,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )

    return TextDataset(examples)



    