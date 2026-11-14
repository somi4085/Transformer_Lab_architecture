
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset


def collate_seq2seq(batch, pad_token_id):
    """
    Pad encoder inputs, decoder inputs, and labels independently.
    """

    if not batch:
        raise ValueError("Batch cannot be empty")

    encoder_sequences = []
    decoder_sequences = []
    label_sequences = []

    for example in batch:
        required_keys = {
            "encoder_input_ids",
            "decoder_input_ids",
            "labels"
        }

        if not isinstance(example, dict):
            raise TypeError("Each example must be a dictionary")

        if not required_keys.issubset(example):
            raise ValueError("Example is missing required keys")

        encoder = torch.as_tensor(
            example["encoder_input_ids"],
            dtype=torch.long
        )

        decoder = torch.as_tensor(
            example["decoder_input_ids"],
            dtype=torch.long
        )

        labels = torch.as_tensor(
            example["labels"],
            dtype=torch.long
        )

        if encoder.ndim != 1 or decoder.ndim != 1 or labels.ndim != 1:
            raise ValueError("All sequences must be one-dimensional")

        if len(encoder) == 0 or len(decoder) == 0 or len(labels) == 0:
            raise ValueError("Sequences cannot be empty")

        if len(decoder) != len(labels):
            raise ValueError(
                "Decoder input and labels must have equal lengths"
            )

        encoder_sequences.append(encoder)
        decoder_sequences.append(decoder)
        label_sequences.append(labels)

    batch_size = len(batch)

    max_encoder_length = max(
        len(sequence) for sequence in encoder_sequences
    )

    max_decoder_length = max(
        len(sequence) for sequence in decoder_sequences
    )

    encoder_input_ids = torch.full(
        (batch_size, max_encoder_length),
        fill_value=pad_token_id,
        dtype=torch.long
    )

    decoder_input_ids = torch.full(
        (batch_size, max_decoder_length),
        fill_value=pad_token_id,
        dtype=torch.long
    )

    labels = torch.full(
        (batch_size, max_decoder_length),
        fill_value=-100,
        dtype=torch.long
    )

    encoder_attention_mask = torch.zeros(
        (batch_size, max_encoder_length),
        dtype=torch.long
    )

    decoder_attention_mask = torch.zeros(
        (batch_size, max_decoder_length),
        dtype=torch.long
    )

    for index in range(batch_size):

        encoder_length = len(encoder_sequences[index])
        decoder_length = len(decoder_sequences[index])

        encoder_input_ids[index, :encoder_length] = (
            encoder_sequences[index]
        )

        decoder_input_ids[index, :decoder_length] = (
            decoder_sequences[index]
        )

        labels[index, :decoder_length] = label_sequences[index]

        encoder_attention_mask[index, :encoder_length] = 1
        decoder_attention_mask[index, :decoder_length] = 1

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels
    }


def create_seq2seq_data_loader(
    dataset: TextDataset,
    pad_token_id: int,
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0
):
    """
    Create a DataLoader for encoder-decoder training.
    """

    if not isinstance(dataset, TextDataset):
        raise TypeError("dataset must be a TextDataset")

    if (
        not isinstance(pad_token_id, int)
        or isinstance(pad_token_id, bool)
        or pad_token_id < 0
    ):
        raise ValueError("pad_token_id must be a non-negative integer")

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_seq2seq(
            batch,
            pad_token_id=pad_token_id
        )
    )


def create_training_and_validation_loaders(
    train_dataset,
    validation_dataset,
    pad_token_id,
    batch_size=16,
    num_workers=0
):
    """
    Create separate training and validation DataLoaders.
    """

    train_loader = create_seq2seq_data_loader(
        dataset=train_dataset,
        pad_token_id=pad_token_id,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = create_seq2seq_data_loader(
        dataset=validation_dataset,
        pad_token_id=pad_token_id,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, validation_loader
