
import torch

from torch.utils.data import DataLoader


def collate_sequences(batch, pad_token_id):

    # Convert sequences into tensors.
    sequences = [
        torch.as_tensor(sequence, dtype=torch.long)
        for sequence in batch
    ]

    if not sequences:
        raise ValueError("Batch cannot be empty")

    # Find the longest sequence in this batch.
    max_length = max(
        len(sequence)
        for sequence in sequences
    )

    if max_length == 0:
        raise ValueError("Sequences cannot be empty")

    batch_size = len(sequences)

    # Initialize padded input IDs.
    input_ids = torch.full(
        (batch_size, max_length),
        fill_value=pad_token_id,
        dtype=torch.long
    )

    # Initialize attention mask.
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    )

    # Copy each sequence into the padded batch.
    for index, sequence in enumerate(sequences):

        sequence_length = len(sequence)

        input_ids[index, :sequence_length] = sequence

        attention_mask[index, :sequence_length] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def create_data_loader(
    dataset,
    pad_token_id,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0
):

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_sequences(
            batch,
            pad_token_id
        )
    )


def create_training_and_validation_loaders(
    train_dataset,
    validation_dataset,
    pad_token_id,
    batch_size=16,
    num_workers=0
):

    # Training loader: shuffle enabled.
    train_loader = create_data_loader(
        dataset=train_dataset,
        pad_token_id=pad_token_id,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    # Validation loader: shuffle disabled.
    validation_loader = create_data_loader(
        dataset=validation_dataset,
        pad_token_id=pad_token_id,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, validation_loader

