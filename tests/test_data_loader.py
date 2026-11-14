
import pytest
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.data.dataset import TextDataset
from src.data.data_loader import (
    PAD_ID,
    collate_sequences,
    create_data_loader
)


def create_dataset(num_samples=5):
    sequences = [
        [2, 10 + i, 20 + i, 3]
        for i in range(num_samples)
    ]

    return TextDataset(sequences)


# 1. Returns a PyTorch DataLoader
def test_returns_dataloader():
    loader = create_data_loader(create_dataset())

    assert isinstance(loader, DataLoader)


# 2. Correct number of batches
def test_number_of_batches():
    loader = create_data_loader(
        create_dataset(5),
        batch_size=2,
        shuffle=False
    )

    assert len(loader) == 3


# 3. Batch output structure
def test_batch_output_structure():
    loader = create_data_loader(
        create_dataset(4),
        batch_size=2,
        shuffle=False
    )

    batch = next(iter(loader))

    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "attention_mask" in batch


# 4. Batch shape
def test_batch_shape():
    loader = create_data_loader(
        create_dataset(4),
        batch_size=2,
        shuffle=False
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)


# 5. Different-length sequences are padded
def test_dynamic_padding():
    sequences = [
        [2, 10, 15, 3],
        [2, 20, 3],
        [2, 30, 35, 40, 3]
    ]

    batch = collate_sequences(sequences)

    expected = torch.tensor([
        [2, 10, 15, 3, 0],
        [2, 20, 3, 0, 0],
        [2, 30, 35, 40, 3]
    ])

    assert torch.equal(batch["input_ids"], expected)


# 6. Attention mask correctness
def test_attention_mask():
    sequences = [
        [2, 10, 15, 3],
        [2, 20, 3],
        [2, 30, 35, 40, 3]
    ]

    batch = collate_sequences(sequences)

    expected_mask = torch.tensor([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1]
    ])

    assert torch.equal(
        batch["attention_mask"],
        expected_mask
    )


# 7. Correct padding token ID
def test_padding_token_id():
    sequences = [
        [2, 10, 3],
        [2, 20, 25, 3]
    ]

    batch = collate_sequences(sequences)

    assert batch["input_ids"][0, -1].item() == PAD_ID


# 8. Tensor dtype
def test_batch_dtype():
    batch = collate_sequences([
        [2, 10, 3],
        [2, 20, 25, 3]
    ])

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long


# 9. Empty batch validation
def test_empty_batch():
    with pytest.raises(ValueError):
        collate_sequences([])


# 10. Invalid sequence dimensions
def test_invalid_sequence_dimensions():
    sequences = [
        torch.tensor([[2, 10, 3]]),
        torch.tensor([2, 20, 3])
    ]

    with pytest.raises(ValueError):
        collate_sequences(sequences)


# 11. Sequential sampling
def test_sequential_sampling():
    loader = create_data_loader(
        create_dataset(5),
        batch_size=2,
        shuffle=False
    )

    assert isinstance(loader.sampler, SequentialSampler)

    batches = list(loader)
    actual = torch.cat(
        [batch["input_ids"] for batch in batches],
        dim=0
    )

    expected = torch.tensor([
        [2, 10, 20, 3],
        [2, 11, 21, 3],
        [2, 12, 22, 3],
        [2, 13, 23, 3],
        [2, 14, 24, 3]
    ])

    assert torch.equal(actual, expected)


# 12. Random sampling
def test_random_sampling():
    loader = create_data_loader(
        create_dataset(5),
        batch_size=2,
        shuffle=True
    )

    assert isinstance(loader.sampler, RandomSampler)


# 13. Incomplete final batch
def test_drop_last_false():
    loader = create_data_loader(
        create_dataset(5),
        batch_size=2,
        shuffle=False,
        drop_last=False
    )

    batches = list(loader)

    assert len(batches) == 3
    assert batches[-1]["input_ids"].shape[0] == 1


# 14. Drop incomplete final batch
def test_drop_last_true():
    loader = create_data_loader(
        create_dataset(5),
        batch_size=2,
        shuffle=False,
        drop_last=True
    )

    batches = list(loader)

    assert len(batches) == 2

    assert all(
        batch["input_ids"].shape[0] == 2
        for batch in batches
    )


# 15. Empty dataset
def test_empty_dataset():
    loader = create_data_loader(
        TextDataset([]),
        batch_size=2,
        shuffle=False
    )

    assert len(loader) == 0
    assert list(loader) == []


# 16. Invalid dataset
def test_invalid_dataset():
    with pytest.raises(TypeError):
        create_data_loader([1, 2, 3])


# 17. Invalid batch size
@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        create_data_loader(
            create_dataset(),
            batch_size=batch_size
        )


# 18. Invalid worker count
def test_invalid_num_workers():
    with pytest.raises(ValueError):
        create_data_loader(
            create_dataset(),
            num_workers=-1
        )


# 19. Zero workers
def test_zero_workers():
    loader = create_data_loader(
        create_dataset(),
        num_workers=0,
        shuffle=False
    )

    assert loader.num_workers == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
