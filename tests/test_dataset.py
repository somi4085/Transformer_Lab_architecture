
import pytest
import torch

from torch.utils.data import Dataset
from src.data.dataset import TextDataset


def create_sequences():
    return [
        [2, 10, 15, 3],
        [2, 20, 25, 3],
        [2, 30, 35, 3]
    ]


# 1. Dataset length
def test_dataset_length():
    dataset = TextDataset(create_sequences())

    assert len(dataset) == 3


# 2. Retrieve first sequence
def test_get_first_sequence():
    dataset = TextDataset(create_sequences())

    assert dataset[0] == [2, 10, 15, 3]


# 3. Retrieve last sequence
def test_get_last_sequence():
    dataset = TextDataset(create_sequences())

    assert dataset[2] == [2, 30, 35, 3]


# 4. Negative indexing
def test_negative_indexing():
    dataset = TextDataset(create_sequences())

    assert dataset[-1] == [2, 30, 35, 3]


# 5. Empty dataset
def test_empty_dataset():
    dataset = TextDataset([])

    assert len(dataset) == 0


# 6. Invalid positive index
def test_invalid_positive_index():
    dataset = TextDataset(create_sequences())

    with pytest.raises(IndexError):
        dataset[3]


# 7. Invalid negative index
def test_invalid_negative_index():
    dataset = TextDataset(create_sequences())

    with pytest.raises(IndexError):
        dataset[-4]


# 8. PyTorch Dataset inheritance
def test_pytorch_dataset_inheritance():
    dataset = TextDataset(create_sequences())

    assert isinstance(dataset, Dataset)


# 9. Preserve original sequences
def test_sequences_are_preserved():
    sequences = create_sequences()
    dataset = TextDataset(sequences)

    assert dataset.sequences == sequences


# 10. Different sequence lengths
def test_different_sequence_lengths():
    sequences = [
        [2, 10, 3],
        [2, 20, 25, 3],
        [2, 30, 35, 40, 3]
    ]

    dataset = TextDataset(sequences)

    assert len(dataset) == 3
    assert dataset[0] == [2, 10, 3]
    assert dataset[2] == [2, 30, 35, 40, 3]


# 11. Tensor sequences
def test_tensor_sequences():
    sequences = [
        torch.tensor([2, 10, 15, 3]),
        torch.tensor([2, 20, 25, 3])
    ]

    dataset = TextDataset(sequences)

    assert torch.equal(
        dataset[0],
        torch.tensor([2, 10, 15, 3])
    )


# 12. Dataset returns the original sequence object
def test_sequence_object_identity():
    sequences = create_sequences()
    dataset = TextDataset(sequences)

    assert dataset[0] is sequences[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
