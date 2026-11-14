
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Dataset for storing tokenized text sequences.
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]



sequences = [
    [2, 10, 15, 3],
    [2, 20, 25, 3],
    [2, 30, 35, 3]
]

dataset = TextDataset(sequences)

print("Dataset size:", len(dataset))
print("First sequence:", dataset[0])
print("Third sequence:", dataset[2])
