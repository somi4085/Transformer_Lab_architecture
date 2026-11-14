
import pytest

from src.data.tokenization_pipeline import (
    train_tokenizer,
    tokenize_texts,
    create_tokenization_pipeline
)
from src.data.dataset import TextDataset
from src.tokenization.tokenizer import BPETokenizer


@pytest.fixture
def sample_texts():
    train_texts = [
        "the cat is sleeping",
        "the dog is running",
        "a little cat is playing"
    ]

    validation_texts = [
        "the cat is running",
        "a little dog is sleeping"
    ]

    return train_texts, validation_texts


@pytest.fixture
def trained_tokenizer(sample_texts):
    train_texts, _ = sample_texts

    return train_tokenizer(
        train_texts=train_texts,
        num_merges=20
    )


# 1. Tokenizer training
def test_tokenizer_training(sample_texts):
    train_texts, _ = sample_texts

    tokenizer = train_tokenizer(
        train_texts,
        num_merges=20
    )

    assert isinstance(tokenizer, BPETokenizer)
    assert len(tokenizer.vocabulary) > 0
    assert len(tokenizer.token_to_id) > 0


# 2. Special token IDs
def test_special_token_ids(trained_tokenizer):
    tokenizer = trained_tokenizer

    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3


# 3. Tokenization output
def test_tokenize_texts_output(trained_tokenizer, sample_texts):
    _, validation_texts = sample_texts

    sequences = tokenize_texts(
        tokenizer=trained_tokenizer,
        texts=validation_texts
    )

    assert len(sequences) == len(validation_texts)

    for sequence in sequences:
        assert isinstance(sequence, list)
        assert len(sequence) > 0


# 4. BOS and EOS preservation
def test_special_tokens_in_sequences(
    trained_tokenizer,
    sample_texts
):
    _, validation_texts = sample_texts

    sequences = tokenize_texts(
        tokenizer=trained_tokenizer,
        texts=validation_texts
    )

    for sequence in sequences:
        assert sequence[0] == trained_tokenizer.bos_token_id
        assert sequence[-1] == trained_tokenizer.eos_token_id


# 5. Maximum sequence length
def test_max_length_truncation(
    trained_tokenizer,
    sample_texts
):
    _, validation_texts = sample_texts

    sequences = tokenize_texts(
        tokenizer=trained_tokenizer,
        texts=validation_texts,
        max_length=5
    )

    for sequence in sequences:
        assert len(sequence) <= 5
        assert sequence[0] == trained_tokenizer.bos_token_id
        assert sequence[-1] == trained_tokenizer.eos_token_id


# 6. Unknown character handling
def test_unknown_character_handling(trained_tokenizer):
    sequences = tokenize_texts(
        tokenizer=trained_tokenizer,
        texts=["the cat 🐱"]
    )

    sequence = sequences[0]

    assert trained_tokenizer.unk_token_id in sequence


# 7. Validation data does not train tokenizer
def test_validation_does_not_change_vocabulary(sample_texts):
    train_texts, validation_texts = sample_texts

    tokenizer_1 = train_tokenizer(
        train_texts,
        num_merges=20
    )

    tokenizer_2 = train_tokenizer(
        train_texts,
        num_merges=20
    )

    assert tokenizer_1.merges == tokenizer_2.merges
    assert tokenizer_1.token_to_id == tokenizer_2.token_to_id

    # Validation texts are encoded, not used for training.
    tokenize_texts(tokenizer_1, validation_texts)

    assert tokenizer_1.merges == tokenizer_2.merges
    assert tokenizer_1.token_to_id == tokenizer_2.token_to_id


# 8. PyTorch Dataset integration
def test_dataset_integration(sample_texts):
    train_texts, validation_texts = sample_texts

    tokenizer, train_dataset, validation_dataset = (
        create_tokenization_pipeline(
            train_texts=train_texts,
            validation_texts=validation_texts,
            num_merges=20
        )
    )

    assert isinstance(tokenizer, BPETokenizer)
    assert isinstance(train_dataset, TextDataset)
    assert isinstance(validation_dataset, TextDataset)

    assert len(train_dataset) == len(train_texts)
    assert len(validation_dataset) == len(validation_texts)


# 9. Dataset sequences contain token IDs
def test_dataset_contains_token_ids(sample_texts):
    train_texts, validation_texts = sample_texts

    _, train_dataset, validation_dataset = (
        create_tokenization_pipeline(
            train_texts,
            validation_texts,
            num_merges=20
        )
    )

    for dataset in [train_dataset, validation_dataset]:
        for sequence in dataset:
            assert isinstance(sequence, list)
            assert all(isinstance(token_id, int) for token_id in sequence)


# 10. Empty training data
def test_empty_training_data():
    with pytest.raises(ValueError):
        train_tokenizer([])


# 11. Empty validation data
def test_empty_validation_data(sample_texts):
    train_texts, _ = sample_texts

    with pytest.raises(ValueError):
        create_tokenization_pipeline(
            train_texts=train_texts,
            validation_texts=[]
        )


# 12. Invalid tokenizer input
def test_invalid_tokenizer_input():
    with pytest.raises(TypeError):
        tokenize_texts(
            tokenizer="invalid",
            texts=["the cat"]
        )


# 13. Invalid text input
def test_invalid_text_input(trained_tokenizer):
    with pytest.raises(TypeError):
        tokenize_texts(
            tokenizer=trained_tokenizer,
            texts=["the cat", 123]
        )


# 14. Invalid max length
@pytest.mark.parametrize("max_length", [0, 1, -1])
def test_invalid_max_length(trained_tokenizer, max_length):
    with pytest.raises(ValueError):
        tokenize_texts(
            tokenizer=trained_tokenizer,
            texts=["the cat"],
            max_length=max_length
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
