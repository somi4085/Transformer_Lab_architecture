
from src.tokenization.tokenizer import BPETokenizer
from src.data.dataset import TextDataset


def train_tokenizer(train_texts, num_merges=1000):
    """
    Train BPE tokenizer using training texts only.
    """

    if not isinstance(train_texts, (list, tuple)):
        raise TypeError("train_texts must be a list or tuple")

    if not train_texts:
        raise ValueError("Training texts cannot be empty")

    if not all(isinstance(text, str) for text in train_texts):
        raise TypeError("Every training sample must be a string")

    if not isinstance(num_merges, int) or isinstance(num_merges, bool):
        raise TypeError("num_merges must be an integer")

    if num_merges < 0:
        raise ValueError("num_merges cannot be negative")

    # Create BPE tokenizer.
    tokenizer = BPETokenizer()

    # Combine training stories into one corpus.
    training_corpus = " ".join(train_texts)

    # Learn frequent character-pair merges.
    tokenizer.train(
        corpus=training_corpus,
        num_merges=num_merges
    )

    # Build token-to-ID and ID-to-token mappings.
    tokenizer.build_vocabulary()

    return tokenizer


def tokenize_texts(tokenizer, texts, max_length=None):
    """
    Convert a list of text samples into token ID sequences.
    """

    if not isinstance(tokenizer, BPETokenizer):
        raise TypeError("tokenizer must be a BPETokenizer instance")

    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple")

    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Every text sample must be a string")

    if max_length is not None:

        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise TypeError("max_length must be an integer")

        if max_length < 2:
            raise ValueError("max_length must be at least 2")

    tokenized_sequences = []

    for text in texts:

        # Convert text into token IDs.
        token_ids = tokenizer.encode_to_ids(
            text,
            add_special_tokens=True
        )

        # Truncate long sequences while preserving BOS and EOS.
        if max_length is not None:
            token_ids = tokenizer.truncate_sequence(
                token_ids,
                max_length=max_length
            )

        tokenized_sequences.append(token_ids)

    return tokenized_sequences


def create_tokenization_pipeline(
    train_texts,
    validation_texts,
    num_merges=1000,
    max_length=None
):
    """
    Train tokenizer, tokenize both splits,
    and create PyTorch datasets.
    """

    if not isinstance(validation_texts, (list, tuple)):
        raise TypeError("validation_texts must be a list or tuple")

    if not validation_texts:
        raise ValueError("Validation texts cannot be empty")

    # Step 1: Train tokenizer only on training data.
    tokenizer = train_tokenizer(
        train_texts=train_texts,
        num_merges=num_merges
    )

    # Step 2: Tokenize training stories.
    train_sequences = tokenize_texts(
        tokenizer=tokenizer,
        texts=train_texts,
        max_length=max_length
    )

    # Step 3: Tokenize validation stories using the same tokenizer.
    validation_sequences = tokenize_texts(
        tokenizer=tokenizer,
        texts=validation_texts,
        max_length=max_length
    )

    # Step 4: Create PyTorch datasets.
    train_dataset = TextDataset(train_sequences)

    validation_dataset = TextDataset(validation_sequences)

    return (
        tokenizer,
        train_dataset,
        validation_dataset
    )
