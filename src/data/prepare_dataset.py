
from src.data.dataset_loader import load_tinystories
from src.data.dataset import TextDataset
from collections import Counter


from src.data.tokenization_pipeline import (
    create_tokenization_pipeline
)

from src.data.seq2seq_data_loader import (
    create_seq2seq_data_loader
)


def create_seq2seq_examples(
    tokenized_dataset,
    bos_token_id,
    eos_token_id
):
    examples = []

    for sequence in tokenized_dataset:

        # Remove existing BOS and EOS tokens.
        content = [
            token_id
            for token_id in sequence
            if token_id not in (bos_token_id, eos_token_id)
        ]

        # A minimum of two content tokens is required.
        if len(content) < 2:
            continue

        # Split story into source and target.
        split_index = len(content) // 2

        source_content = content[:split_index]
        target_content = content[split_index:]

        # Encoder receives the source sequence.
        encoder_input_ids = (
            [bos_token_id]
            + source_content
            + [eos_token_id]
        )

        # Decoder receives target without its final EOS.
        decoder_input_ids = (
            [bos_token_id]
            + target_content
        )

        # Labels are shifted one position to the left.
        labels = (
            target_content
            + [eos_token_id]
        )

        examples.append({
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        })

    return examples


def inspect_dataset(
    tokenizer,
    train_dataset,
    validation_dataset
):
    print("\n========== DATASET SUMMARY ==========")

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    print("Vocabulary size:", len(tokenizer.token_to_id))
    print("BPE merges:", len(tokenizer.merges))

    print("\n========== SPECIAL TOKEN IDs ==========")

    print("PAD:", tokenizer.pad_token_id)
    print("UNK:", tokenizer.unk_token_id)
    print("BOS:", tokenizer.bos_token_id)
    print("EOS:", tokenizer.eos_token_id)

    print("\n========== SAMPLE SEQ2SEQ EXAMPLE ==========")

    sample = train_dataset[0]

    print("Encoder input:", sample["encoder_input_ids"])
    print("Decoder input:", sample["decoder_input_ids"])
    print("Labels:", sample["labels"])

    print(
        "Encoder length:",
        len(sample["encoder_input_ids"])
    )

    print(
        "Decoder length:",
        len(sample["decoder_input_ids"])
    )

    print(
        "Labels length:",
        len(sample["labels"])
    )


def inspect_target_distribution(
    train_examples,
    tokenizer,
    num_examples=10
):
    print("\n========== TRAINING TARGET INSPECTION ==========")

    target_sequences = []
    target_lengths = []

    # Inspect first N examples.
    for index, example in enumerate(
        train_examples[:num_examples]
    ):
        decoder_ids = example["decoder_input_ids"]
        label_ids = example["labels"]

        decoder_text = tokenizer.decode(
            decoder_ids,
            skip_special_tokens=True
        )

        target_text = tokenizer.decode(
            label_ids,
            skip_special_tokens=True
        )

        print(f"\nExample {index + 1}")
        print("Decoder input:", decoder_text)
        print("Target labels:", target_text)

    # Analyse all target sequences.
    for example in train_examples:

        label_ids = example["labels"]

        target_text = tokenizer.decode(
            label_ids,
            skip_special_tokens=True
        )

        target_sequences.append(target_text)

        # Exclude EOS and padding tokens from length.
        content_ids = [
            token_id
            for token_id in label_ids
            if token_id not in (
                tokenizer.eos_token_id,
                tokenizer.pad_token_id
            )
        ]

        target_lengths.append(len(content_ids))

    # Count unique targets.
    target_counts = Counter(target_sequences)

    unique_targets = len(target_counts)

    # Find most frequent targets.
    most_common_targets = target_counts.most_common(5)

    print("\n========== TARGET DISTRIBUTION ==========")

    print("Total training examples:", len(train_examples))
    print("Unique target sequences:", unique_targets)

    print("\nMost frequent target sequences:")

    for target, frequency in most_common_targets:
        print(f"Frequency: {frequency} | Target: {target}")

    # Calculate target lengths.
    if target_lengths:

        average_length = (
            sum(target_lengths) / len(target_lengths)
        )

        print("\n========== TARGET LENGTH STATISTICS ==========")

        print("Minimum target length:", min(target_lengths))
        print("Maximum target length:", max(target_lengths))
        print("Average target length:", round(average_length, 2))

    print("\n========== INSPECTION COMPLETED ==========")


def inspect_dataloader(
    train_loader,
    validation_loader
):
    print("\n========== DATALOADER INSPECTION ==========")

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    train_batch = next(iter(train_loader))

    print("\nTraining batch:")

    for key, value in train_batch.items():
        print(
            f"{key}: shape={value.shape}, dtype={value.dtype}"
        )

    validation_batch = next(iter(validation_loader))

    print("\nValidation batch:")

    for key, value in validation_batch.items():
        print(
            f"{key}: shape={value.shape}, dtype={value.dtype}"
        )


def inspect_token_frequency(train_dataset, tokenizer):
    """
    Inspect the frequency distribution of target tokens.
    Ignore padding labels represented by -100.
    """

    token_frequency = Counter()

    total_target_tokens = 0

    for index in range(len(train_dataset)):

        example = train_dataset[index]

        labels = example["labels"]

        for token_id in labels:

            token_id = int(token_id)

            if token_id == -100:
                continue

            token_frequency[token_id] += 1
            total_target_tokens += 1

    unique_target_tokens = len(token_frequency)

    print("\n========== TARGET TOKEN DISTRIBUTION ==========")

    print(f"Total target tokens: {total_target_tokens}")
    print(f"Unique target tokens: {unique_target_tokens}")

    print("\nTop 20 frequent target tokens:\n")
    print(f"{'Token':<25} {'Frequency':>10}")
    print("-" * 38)

    for token_id, frequency in token_frequency.most_common(20):

        token = tokenizer.id_to_token.get(
            token_id,
            "[UNKNOWN]"
        )

        print(f"{token:<25} {frequency:>10}")

    print("=" * 50)

    return token_frequency


def inspect_bigram_frequency(train_dataset, tokenizer):
    """
    Inspect consecutive target-token pair frequencies.
    Ignore padding labels represented by -100.
    """

    bigram_frequency = Counter()

    total_bigrams = 0

    for index in range(len(train_dataset)):

        example = train_dataset[index]

        labels = [
            int(token_id)
            for token_id in example["labels"]
            if int(token_id) != -100
        ]

        # Create consecutive token pairs.
        for position in range(len(labels) - 1):

            first_token = labels[position]
            second_token = labels[position + 1]

            bigram = (first_token, second_token)

            bigram_frequency[bigram] += 1
            total_bigrams += 1

    print("\n========== BIGRAM FREQUENCY ==========")

    print(f"Total bigrams: {total_bigrams}")
    print(f"Unique bigrams: {len(bigram_frequency)}")

    print("\nTop 20 frequent bigrams:\n")
    print(f"{'Bigram':<35} {'Frequency':>10}")
    print("-" * 47)

    for (first_id, second_id), frequency in (
        bigram_frequency.most_common(20)
    ):

        first_token = tokenizer.id_to_token.get(
            first_id, "[UNKNOWN]"
        )

        second_token = tokenizer.id_to_token.get(
            second_id, "[UNKNOWN]"
        )

        bigram_text = f"{first_token} {second_token}"

        print(f"{bigram_text:<35} {frequency:>10}")

    # Check specific bigrams.
    token_to_id = tokenizer.token_to_id

    print("\n========== SPECIFIC BIGRAM CHECK ==========")

    pairs_to_check = [
        ("to</w>", "the</w>"),
        ("the</w>", "park</w>")
    ]

    for first_token, second_token in pairs_to_check:

        first_id = token_to_id.get(first_token)
        second_id = token_to_id.get(second_token)

        if first_id is None or second_id is None:
            frequency = 0
        else:
            frequency = bigram_frequency.get(
                (first_id, second_id), 0
            )

        print(
            f"{first_token} {second_token}: {frequency}"
        )

    print("=" * 47)

    return bigram_frequency


def prepare_dataset(
    train_samples=10_000,
    validation_samples=1_000,
    num_merges=1_000,
    max_length=256,
    batch_size=16,
    num_workers=0
):

    print("Loading TinyStories dataset...")

    # Step 1: Load raw stories.
    train_texts, validation_texts = load_tinystories(
        train_samples=train_samples,
        validation_samples=validation_samples
    )

    if not train_texts:
        raise ValueError("No valid training stories were loaded")

    if not validation_texts:
        raise ValueError("No valid validation stories were loaded")

    print("Training stories:", len(train_texts))
    print("Validation stories:", len(validation_texts))

    # Step 2: Train tokenizer and tokenize stories.
    print("\nTraining BPE tokenizer...")

    (
        tokenizer,
        tokenized_train_dataset,
        tokenized_validation_dataset
    ) = create_tokenization_pipeline(
        train_texts=train_texts,
        validation_texts=validation_texts,
        num_merges=num_merges,
        max_length=max_length
    )

    print("Tokenization completed.")

    # Step 3: Convert tokenized stories into Seq2Seq examples.
    print("\nPreparing Seq2Seq examples...")

    train_examples = create_seq2seq_examples(
        tokenized_dataset=tokenized_train_dataset,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    validation_examples = create_seq2seq_examples(
        tokenized_dataset=tokenized_validation_dataset,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    inspect_target_distribution(
    train_examples=train_examples,
    tokenizer=tokenizer,
    num_examples=10
    )

    if not train_examples:
        raise ValueError("No valid training Seq2Seq examples created")

    if not validation_examples:
        raise ValueError(
            "No valid validation Seq2Seq examples created"
        )

    # Step 4: Wrap examples in PyTorch datasets.
    train_dataset = TextDataset(train_examples)
    validation_dataset = TextDataset(validation_examples)

    inspect_token_frequency(train_dataset, tokenizer)
    inspect_bigram_frequency(train_dataset, tokenizer)

    # Step 5: Inspect datasets.
    inspect_dataset(
        tokenizer,
        train_dataset,
        validation_dataset
    )

    # Step 6: Create Seq2Seq DataLoaders.
    print("\nCreating Seq2Seq DataLoaders...")

    train_loader = create_seq2seq_data_loader(
        dataset=train_dataset,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    validation_loader = create_seq2seq_data_loader(
        dataset=validation_dataset,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # Step 7: Inspect batches.
    inspect_dataloader(
        train_loader,
        validation_loader
    )

    return (
        tokenizer,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    )


if __name__ == "__main__":

    (
        tokenizer,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader
    ) = prepare_dataset(
        train_samples=100,
        validation_samples=20,
        num_merges=200,
        max_length=64,
        batch_size=4,
        num_workers=0
    )

