
from itertools import islice

from datasets import load_dataset


def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    # Normalize whitespace.
    text = " ".join(text.split())

    return text.strip()


def load_tinystories(
    train_samples=10_000,
    validation_samples=1_000
):

    if train_samples <= 0:
        raise ValueError("train_samples must be greater than zero")

    if validation_samples <= 0:
        raise ValueError(
            "validation_samples must be greater than zero"
        )

    print("Loading TinyStories training split...")

    train_stream = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True
    )

    print("Loading TinyStories validation split...")

    validation_stream = load_dataset(
        "roneneldan/TinyStories",
        split="validation",
        streaming=True
    )

    # Collect training stories.
    train_texts = []

    for example in islice(train_stream, train_samples):

        text = preprocess_text(example.get("text", ""))

        if text:
            train_texts.append(text)

    # Collect validation stories.
    validation_texts = []

    for example in islice(
        validation_stream,
        validation_samples
    ):

        text = preprocess_text(example.get("text", ""))

        if text:
            validation_texts.append(text)

    if not train_texts:
        raise RuntimeError(
            "No training stories were loaded from TinyStories"
        )

    if not validation_texts:
        raise RuntimeError(
            "No validation stories were loaded from TinyStories"
        )

    print("\n========== DATASET LOADING SUMMARY ==========")

    print("Training stories loaded:", len(train_texts))
    print("Validation stories loaded:", len(validation_texts))

    print("\nSample training story:")
    print(train_texts[0])

    return train_texts, validation_texts


if __name__ == "__main__":

    train_texts, validation_texts = load_tinystories(
        train_samples=100,
        validation_samples=20
    )

    print("\nDataset loading completed.")
