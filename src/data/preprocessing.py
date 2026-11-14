
def clean_text(text: str) -> str:
    """
    Normalize whitespace in raw text.
    """
    return " ".join(text.split())


def is_valid_text(text: str) -> bool:
    """
    Check whether the input is a non-empty string.
    """
    if not isinstance(text, str):
        return False

    return bool(text.strip())


def preprocess_text(text: str) -> str:
    """
    Clean and validate raw text.
    """
    if not is_valid_text(text):
        raise ValueError("Input must be a non-empty string.")

    cleaned_text = clean_text(text)

    if not is_valid_text(cleaned_text):
        raise ValueError("Text is empty after cleaning.")

    return cleaned_text


def remove_duplicates(texts: list[str]) -> list[str]:
    """
    Remove duplicate texts while preserving their original order.
    """
    unique_texts = []
    seen = set()

    for text in texts:
        cleaned_text = preprocess_text(text)

        if cleaned_text not in seen:
            seen.add(cleaned_text)
            unique_texts.append(cleaned_text)

    return unique_texts
