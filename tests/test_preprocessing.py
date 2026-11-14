
import pytest

from src.data.preprocessing import (
    clean_text,
    is_valid_text,
    preprocess_text,
    remove_duplicates
)


# 1. Clean normal text
def test_clean_text():
    text = "  Once   upon a time.  "

    result = clean_text(text)

    assert result == "Once upon a time."


# 2. Clean newlines and tabs
def test_clean_text_with_newlines():
    text = "Once upon a time.\n\nA cat\t lived there."

    result = clean_text(text)

    assert result == "Once upon a time. A cat lived there."


# 3. Empty string validation
def test_empty_string_is_invalid():
    assert is_valid_text("") is False


# 4. Whitespace-only validation
def test_whitespace_only_is_invalid():
    assert is_valid_text("   \n\t ") is False


# 5. Non-string validation
@pytest.mark.parametrize("value", [123, None, 12.5, ["hello"]])
def test_non_string_is_invalid(value):
    assert is_valid_text(value) is False


# 6. Valid string validation
def test_valid_string():
    assert is_valid_text("A cat lived there.") is True


# 7. Preprocess valid text
def test_preprocess_text():
    text = "  Once   upon a time.\n A cat lived there.  "

    result = preprocess_text(text)

    assert result == "Once upon a time. A cat lived there."


# 8. Preprocess invalid text
@pytest.mark.parametrize("value", ["", "   ", None, 123])
def test_preprocess_invalid_text(value):
    with pytest.raises(ValueError):
        preprocess_text(value)


# 9. Remove duplicate texts
def test_remove_duplicates():
    texts = [
        "A cat lived there.",
        "A dog lived there.",
        "A cat lived there."
    ]

    result = remove_duplicates(texts)

    assert result == [
        "A cat lived there.",
        "A dog lived there."
    ]


# 10. Preserve original order
def test_remove_duplicates_preserves_order():
    texts = [
        "Story C",
        "Story A",
        "Story B",
        "Story A"
    ]

    result = remove_duplicates(texts)

    assert result == [
        "Story C",
        "Story A",
        "Story B"
    ]


# 11. Remove duplicates after whitespace cleaning
def test_remove_whitespace_duplicates():
    texts = [
        "Hello world",
        "  Hello   world  ",
        "Good morning"
    ]

    result = remove_duplicates(texts)

    assert result == [
        "Hello world",
        "Good morning"
    ]


# 12. Empty list
def test_remove_duplicates_empty_list():
    assert remove_duplicates([]) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
