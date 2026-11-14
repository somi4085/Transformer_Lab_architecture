import pytest

from src.tokenization.tokenizer import BPETokenizer


@pytest.fixture
def tokenizer():
    tokenizer = BPETokenizer()
    corpus = "low lower lowest low lower"
    tokenizer.train(corpus, num_merges=5)
    tokenizer.build_vocabulary()
    return tokenizer


def test_empty_string(tokenizer):
    result = tokenizer.encode_to_ids("")
    assert isinstance(result, list)


def test_multiple_spaces(tokenizer):
    result = tokenizer.encode_to_ids("low     lower")
    assert isinstance(result, list)
    assert len(result) > 0


def test_leading_trailing_spaces(tokenizer):
    result = tokenizer.encode_to_ids("  low lower  ")
    assert isinstance(result, list)
    assert len(result) > 0


def test_punctuation(tokenizer):
    result = tokenizer.encode_to_ids("Hello, world!")
    assert isinstance(result, list)
    assert len(result) > 0


def test_unknown_unicode(tokenizer):
    result = tokenizer.encode_to_ids("low🙂")
    assert isinstance(result, list)
    assert tokenizer.unk_token_id in result