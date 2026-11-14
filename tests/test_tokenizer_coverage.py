from src.tokenization.tokenizer import BPETokenizer


def test_known_vocabulary_coverage():
    tokenizer = BPETokenizer()
    corpus = "low lower lowest low lower"

    tokenizer.train(corpus, num_merges=5)
    tokenizer.build_vocabulary()

    encode_ids = tokenizer.encode_to_ids(corpus)
    unk_count = encode_ids.count(tokenizer.unk_token_id)

    assert unk_count == 0

def test_unk_character():
    tokenizer = BPETokenizer()
    corpus = "low lower lowest low lower"
    tokenizer.train(corpus, num_merges=5)
    tokenizer.build_vocabulary()
    text = "low🙂"
    encoded_ids = tokenizer.encode_to_ids(text)
    assert tokenizer.unk_token_id in encoded_ids

