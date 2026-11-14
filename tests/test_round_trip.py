
from src.tokenization.tokenizer import BPETokenizer

def test_basic_round_trip():
    tokenizer = BPETokenizer()
    text = "low lower lowest low lower"
    trained = tokenizer.train(text,num_merges=5)
    tokenizer.build_vocabulary()
    original_text = "low lower"
    token_ids = tokenizer.encode_to_ids(original_text)
    print(tokenizer.encode("low lower"))
    print(token_ids)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == original_text
    assert len(token_ids) > 0
    assert isinstance(decoded_text,str)

def test_punctuation_round_trip():
    tokenizer = BPETokenizer()
    corpus = "Hello, world! How are you?"
    tokenizer.train(corpus, num_merges=5)
    tokenizer.build_vocabulary()
    text = "Hello, world! How are you?"
    token_ids = tokenizer.encode_to_ids(text)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == text
    



