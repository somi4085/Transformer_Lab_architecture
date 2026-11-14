
from src.tokenization.tokenizer import BPETokenizer


def calculate_compression_ratio(tokenizer, text):
    total_words = len(text.split())

    token_ids = tokenizer.encode_to_ids(text)

    special_tokens = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id
    }

    content_tokens = [
        token_id for token_id in token_ids
        if token_id not in special_tokens
    ]

    print("Input:", text)
    print("Token IDs:", token_ids)
    print("Special IDs:", special_tokens)
    print("Content tokens:", content_tokens)

    total_tokens = len(content_tokens)

    if total_tokens == 0:
        compression_ratio = 0.0
    else:
        compression_ratio = (total_tokens / total_words)

    return {
        "total_words": total_words,
        "total_tokens": total_tokens,
        "compression_ratio": compression_ratio
    }


def main():
    tokenizer = BPETokenizer()

    corpus = "low lower lowest low lower"
    tokenizer.train(corpus, num_merges=5)
    tokenizer.build_vocabulary()

    test_texts = [
        "low lower",
        "low lower low lower",
        "low lower lowest low lower"
    ]

    for text in test_texts:
        result = calculate_compression_ratio(tokenizer, text)
        print(result)


if __name__ == "__main__":
    main()
