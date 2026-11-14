
from src.tokenization.tokenizer import BPETokenizer


def calculate_unk_rate(tokenizer, text):
    token_ids = tokenizer.encode_to_ids(text)

    special_token_ids = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id
    }

    content_tokens = [
        token_id for token_id in token_ids
        if token_id not in special_token_ids
    ]

    unk_count = content_tokens.count(tokenizer.unk_token_id)
    total_tokens = len(content_tokens)

    if total_tokens == 0:
        return 0.0

    unk_rate = (unk_count / total_tokens) * 100

    return {
        "total_tokens": total_tokens,
        "unk_count": unk_count,
        "unk_rate_percent": unk_rate
    }


def main():
    tokenizer = BPETokenizer()

    corpus = "low lower lowest low lower"
    tokenizer.train(corpus, num_merges=5)
    tokenizer.build_vocabulary()

    known_result = calculate_unk_rate(tokenizer, "low lower")
    unknown_result = calculate_unk_rate(tokenizer, "low🙂")

    print("Known text:", known_result)
    print("Unknown character:", unknown_result)


if __name__ == "__main__":
    main()
