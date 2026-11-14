
class BPETokenizer:

    def __init__(self):
        self.word_frequencies = {}
        self.vocabulary = {}
        self.merges = {}

        self.token_to_id = {}
        self.id_to_token = {}

        # Special tokens
        self.pad_token = "[PAD]"
        self.pad_token_id = None

        self.unk_token = "[UNK]"
        self.unk_token_id = None

        self.bos_token = "[BOS]"
        self.bos_token_id = None

        self.eos_token = "[EOS]"
        self.eos_token_id = None

    def train(self, corpus, num_merges=10):
        self.word_frequencies = {}
        self.vocabulary = {}
        self.merges = {}

        words = corpus.split()

        # Step 1: Count word frequencies
        for word in words:
            if word in self.word_frequencies:
                self.word_frequencies[word] += 1
            else:
                self.word_frequencies[word] = 1

        # Step 2: Create character-level vocabulary
        for word, freq in self.word_frequencies.items():
            characters = list(word)
            characters.append("</w>")

            token_sequence = tuple(characters)
            self.vocabulary[token_sequence] = freq

        # Step 3: Perform BPE merges
        for merge_step in range(num_merges):
            pair = self.get_most_frequent_pair()

            if pair is None:
                break

            merged_token = pair[0] + pair[1]

            self.merge_pair(pair)
            self.merges[pair] = merged_token

        return self.vocabulary

    def get_pair_frequency(self):
        pair_frequencies = {}

        for token_sequence, frequency in self.vocabulary.items():
            for i in range(len(token_sequence) - 1):
                pair = (token_sequence[i], token_sequence[i + 1])

                if pair in pair_frequencies:
                    pair_frequencies[pair] += frequency
                else:
                    pair_frequencies[pair] = frequency

        return pair_frequencies

    def merge_pair(self, pair):
        new_vocabulary = {}

        for token_sequence, frequency in self.vocabulary.items():
            new_sequence = []
            i = 0

            while i < len(token_sequence):
                if i < len(token_sequence) - 1:
                    current_pair = (
                        token_sequence[i],
                        token_sequence[i + 1]
                    )

                    if current_pair == pair:
                        merged_token = (
                            token_sequence[i] + token_sequence[i + 1]
                        )

                        new_sequence.append(merged_token)
                        i += 2
                        continue

                new_sequence.append(token_sequence[i])
                i += 1

            new_sequence = tuple(new_sequence)

            if new_sequence in new_vocabulary:
                new_vocabulary[new_sequence] += frequency
            else:
                new_vocabulary[new_sequence] = frequency

        self.vocabulary = new_vocabulary

        return self.vocabulary

    def get_most_frequent_pair(self):
        pair_frequencies = self.get_pair_frequency()

        if not pair_frequencies:
            return None

        return max(
            pair_frequencies,
            key=pair_frequencies.get
        )

    def encode_word(self, word):
        tokens = list(word) + ["</w>"]

        # Apply learned BPE merges
        for pair, merged_token in self.merges.items():
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1:
                    current_pair = (
                        tokens[i],
                        tokens[i + 1]
                    )

                    if current_pair == pair:
                        new_tokens.append(merged_token)
                        i += 2
                        continue

                new_tokens.append(tokens[i])
                i += 1

            tokens = new_tokens

        return tuple(tokens)

    def encode(self, text, add_special_tokens=True):
        encoded_tokens = []

        words = text.split()

        if add_special_tokens:
            encoded_tokens.append(self.bos_token)

        for word in words:
            word_tokens = self.encode_word(word)
            encoded_tokens.extend(word_tokens)

        if add_special_tokens:
            encoded_tokens.append(self.eos_token)

        return tuple(encoded_tokens)

    def build_vocabulary(self):
        self.token_to_id = {}
        self.id_to_token = {}

        # Add special tokens first
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        for token in special_tokens:
            next_id = len(self.token_to_id)

            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token

        # Add learned tokens
        for sequence in self.vocabulary:
            for token in sequence:
                if token not in self.token_to_id:
                    next_id = len(self.token_to_id)

                    self.token_to_id[token] = next_id
                    self.id_to_token[next_id] = token

        # Store special token IDs
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]

        return self.token_to_id, self.id_to_token

    def encode_to_ids(self, text, add_special_tokens=True):
        tokens = self.encode(
            text,
            add_special_tokens=add_special_tokens
        )

        token_ids = []

        for token in tokens:
            token_id = self.token_to_id.get(
                token,
                self.unk_token_id
            )

            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        words = []
        current_word = ""

        special_tokens = {
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        }

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")

            token = self.id_to_token[token_id]

            if skip_special_tokens and token in special_tokens:
                continue

            if token.endswith("</w>"):
                current_word += token[:-4]
                words.append(current_word)
                current_word = ""
            else:
                current_word += token

        if current_word:
            words.append(current_word)

        return " ".join(words)
    

    def pad_sequence(self,token_ids, max_length):

        if max_length < 0:
            raise ValueError("max length cannot be negative")
        if len(token_ids) > max_length:
            raise ValueError("sequence length exceeds max_length")

        padding_length = max_length - len(token_ids)

        padded_sequence = (
            token_ids + [self.pad_token_id]* padding_length
        )
        return padded_sequence

    def truncate_sequence(self,token_ids, max_length):
        if max_length < 0:
            raise ValueError("max length cannot be negative")
        if len(token_ids) <= max_length:
            return token_ids

        has_bos = token_ids[0] == self.bos_token_id
        has_eos = token_ids[-1] == self.eos_token_id

        if has_bos and has_eos:
            if max_length < 2:
                raise ValueError("max length must be at least preserve BOS and EOS")

            truncated_sequence = (
                [self.bos_token_id] 
                + token_ids[1:max_length-1]
                + [self.eos_token_id]
            )
        elif has_bos:
            truncated_sequence = (
                token_ids[:max_length-1]
                + [self.eos_token_id]
            )

        else:
            truncated_sequence = token_ids[:max_length]

        return truncated_sequence

    def create_attention_mask(self,token_ids):
        attention_mask = []

        for token_id in token_ids:
            if token_id == self.pad_token_id:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return attention_mask

    def batch_encode(
    self,
    texts,
    padding=True,
    truncation=False,
    max_length=None,
    add_special_tokens=True
):
        
        # Step 0: Validate texts
        if texts is None or isinstance(texts, str):
            raise TypeError("texts must be a list or sequence of strings.")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every item in texts must be a string.")

        # Step 1: Validate max_length
        if max_length is not None:
            if not isinstance(max_length, int) or isinstance(max_length, bool):
                raise TypeError("max_length must be an integer.")

            if max_length < 0:
                raise ValueError("max_length cannot be negative.")

        # Step 2: Handle empty batch
        if len(texts) == 0:
            return {
                "input_ids": [],
                "attention_mask": []
            }
        
        batch_input_ids = []

        # Step 1: Encode every text
        for text in texts:
            token_ids = self.encode_to_ids(
                text,
                add_special_tokens=add_special_tokens
            )

            # Step 2: Apply truncation if required
            if truncation and max_length is not None:
                token_ids = self.truncate_sequence(
                    token_ids,
                    max_length
                )

            batch_input_ids.append(token_ids)

        # Step 3: Determine padding length
        if padding:
            if max_length is not None:
                target_length = max_length
            else:
                target_length = max(
                    len(sequence) for sequence in batch_input_ids
                )

            # Step 4: Pad every sequence
            batch_input_ids = [
                self.pad_sequence(sequence, target_length)
                for sequence in batch_input_ids
            ]

        # Step 5: Generate attention masks
        batch_attention_mask = [
            self.create_attention_mask(sequence)
            for sequence in batch_input_ids
        ]

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask
        }


# =====================================
# TESTING
# =====================================

tokenizer = BPETokenizer()

# Step 1: Training corpus
corpus = "low lower lowest low lower"

# Step 2: Train tokenizer
tokenizer.train(corpus, num_merges=5)

# Step 3: Build vocabulary
tokenizer.build_vocabulary()

text = "low lower"

# Test special token IDs
print("\n--- Special Token IDs ---")

print("PAD:", tokenizer.pad_token_id)
print("UNK:", tokenizer.unk_token_id)
print("BOS:", tokenizer.bos_token_id)
print("EOS:", tokenizer.eos_token_id)

# Test encoding with special tokens
print("\n--- Encoding Test ---")

tokens_with_special = tokenizer.encode(text)
ids_with_special = tokenizer.encode_to_ids(text)

print("With special tokens:", tokens_with_special)
print("With special token IDs:", ids_with_special)

# Test encoding without special tokens
tokens_without_special = tokenizer.encode(
    text,
    add_special_tokens=False
)

ids_without_special = tokenizer.encode_to_ids(
    text,
    add_special_tokens=False
)

print("Without special tokens:", tokens_without_special)
print("Without special token IDs:", ids_without_special)

# Test decoding
print("\n--- Decoding Test ---")

decoded_text = tokenizer.decode(ids_without_special)

print("Original text:", text)
print("Decoded text:", decoded_text)

# Test vocabulary
print("\n--- Vocabulary ---")

print("Token to ID:", tokenizer.token_to_id)
print("ID to Token:", tokenizer.id_to_token)


print("\n--- Special Token Decoding Test ---")

# Encode with BOS and EOS
encoded_ids = tokenizer.encode_to_ids("low lower")

print("Encoded IDs:", encoded_ids)

# Decode while skipping special tokens
decoded_skip = tokenizer.decode(
    encoded_ids,
    skip_special_tokens=True
)

print("Skip special tokens:", decoded_skip)

# Decode while retaining special tokens
decoded_keep = tokenizer.decode(
    encoded_ids,
    skip_special_tokens=False
)

print("Keep special tokens:", decoded_keep)


print("\n--- Padding Test ---")

sequence = tokenizer.encode_to_ids(
    "low",
    add_special_tokens=True
)

print("Original sequence:", sequence)
print("Original length:", len(sequence))

padded_sequence = tokenizer.pad_sequence(
    sequence,
    max_length=5
)

print("Padded sequence:", padded_sequence)
print("Padded length:", len(padded_sequence))


print("\n--- Truncation Test ---")

sequence = tokenizer.encode_to_ids(
    "low lower lowest",
    add_special_tokens=True
)

print("Original sequence:", sequence)
print("Original length:", len(sequence))

truncated_sequence = tokenizer.truncate_sequence(
    sequence,
    max_length=5
)

print("Truncated sequence:", truncated_sequence)
print("Truncated length:", len(truncated_sequence))

print("First token:", tokenizer.id_to_token[truncated_sequence[0]])
print("Last token:", tokenizer.id_to_token[truncated_sequence[-1]])


print("\n--- Attention Mask Test ---")

sequence = tokenizer.encode_to_ids(
    "low",
    add_special_tokens=True
)

padded_sequence = tokenizer.pad_sequence(
    sequence,
    max_length=5
)

attention_mask = tokenizer.create_attention_mask(
    padded_sequence
)

print("Token IDs:", padded_sequence)
print("Attention Mask:", attention_mask)
print("Sequence Length:", len(padded_sequence))
print("Mask Length:", len(attention_mask))




texts = [
    "low",
    "lower"
]

result = tokenizer.batch_encode(
    texts,
    padding=True,
    truncation=False,
    add_special_tokens=True
)

print("Input IDs:", result["input_ids"])
print("Attention Mask:", result["attention_mask"])



# Test 1 — Fixed-length padding

result = tokenizer.batch_encode(
    ["low", "lower"],
    padding=True,
    max_length=8
)

print(result)


# Test 2 — Truncation
result = tokenizer.batch_encode(
    ["low", "lower", "lowest"],
    padding=True,
    truncation=True,
    max_length=5
)

print(result)


# padding disabled
result = tokenizer.batch_encode(
    ["low", "lower"],
    padding=False
)

print(result)


# empty batch
result = tokenizer.batch_encode([])
print(result)




result = tokenizer.batch_encode(
    ["low", "lower", "lowest"],
    padding=False,
    truncation=True,
    max_length=4
)
print("Truncation without padding: ",result)


result = tokenizer.batch_encode(
    ["low", "lower"],
    padding=True,
    truncation=False,
    max_length=8
)
print("Fixed padding without truncation: ",result)



result = tokenizer.batch_encode(
    ["low", "lower", "lowest"],
    padding=True,
    truncation=True,
    max_length=6
)
print("Truncation with fixed padding: ",result)


# ==========================================
# INPUT VALIDATION TESTS
# ==========================================

print("\n--- Input Validation Tests ---")

# Test 1: None input
try:
    tokenizer.batch_encode(None)
except TypeError as e:
    print("Test 1 PASS:", e)


# Test 2: Single string instead of list
try:
    tokenizer.batch_encode("low")
except TypeError as e:
    print("Test 2 PASS:", e)


# Test 3: Non-string item inside list
try:
    tokenizer.batch_encode(["low", 123])
except TypeError as e:
    print("Test 3 PASS:", e)


# Test 4: Negative max_length
try:
    tokenizer.batch_encode(
        ["low", "lower"],
        max_length=-1
    )
except ValueError as e:
    print("Test 4 PASS:", e)


# Test 5: Float max_length
try:
    tokenizer.batch_encode(
        ["low", "lower"],
        max_length=2.5
    )
except TypeError as e:
    print("Test 5 PASS:", e)


print("\n--- All Validation Tests Completed ---")