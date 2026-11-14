import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pickle
import statistics
import time

import torch

from src.model.transformer import Transformer


# Configuration

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = Path("checkpoints/tinystories_smoke/best.pt")
TOKENIZER_PATH = Path("artifacts/tokenizer.pkl")

OUTPUT_PATH = Path("artifacts/kv_cache_benchmark.json")

MAX_NEW_TOKENS = 50
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10

BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0

PROMPTS = [
    "Once upon a time",
    "The little girl went to the forest",
    "A small dog was playing",
    "The king lived in a beautiful castle",
    "One day a little boy found"
]



# Load tokenizer

def load_tokenizer():

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


# Load model

def load_model():

    model = Transformer(
        vocab_size=266,
        embedding_dim=64,
        max_sequence_length=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        hidden_dim=256,
        dropout=0.0
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# Prepare input batch

def prepare_inputs(tokenizer):

    encoded_prompts = [
        tokenizer.encode_to_ids(
            prompt,
            add_special_tokens=True
        )
        for prompt in PROMPTS
    ]

    max_length = max(
        len(ids) for ids in encoded_prompts
    )

    source_token_ids = []
    source_mask = []

    for ids in encoded_prompts:

        padding_length = max_length - len(ids)

        padded_ids = (
            ids + [PAD_TOKEN_ID] * padding_length
        )

        mask = (
            [1] * len(ids) + [0] * padding_length
        )

        source_token_ids.append(padded_ids)
        source_mask.append(mask)

    source_token_ids = torch.tensor(
        source_token_ids,
        dtype=torch.long,
        device=DEVICE
    )

    source_mask = torch.tensor(
        source_mask,
        dtype=torch.long,
        device=DEVICE
    )

    return source_token_ids, source_mask


# Non-cached greedy decoding

@torch.inference_mode()
def generate_without_cache(
    model,
    encoder_output,
    max_new_tokens
):

    batch_size = encoder_output.size(0)

    generated_ids = torch.full(
        (batch_size, 1),
        BOS_TOKEN_ID,
        dtype=torch.long,
        device=DEVICE
    )

    first_token_time = None

    start_time = time.perf_counter()

    for step in range(max_new_tokens):

        decoder_output = model.decoder(
            token_ids=generated_ids,
            encoder_output=encoder_output
        )

        logits = model.output_projection(decoder_output)

        next_token = logits[:, -1, :].argmax(dim=-1)

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token.unsqueeze(1)
            ],
            dim=1
        )

        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time

    total_time = time.perf_counter() - start_time

    return {
        "generated_ids": generated_ids,
        "ttft": first_token_time,
        "total_time": total_time
    }


# Cached greedy decoding

@torch.inference_mode()
def generate_with_cache(
    model,
    encoder_output,
    max_new_tokens
):

    batch_size = encoder_output.size(0)

    generated_ids = torch.full(
        (batch_size, 1),
        BOS_TOKEN_ID,
        dtype=torch.long,
        device=DEVICE
    )

    past_key_values = None

    first_token_time = None

    start_time = time.perf_counter()

    for step in range(max_new_tokens):

        current_token = generated_ids[:, -1:]

        logits, past_key_values = model.decode_step(
            target_token_ids=current_token,
            encoder_output=encoder_output,
            past_key_values=past_key_values,
            position_offset=step
        )

        next_token = logits[:, -1, :].argmax(dim=-1)

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token.unsqueeze(1)
            ],
            dim=1
        )

        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time

    total_time = time.perf_counter() - start_time

    return {
        "generated_ids": generated_ids,
        "ttft": first_token_time,
        "total_time": total_time,
        "cache": past_key_values
    }



# Calculate KV cache memory

def calculate_cache_memory(cache):

    total_bytes = 0

    for layer_cache in cache:

        for attention_type in (
            "self_attention",
            "cross_attention"
        ):

            key, value = layer_cache[attention_type]

            total_bytes += key.numel() * key.element_size()
            total_bytes += value.numel() * value.element_size()

    return {
        "bytes": total_bytes,
        "kilobytes": total_bytes / 1024,
        "megabytes": total_bytes / (1024 ** 2)
    }


# Benchmark one decoding mode

def benchmark_mode(
    model,
    encoder_output,
    use_cache
):

    results = []

    for run in range(BENCHMARK_RUNS):

        if use_cache:

            output = generate_with_cache(
                model,
                encoder_output,
                MAX_NEW_TOKENS
            )

        else:

            output = generate_without_cache(
                model,
                encoder_output,
                MAX_NEW_TOKENS
            )

        generated_ids = output["generated_ids"]

        generated_token_count = (
            generated_ids.size(1) - 1
        )

        tokens_per_second = (
            generated_token_count
            * generated_ids.size(0)
            / output["total_time"]
        )

        run_result = {
            "ttft": output["ttft"],
            "total_time": output["total_time"],
            "tokens_per_second": tokens_per_second
        }

        if use_cache:

            run_result["cache_memory"] = (
                calculate_cache_memory(output["cache"])
            )

        results.append(run_result)

    return results


# Warmup

def warmup(model, encoder_output):

    print("\nRunning warmup...")

    for _ in range(WARMUP_RUNS):

        generate_without_cache(
            model,
            encoder_output,
            MAX_NEW_TOKENS
        )

        generate_with_cache(
            model,
            encoder_output,
            MAX_NEW_TOKENS
        )

    print("Warmup completed.")


# Calculate benchmark statistics

def calculate_statistics(results):

    ttft_values = [
        result["ttft"] for result in results
    ]

    latency_values = [
        result["total_time"] for result in results
    ]

    throughput_values = [
        result["tokens_per_second"] for result in results
    ]

    return {
        "average_ttft_seconds": statistics.mean(ttft_values),
        "average_latency_seconds": statistics.mean(latency_values),
        "median_latency_seconds": statistics.median(latency_values),
        "average_tokens_per_second": statistics.mean(throughput_values),
        "p95_latency_seconds": sorted(latency_values)[
            min(
                len(latency_values) - 1,
                int(0.95 * len(latency_values))
            )
        ]
    }


# Main benchmark

def main():

    print("Loading tokenizer...")

    tokenizer = load_tokenizer()

    print("Loading model...")

    model = load_model()

    print("Preparing inputs...")

    source_token_ids, source_mask = prepare_inputs(tokenizer)

    print("Source shape:", source_token_ids.shape)

    print("\nEncoding source...")

    with torch.inference_mode():

        encoder_start = time.perf_counter()

        encoder_output = model.encode(
            source_token_ids,
            source_mask=source_mask
        )

        encoder_time = (
            time.perf_counter() - encoder_start
        )

    print("Encoder output shape:", encoder_output.shape)
    print("Encoder time:", encoder_time)

    warmup(model, encoder_output)

    print("\nRunning non-cached benchmark...")

    non_cached_results = benchmark_mode(
        model,
        encoder_output,
        use_cache=False
    )

    print("Running cached benchmark...")

    cached_results = benchmark_mode(
        model,
        encoder_output,
        use_cache=True
    )

    print("\nChecking output equivalence...")

    reference_output = generate_without_cache(
        model,
        encoder_output,
        MAX_NEW_TOKENS
    )

    cached_output = generate_with_cache(
        model,
        encoder_output,
        MAX_NEW_TOKENS
    )

    outputs_match = torch.equal(
        reference_output["generated_ids"],
        cached_output["generated_ids"]
    )

    if not outputs_match:

        raise AssertionError(
            "Cached and non-cached generated tokens differ."
        )

    print("Generated tokens match.")

    non_cached_stats = calculate_statistics(
        non_cached_results
    )

    cached_stats = calculate_statistics(
        cached_results
    )

    speedup = (
        non_cached_stats["average_latency_seconds"]
        / cached_stats["average_latency_seconds"]
    )

    cache_memory = calculate_cache_memory(
        cached_output["cache"]
    )

    report = {
        "configuration": {
            "device": str(DEVICE),
            "batch_size": source_token_ids.size(0),
            "max_new_tokens": MAX_NEW_TOKENS,
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS
        },

        "encoder_time_seconds": encoder_time,

        "non_cached": non_cached_stats,

        "cached": cached_stats,

        "kv_cache_memory": cache_memory,

        "speedup": speedup,

        "generated_tokens_match": outputs_match
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n========== KV CACHE BENCHMARK ==========")

    print(
        "Non-cached latency:",
        non_cached_stats["average_latency_seconds"]
    )

    print(
        "Cached latency:",
        cached_stats["average_latency_seconds"]
    )

    print(
        "Non-cached throughput:",
        non_cached_stats["average_tokens_per_second"]
    )

    print(
        "Cached throughput:",
        cached_stats["average_tokens_per_second"]
    )

    print("Speedup:", speedup)

    print(
        "KV Cache memory:",
        cache_memory["kilobytes"],
        "KB"
    )

    print("Generated tokens match:", outputs_match)

    print("\nReport saved:", OUTPUT_PATH)


if __name__ == "__main__":

    main()
