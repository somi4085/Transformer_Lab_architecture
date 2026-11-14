import numpy as np
import time
import cProfile
import pstats
import statistics

from src.tokenization.tokenizer import BPETokenizer



# Words used to generate benchmark data
WORD_VOCABULARY = [
    "low",
    "lower",
    "lowest",
    "learning",
    "tokenizer",
    "benchmark",
    "transformer",
    "model",
    "data",
    "system",
    "python",
    "language"
]


def generate_corpus():
    """
    Generate training corpus using our vocabulary.
    """

    corpus = " ".join(WORD_VOCABULARY * 20)

    return corpus


def create_tokenizer():

    tokenizer = BPETokenizer()

    corpus = generate_corpus()

    tokenizer.train(corpus, num_merges=30)
    tokenizer.build_vocabulary()

    return tokenizer


def generate_input_text(target_words):
    """
    Generate exact number of words automatically.
    """

    generated_words = []

    for i in range(target_words):
        word = WORD_VOCABULARY[i % len(WORD_VOCABULARY)]
        generated_words.append(word)

    return " ".join(generated_words)


def benchmark_latency(
    tokenizer,
    text,
    warmup_runs=100,
    measurement_runs=1000
):

    if measurement_runs <= 0:
        raise ValueError("measurement_runs must be greater than zero")

    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative")

    # Warm-up runs
    for _ in range(warmup_runs):
        tokenizer.encode_to_ids(text)

    latency_times = []

    # Actual measurements
    for _ in range(measurement_runs):

        start_time = time.perf_counter()

        tokenizer.encode_to_ids(text)

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        latency_times.append(latency_ms)
    sorted_latencies = sorted(latency_times)
    average_latency = statistics.mean(latency_times)
    median_latency = statistics.median(latency_times)
    minimum_latency = min(latency_times)
    maximum_latency = max(latency_times)


    p50_latency = np.percentile(sorted_latencies,50)
    p95_latency = np.percentile(sorted_latencies,95)
    p99_latency = np.percentile(sorted_latencies,99)


    return {
        "average_latency_ms": average_latency,
        "median_latency_ms": median_latency,
        "minimum_latency_ms": minimum_latency,
        "maximum_latency_ms": maximum_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency": p95_latency,
        "p99_latency": p99_latency,
        "total_runs": measurement_runs
    }


def run_multiple_input_benchmark(tokenizer):

    input_sizes = {
        100: 1000,
        1000: 300,
        10000: 100,
        100000: 20
    }

    print("\n--- Multiple Input Sizes Benchmark ---")

    for target_words, measurement_runs in input_sizes.items():

        # Generate input automatically
        text = generate_input_text(target_words)

        # Verify word count
        actual_word_count = len(text.split())

        # Verify token count
        token_ids = tokenizer.encode_to_ids(text)
        actual_token_count = len(token_ids)

        print(f"Input_size: {target_words} words")
        print(f"Actual words: {actual_word_count} words")
        print(f"actual_token_count: {actual_token_count} words")

        # Run latency benchmark
        results = benchmark_latency(
            tokenizer=tokenizer,
            text=text,
            warmup_runs=100,
            measurement_runs=measurement_runs
        )

        for key,value in results.items():
            if key == "total_runs":
                print(f"Measurement runs: {value}")
            else:
                print(f"{key}:{value:.4f} ms")

        
        throughput_results = benchmark_throughput(
            tokenizer=tokenizer,
            text=text,
            measurement_runs=100
        )

        print("\nThroughput Results:")

        for key, value in throughput_results.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float)
                  else f"{key}: {value}")



def benchmark_throughput(tokenizer, text, measurement_runs=100):

    if measurement_runs <= 0:
        raise ValueError("measurement_runs must be greater than zero")

    # Warm-up
    for _ in range(10):
        tokenizer.encode_to_ids(text)

    total_tokens_processed = 0

    start_time = time.perf_counter()

    for _ in range(measurement_runs):
        token_ids = tokenizer.encode_to_ids(text)
        total_tokens_processed += len(token_ids)

    end_time = time.perf_counter()

    total_time_seconds = end_time - start_time

    throughput_tokens_per_second = (
        total_tokens_processed / total_time_seconds
    )

    return {
        "total_tokens_processed": total_tokens_processed,
        "total_time_seconds": total_time_seconds,
        "throughput_tokens_per_second": throughput_tokens_per_second,
        "measurement_runs": measurement_runs
    }






if __name__ == "__main__":

    tokenizer = create_tokenizer()

    run_multiple_input_benchmark(tokenizer)
