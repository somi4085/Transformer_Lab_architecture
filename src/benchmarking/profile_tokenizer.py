
import cProfile
import pstats

from src.tokenization.tokenizer import BPETokenizer
from src.benchmarking.latency_benchmark import (
    create_tokenizer,
    generate_input_text
)


def profile_tokenizer():

    tokenizer = create_tokenizer()

    # Large input for profiling
    text = generate_input_text(10000)

    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    tokenizer.encode_to_ids(text)

    # Stop profiling
    profiler.disable()

    # Display profiling results
    stats = pstats.Stats(profiler)

    stats.strip_dirs()
    stats.sort_stats("cumulative")

    stats.print_stats(20)


if __name__ == "__main__":
    profile_tokenizer()
