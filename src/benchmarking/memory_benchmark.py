import tracemalloc
from src.benchmarking.latency_benchmark import (
    create_tokenizer,
    generate_input_text
)

def benchmarking_memory(tokenizer,text):
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    token_ids = tokenizer.encode_to_ids(text)
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    snapshot_after = tracemalloc.take_snapshot()
    memory_diff = snapshot_after.compare_to(
        snapshot_before,
        "lineno"
    )
    tracemalloc.stop()
    return {
        "total_tokens":len(token_ids),
        "current_memory_mb":current_memory/ (1024*1024),
        "peak_memory_mb":peak_memory / (1024*1024),
        "memory_diff":memory_diff
    }

def run_memory_benchmark():
    tokenizer = create_tokenizer()
    input_sizes = [100,1000,10000]
    for size in input_sizes:
        text = generate_input_text(size)
        results = benchmarking_memory(tokenizer,text)

        print(f"\n Input words: {size}")
        print(f"Total tokens: {results['total_tokens']}")
        print(
            f"current memory: "
            f"{results['current_memory_mb']:.4f} MB"
        )
        print(
        f"Peak memory: "
        f"{results['peak_memory_mb']:.4f} MB"
    )

    print("\nTop memory allocations:")

    for stat in results["memory_diff"][:5]:
        print(stat)


if __name__ == "__main__":
    run_memory_benchmark()
