

import json
import time
import torch

from pathlib import Path

from src.model.transformer import Transformer
from src.data.dataset import prepare_dataset


def count_generated_tokens(tokenizer, text):
    token_ids = tokenizer.encode_to_ids(
        text,
        add_special_tokens=False
    )
    return len(token_ids)


def calculate_percentiles(latencies):
    sorted_latencies = sorted(latencies)

    def percentile(p):
        index = int((len(sorted_latencies) - 1) * p)
        return sorted_latencies[index]

    return {
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95)
    }


def generate_text(model, tokenizer, prompt, device, max_new_tokens=20):
    token_ids = tokenizer.encode_to_ids(
        prompt,
        add_special_tokens=True
    )

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    generated_ids = input_ids

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits = model(
                generated_ids,
                generated_ids
            )

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=1
        )

        if next_token_id.item() == tokenizer.token_to_id["[EOS]"]:
            break

    return tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True
    )


def measure_latency(
    model,
    tokenizer,
    prompts,
    device,
    max_new_tokens=20,
    warmup_runs=3,
    benchmark_runs=5
):
    model.eval()

    for _ in range(warmup_runs):
        for prompt in prompts:
            generate_text(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens
            )

    latencies = []
    total_generated_tokens = 0
    total_time = 0.0

    for run in range(benchmark_runs):
        run_start = time.perf_counter()

        for prompt in prompts:
            start = time.perf_counter()

            generated_text = generate_text(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens
            )

            end = time.perf_counter()

            latencies.append(end - start)

            total_generated_tokens += count_generated_tokens(
                tokenizer,
                generated_text
            )

        run_time = time.perf_counter() - run_start
        total_time += run_time

        print(
            f"Benchmark {run + 1}/{benchmark_runs} | "
            f"Prompts: {len(prompts)} | "
            f"Time: {run_time:.4f}s"
        )

    total_requests = len(prompts) * benchmark_runs

    return {
        "total_requests": total_requests,
        "total_generated_tokens": total_generated_tokens,
        "average_latency_seconds": sum(latencies) / len(latencies),
        "percentiles": calculate_percentiles(latencies),
        "tokens_per_second": total_generated_tokens / total_time,
        "requests_per_second": total_requests / total_time,
        "total_generation_time_seconds": total_time
    }


def generate_batch(
    model,
    tokenizer,
    prompts,
    device,
    max_new_tokens=20
):
    encoded_prompts = [
        tokenizer.encode_to_ids(
            prompt,
            add_special_tokens=True
        )
        for prompt in prompts
    ]

    max_length = max(len(ids) for ids in encoded_prompts)

    pad_id = tokenizer.token_to_id["[PAD]"]

    padded_ids = [
        ids + [pad_id] * (max_length - len(ids))
        for ids in encoded_prompts
    ]

    input_ids = torch.tensor(
        padded_ids,
        dtype=torch.long,
        device=device
    )

    attention_mask = (input_ids != pad_id).long()

    generated_ids = input_ids

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits = model(
                generated_ids,
                generated_ids,
                source_mask=attention_mask,
                target_mask=attention_mask
            )

        next_token_logits = logits[:, -1, :]

        next_token_ids = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        generated_ids = torch.cat(
            [generated_ids, next_token_ids],
            dim=1
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(next_token_ids)
            ],
            dim=1
        )

        if torch.all(
            next_token_ids.squeeze(-1)
            == tokenizer.token_to_id["[EOS]"]
        ):
            break

    return [
        tokenizer.decode(
            sequence.tolist(),
            skip_special_tokens=True
        )
        for sequence in generated_ids
    ]


def measure_batch_performance(
    model,
    tokenizer,
    prompts,
    device,
    batch_sizes=(1, 2, 4, 8),
    max_new_tokens=20,
    warmup_runs=2,
    benchmark_runs=5
):
    model.eval()

    batch_results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")

        latencies = []
        total_generated_tokens = 0
        total_time = 0.0
        total_requests = 0

        batches = [
            prompts[i:i + batch_size]
            for i in range(0, len(prompts), batch_size)
        ]

        for _ in range(warmup_runs):
            for batch in batches:
                generate_batch(
                    model,
                    tokenizer,
                    batch,
                    device,
                    max_new_tokens
                )

        for run in range(benchmark_runs):
            run_start = time.perf_counter()

            for batch in batches:
                start = time.perf_counter()

                generated_texts = generate_batch(
                    model,
                    tokenizer,
                    batch,
                    device,
                    max_new_tokens
                )

                end = time.perf_counter()

                latencies.append(end - start)

                total_requests += len(batch)

                for generated_text in generated_texts:
                    total_generated_tokens += count_generated_tokens(
                        tokenizer,
                        generated_text
                    )

            run_time = time.perf_counter() - run_start
            total_time += run_time

            print(
                f"Run {run + 1}/{benchmark_runs} | "
                f"Batch size: {batch_size} | "
                f"Time: {run_time:.4f}s"
            )

        batch_results[str(batch_size)] = {
            "batch_size": batch_size,
            "total_requests": total_requests,
            "total_generated_tokens": total_generated_tokens,
            "average_batch_latency_seconds": (
                sum(latencies) / len(latencies)
            ),
            "average_request_latency_seconds": (
                sum(latencies) / total_requests
            ),
            "tokens_per_second": (
                total_generated_tokens / total_time
            ),
            "requests_per_second": (
                total_requests / total_time
            ),
            "total_generation_time_seconds": total_time
        }

    return batch_results


def save_benchmark_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print(f"\nBenchmark results saved: {output_path.resolve()}")


def main():
    device = torch.device("cpu")

    tokenizer, train_dataset, validation_dataset, train_loader, validation_loader = (
        prepare_dataset()
    )

    vocab_size = len(tokenizer.token_to_id)

    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=64,
        max_sequence_length=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        hidden_dim=256,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(
        "checkpoints/tinystories_smoke/best.pt",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompts = [
        "Once upon a time",
        "A little girl",
        "The young boy",
        "One sunny day",
        "A small cat"
    ] * 4

    print("\nStarting single-request benchmark...")

    single_results = measure_latency(
        model,
        tokenizer,
        prompts,
        device
    )

    print("\nStarting batch-size benchmark...")

    batch_results = measure_batch_performance(
        model,
        tokenizer,
        prompts,
        device,
        batch_sizes=(1, 2, 4, 8)
    )

    results = {
        "configuration": {
            "device": str(device),
            "prompts": len(prompts),
            "max_new_tokens": 20,
            "batch_sizes": [1, 2, 4, 8]
        },
        "single_request_results": single_results,
        "batch_results": batch_results
    }

    save_benchmark_results(
        results,
        "artifacts/performance_benchmark.json"
    )

    print("\n========== BATCH BENCHMARK SUMMARY ==========")

    for batch_size, metrics in batch_results.items():
        print(
            f"Batch size: {batch_size} | "
            f"Average batch latency: "
            f"{metrics['average_batch_latency_seconds']:.4f}s | "
            f"Requests/sec: "
            f"{metrics['requests_per_second']:.2f} | "
            f"Tokens/sec: "
            f"{metrics['tokens_per_second']:.2f}"
        )


if __name__ == "__main__":
    main()
