
import json
import torch
import pickle

from pathlib import Path
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function
)

from src.model.transformer import Transformer
from src.tokenization.tokenizer import BPETokenizer


# 1. Configuration


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "checkpoints/tinystories_smoke/best.pt"
)

TOKENIZER_PATH = "artifacts/tokenizer.pkl"

OUTPUT_PATH = "artifacts/inference_profile.json"

MAX_NEW_TOKENS = 20

WARMUP_RUNS = 2


# 2. Load tokenizer

def load_tokenizer():

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer



# 3. Load model

def load_model(tokenizer):

    model = Transformer(
        vocab_size=len(tokenizer.token_to_id),
        embedding_dim=64,
        max_sequence_length=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        hidden_dim=256,
        dropout=0.1
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# 4. Prepare prompts


def prepare_prompts():

    return [
        "Once upon a time",
        "A little girl",
        "The young boy",
        "One sunny day",
        "A small cat"
    ]

# 5. Generate text

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=MAX_NEW_TOKENS
):

    token_ids = tokenizer.encode_to_ids(
        prompt,
        add_special_tokens=True
    )

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = input_ids

    with torch.inference_mode():

        for _ in range(max_new_tokens):

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

            if (
                next_token_id.item()
                == tokenizer.token_to_id["[EOS]"]
            ):
                break

    return tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True
    )


# 6. Warmup

def warmup_model(model, tokenizer, prompts):

    print("\nStarting profiler warmup...")

    for run in range(WARMUP_RUNS):

        for prompt in prompts:

            generate_text(
                model,
                tokenizer,
                prompt
            )

        print(
            f"Warmup {run + 1}/{WARMUP_RUNS} completed"
        )


# 7. Profile inference

def profile_inference(model, tokenizer, prompts):

    print("\nStarting inference profiling...")

    activities = [
        ProfilerActivity.CPU
    ]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:

        for prompt_index, prompt in enumerate(prompts):

            with record_function(
                f"inference_prompt_{prompt_index + 1}"
            ):

                generate_text(
                    model,
                    tokenizer,
                    prompt
                )

    return prof


# 8. Extract profiler results

def extract_profiler_results(prof):

    events = prof.key_averages()

    sorted_events = sorted(
        events,
        key=lambda event: event.self_cpu_time_total,
        reverse=True
    )

    top_operations = []

    for event in sorted_events[:20]:

        top_operations.append({
            "operator_name": event.key,
            "calls": event.count,
            "self_cpu_time_microseconds": (
                event.self_cpu_time_total
            ),
            "total_cpu_time_microseconds": (
                event.cpu_time_total
            ),
            "self_cpu_memory_bytes": (
                event.self_cpu_memory_usage
            ),
            "total_cpu_memory_bytes": (
                event.cpu_memory_usage
            )
        })

    total_cpu_time = sum(
        event.self_cpu_time_total
        for event in events
    )

    return {
        "total_cpu_time_microseconds": total_cpu_time,
        "total_cpu_time_seconds": total_cpu_time / 1_000_000,
        "top_operations": top_operations
    }


# 9. Save results
def save_results(results):

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        f"\nProfiler results saved: "
        f"{output_path.resolve()}"
    )



# 10. Display results


def display_results(results):

    print("\n========== INFERENCE PROFILE ==========")

    print(
        "Total CPU time:",
        f"{results['total_cpu_time_seconds']:.6f}s"
    )

    print("\nTop 20 CPU operations:\n")

    for index, operation in enumerate(
        results["top_operations"],
        start=1
    ):

        print(
            f"{index}. {operation['operator_name']}"
        )

        print(
            f"   Calls: {operation['calls']}"
        )

        print(
            "   Self CPU time:",
            f"{operation['self_cpu_time_microseconds']:.2f} us"
        )

        print(
            "   Total CPU time:",
            f"{operation['total_cpu_time_microseconds']:.2f} us"
        )

        print(
            "   Self CPU memory:",
            f"{operation['self_cpu_memory_bytes']} bytes"
        )

        print()


# 11. Main

def main():

    print("Loading tokenizer...")

    tokenizer = load_tokenizer()

    print("Loading Transformer model...")

    model = load_model(tokenizer)

    prompts = prepare_prompts()

    print("\nModel configuration:")
    print("Device:", DEVICE)
    print("Vocabulary size:", len(tokenizer.token_to_id))
    print("Number of prompts:", len(prompts))

    warmup_model(
        model,
        tokenizer,
        prompts
    )

    profiler = profile_inference(
        model,
        tokenizer,
        prompts
    )

    results = extract_profiler_results(
        profiler
    )

    display_results(results)

    save_results(results)


if __name__ == "__main__":

    main()
