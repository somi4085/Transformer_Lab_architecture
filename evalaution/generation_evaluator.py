
import json
import time
from pathlib import Path

import sacrebleu
import torch
from rouge_score import rouge_scorer

from src.inference.generate import generate_text


# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "generation_metrics.json"
)


# TOKEN COUNT

def count_generated_tokens(tokenizer, text):

    if not text:
        return 0

    token_ids = tokenizer.encode_to_ids(
        text,
        add_special_tokens=False
    )

    return len(token_ids)


# REPETITION RATE

def calculate_repetition_rate(tokenizer, generated_texts):

    total_tokens = 0
    repeated_tokens = 0

    for text in generated_texts:

        token_ids = tokenizer.encode_to_ids(
            text,
            add_special_tokens=False
        )

        total_tokens += len(token_ids)

        unique_tokens = len(set(token_ids))

        repeated_tokens += (
            len(token_ids) - unique_tokens
        )

    if total_tokens == 0:
        return 0.0

    return round(
        repeated_tokens / total_tokens,
        4
    )


# FIRST TOKEN DIAGNOSTICS

def inspect_first_token_predictions(
    model,
    tokenizer,
    prompts,
    top_k=5,
    device="cpu"
):

    if not prompts:
        raise ValueError("prompts cannot be empty")

    device = torch.device(device)
    model = model.to(device)

    original_training_mode = model.training
    model.eval()

    token_to_id = tokenizer.token_to_id

    bos_token_id = token_to_id["[BOS]"]
    eos_token_id = token_to_id["[EOS]"]

    max_sequence_length = (
        model.encoder.input_embedding
        .positional_encoding.max_sequence_length
    )

    results = []

    try:

        with torch.inference_mode():

            for index, prompt in enumerate(prompts):

                source_ids = tokenizer.encode_to_ids(
                    prompt,
                    add_special_tokens=False
                )

                if not source_ids:
                    raise ValueError(
                        f"Prompt {index} produced no tokens"
                    )

                source_ids = [
                    bos_token_id,
                    *source_ids,
                    eos_token_id
                ]

                if len(source_ids) > max_sequence_length:
                    raise ValueError(
                        f"Prompt {index} exceeds maximum "
                        "source sequence length"
                    )

                source_token_ids = torch.tensor(
                    [source_ids],
                    dtype=torch.long,
                    device=device
                )

                source_mask = torch.ones_like(
                    source_token_ids
                )

                decoder_input_ids = torch.tensor(
                    [[bos_token_id]],
                    dtype=torch.long,
                    device=device
                )

                logits = model(
                    source_token_ids=source_token_ids,
                    target_token_ids=decoder_input_ids,
                    source_mask=source_mask,
                    target_mask=None
                )

                first_step_logits = logits[0, -1, :]

                probabilities = torch.softmax(
                    first_step_logits,
                    dim=-1
                )

                actual_top_k = min(
                    top_k,
                    probabilities.numel()
                )

                top_probabilities, top_token_ids = (
                    torch.topk(
                        probabilities,
                        k=actual_top_k
                    )
                )

                top_predictions = []

                for token_id, probability in zip(
                    top_token_ids.tolist(),
                    top_probabilities.tolist()
                ):

                    token = tokenizer.id_to_token.get(
                        token_id,
                        "[UNKNOWN]"
                    )

                    top_predictions.append({
                        "token_id": token_id,
                        "token": token,
                        "probability": round(
                            probability, 6
                        ),
                        "logit": round(
                            first_step_logits[token_id].item(),
                            6
                        )
                    })

                results.append({
                    "prompt_index": index,
                    "prompt": prompt,
                    "predicted_token_id": top_token_ids[0].item(),
                    "predicted_token": tokenizer.id_to_token.get(
                        top_token_ids[0].item(),
                        "[UNKNOWN]"
                    ),
                    "top_predictions": top_predictions
                })

    finally:
        model.train(original_training_mode)

    print("\n========== FIRST TOKEN DIAGNOSTICS ==========")

    for result in results:

        print(f"\nPrompt {result['prompt_index']}:")
        print(result["prompt"])

        print(
            "Top predicted token:",
            result["predicted_token"]
        )

        print("\nTop predictions:")

        for prediction in result["top_predictions"]:

            print(
                f"{prediction['token']:<15} "
                f"Probability: {prediction['probability']:.6f} "
                f"Logit: {prediction['logit']:.6f}"
            )

    unique_first_tokens = {
        result["predicted_token_id"]
        for result in results
    }

    print(
        "\nUnique first-token predictions:",
        len(unique_first_tokens)
    )

    

    return results


# BLEU SCORE

def calculate_bleu_score(references, generated_outputs):

    if len(references) != len(generated_outputs):
        raise ValueError(
            "References and generated outputs must have equal length."
        )

    if not references:
        raise ValueError(
            "References and generated outputs cannot be empty."
        )

    if not all(isinstance(text, str) for text in references):
        raise TypeError(
            "Every reference must be a string."
        )

    if not all(isinstance(text, str) for text in generated_outputs):
        raise TypeError(
            "Every generated output must be a string."
        )

    bleu_result = sacrebleu.corpus_bleu(
        generated_outputs,
        [references]
    )

    return {
        "bleu_score": round(
            bleu_result.score, 4
        ),
        "bleu_score_normalized": round(
            bleu_result.score / 100,
            4
        )
    }


# ROUGE SCORE


def calculate_rouge_score(references, generated_outputs):

    if len(references) != len(generated_outputs):
        raise ValueError(
            "References and generated outputs must have equal length."
        )

    if not references:
        raise ValueError(
            "References and generated outputs cannot be empty."
        )

    if not all(isinstance(text, str) for text in references):
        raise TypeError(
            "Every reference must be a string."
        )

    if not all(isinstance(text, str) for text in generated_outputs):
        raise TypeError(
            "Every generated output must be a string."
        )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    totals = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0
    }

    for reference, generated in zip(
        references,
        generated_outputs
    ):

        scores = scorer.score(
            reference,
            generated
        )

        for metric in totals:
            totals[metric] += scores[metric].fmeasure

    count = len(references)

    return {
        f"{metric}_f1": round(
            total / count, 4
        )
        for metric, total in totals.items()
    }


# GENERATION EVALUATOR

def evaluate_generation(
    model,
    tokenizer,
    prompts,
    references,
    max_new_tokens=20,
    device="cpu"
):

    if not prompts:
        raise ValueError(
            "Prompts cannot be empty."
        )

    if len(prompts) != len(references):
        raise ValueError(
            "Prompts and references must have equal length."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be positive."
        )

    if not all(isinstance(text, str) for text in prompts):
        raise TypeError("Every prompt must be a string.")

    if not all(isinstance(text, str) for text in references):
        raise TypeError("Every reference must be a string.")

    device = torch.device(device)
    model.to(device)

    original_training_mode = model.training
    model.eval()

    generated_outputs = []
    generation_times = []
    generated_token_counts = []

    try:

        for prompt in prompts:

            start_time = time.perf_counter()

            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                input_text=prompt,
                max_new_tokens=max_new_tokens,
                device=device
            )

            end_time = time.perf_counter()

            generation_time = end_time - start_time

            generated_outputs.append(generated_text)

            generation_times.append(
                generation_time
            )

            token_count = count_generated_tokens(
                tokenizer,
                generated_text
            )

            generated_token_counts.append(
                token_count
            )

    finally:
        model.train(original_training_mode)

    total_generation_time = sum(generation_times)

    total_generated_tokens = sum(
        generated_token_counts
    )

    total_prompts = len(prompts)

    average_latency = (
        total_generation_time / total_prompts
    )

    average_generated_length = (
        total_generated_tokens / total_prompts
    )

    throughput = (
        total_generated_tokens / total_generation_time
        if total_generation_time > 0
        else 0.0
    )

    empty_outputs = sum(
        not output.strip()
        for output in generated_outputs
    )

    metrics = {
        "total_prompts": total_prompts,

        "total_generated_tokens": total_generated_tokens,

        "average_generated_length": round(
            average_generated_length, 4
        ),

        "empty_outputs": empty_outputs,

        "empty_output_rate": round(
            empty_outputs / total_prompts, 4
        ),

        "repetition_rate": calculate_repetition_rate(
            tokenizer,
            generated_outputs
        ),

        "total_generation_time_seconds": round(
            total_generation_time, 6
        ),

        "average_latency_seconds": round(
            average_latency, 6
        ),

        "throughput_tokens_per_second": round(
            throughput, 4
        ),

        "max_new_tokens": max_new_tokens
    }

    # BLEU evaluation
    bleu_metrics = calculate_bleu_score(
        references,
        generated_outputs
    )

    metrics.update(bleu_metrics)

    # ROUGE evaluation
    rouge_metrics = calculate_rouge_score(
        references,
        generated_outputs
    )

    metrics.update(rouge_metrics)

    return metrics


# SAVE METRICS

def save_generation_metrics(
    metrics,
    output_path=METRICS_PATH
):

    output_path = Path(output_path)

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
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n--- GENERATION METRICS SAVED ---")
    print("File:", output_path)
    print(
        "Size:",
        output_path.stat().st_size,
        "bytes"
    )


# DISPLAY METRICS

def display_generation_metrics(metrics):

    

    print(
        "Total prompts:",
        metrics["total_prompts"]
    )

    print(
        "Total generated tokens:",
        metrics["total_generated_tokens"]
    )

    print(
        "Average generated length:",
        metrics["average_generated_length"],
        "tokens"
    )

    print(
        "Empty output rate:",
        metrics["empty_output_rate"]
    )

    print(
        "Repetition rate:",
        metrics["repetition_rate"]
    )

    print(
        "BLEU score:",
        metrics["bleu_score"]
    )

    print(
        "ROUGE-1 F1:",
        metrics["rouge1_f1"]
    )

    print(
        "ROUGE-2 F1:",
        metrics["rouge2_f1"]
    )

    print(
        "ROUGE-L F1:",
        metrics["rougeL_f1"]
    )

    print(
        "Average latency:",
        metrics["average_latency_seconds"],
        "seconds"
    )

    print(
        "Throughput:",
        metrics["throughput_tokens_per_second"],
        "tokens/second"
    )
