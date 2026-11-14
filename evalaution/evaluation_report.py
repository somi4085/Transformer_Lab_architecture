
import json
from pathlib import Path
from datetime import datetime, timezone


# 1. Project paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "generation_metrics.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation_report.json"
)


# 2. Load evaluation metrics

def load_metrics():

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        raise ValueError(
            "Evaluation metrics must be a JSON object."
        )

    return metrics


# 3. Model configuration

def create_model_configuration():

    return {
        "embedding_dim": 64,
        "max_sequence_length": 64,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "attention_heads": 4,
        "hidden_dim": 256,
        "dropout": 0.1,
        "device": "CPU"
    }


# 4. Generation quality

def extract_generation_quality(metrics):

    return {
        "bleu_score": metrics.get("bleu_score"),
        "bleu_score_normalized": metrics.get(
            "bleu_score_normalized"
        ),
        "rouge1_f1": metrics.get("rouge1_f1"),
        "rouge2_f1": metrics.get("rouge2_f1"),
        "rougeL_f1": metrics.get("rougeL_f1")
    }


# 5. Inference performance


def extract_inference_performance(metrics):

    return {
        "total_generation_time_seconds": metrics.get(
            "total_generation_time_seconds"
        ),
        "average_latency_seconds": metrics.get(
            "average_latency_seconds"
        ),
        "throughput_tokens_per_second": metrics.get(
            "throughput_tokens_per_second"
        ),
        "total_generated_tokens": metrics.get(
            "total_generated_tokens"
        )
    }


# 6. Generation behaviour

def extract_generation_behaviour(metrics):

    return {
        "total_prompts": metrics.get("total_prompts"),
        "average_generated_length": metrics.get(
            "average_generated_length"
        ),
        "empty_outputs": metrics.get("empty_outputs"),
        "empty_output_rate": metrics.get(
            "empty_output_rate"
        ),
        "repetition_rate": metrics.get(
            "repetition_rate"
        ),
        "max_new_tokens": metrics.get("max_new_tokens")
    }


# 7. Overall assessment


def create_overall_assessment(metrics):

    limitations = []

    bleu_score = metrics.get("bleu_score")
    rouge_score = metrics.get("rougeL_f1")

    if bleu_score is None:
        limitations.append(
            "BLEU score is unavailable."
        )

    if rouge_score is None:
        limitations.append(
            "ROUGE-L score is unavailable."
        )

    if metrics.get("empty_output_rate", 0) > 0:
        limitations.append(
            "Some prompts produced empty outputs."
        )

    if metrics.get("repetition_rate", 0) > 0:
        limitations.append(
            "Repeated tokens were detected in generated outputs."
        )

    if metrics.get("total_prompts", 0) < 100:
        limitations.append(
            "Evaluation dataset is small; results are preliminary."
        )

    if (
        bleu_score is not None
        and rouge_score is not None
        and bleu_score == 0
        and rouge_score == 0
    ):
        limitations.append(
            "Both lexical-overlap metrics are zero."
        )

    return {
        "evaluation_status": "completed",
        "assessment": (
            "Preliminary evaluation. "
            "Model quality should be interpreted using "
            "generation metrics and manual inspection."
        ),
        "limitations": limitations,
        "recommendations": [
            "Evaluate on a larger validation dataset.",
            "Inspect generated outputs manually.",
            "Compare model quality across training checkpoints.",
            "Benchmark inference under consistent conditions."
        ]
    }


# 8. Build evaluation report

def build_evaluation_report():

    metrics = load_metrics()

    report = {
        "report_metadata": {
            "project": "Transformer Architecture Laboratory",
            "report_generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_file": str(METRICS_PATH.name)
        },

        "model_configuration": create_model_configuration(),

        "generation_quality": extract_generation_quality(
            metrics
        ),

        "inference_performance": extract_inference_performance(
            metrics
        ),

        "generation_behaviour": extract_generation_behaviour(
            metrics
        ),

        "overall_assessment": create_overall_assessment(
            metrics
        )
    }

    return report


# 9. Save report

def save_evaluation_report(report):

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nEvaluation report saved successfully.")
    print("Report path:", REPORT_PATH)


# 10. Display report

def display_evaluation_report(report):

    print("\n========== EVALUATION REPORT ==========")

    for section, values in report.items():

        print(f"\n{section.upper()}")

        if isinstance(values, dict):

            for key, value in values.items():
                print(f"{key}: {value}")

        else:
            print(values)

    print("\n=======================================")


# 11. Main


def main():

    report = build_evaluation_report()

    display_evaluation_report(report)

    save_evaluation_report(report)


if __name__ == "__main__":
    main()
