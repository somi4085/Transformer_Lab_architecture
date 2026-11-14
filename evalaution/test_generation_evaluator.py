
from evalaution.generation_evaluator import (
    calculate_bleu_score
)


def test_calculate_bleu_score():

    references = [
        "The little boy is playing in the park",
        "A cat is sitting on the table"
    ]

    generated_outputs = [
        "The boy is playing in the park",
        "A cat sits on the table"
    ]

    result = calculate_bleu_score(
        references,
        generated_outputs
    )

    assert 0 <= result["bleu_score"] <= 100

    assert 0 <= result["bleu_score_normalized"] <= 1


def test_bleu_rejects_unequal_lengths():

    references = ["The cat is sleeping"]

    generated_outputs = [
        "The cat sleeps",
        "A dog runs"
    ]

    import pytest

    with pytest.raises(ValueError):
        calculate_bleu_score(
            references,
            generated_outputs
        )



import pytest

from evalaution.generation_evaluator import (
    calculate_rouge_score
)


def test_calculate_rouge_score():

    references = [
        "The little boy is playing in the park",
        "A cat is sitting on the table"
    ]

    generated_outputs = [
        "The boy is playing in the park",
        "A cat sits on the table"
    ]

    result = calculate_rouge_score(
        references,
        generated_outputs
    )

    assert 0 <= result["rouge1_f1"] <= 1
    assert 0 <= result["rouge2_f1"] <= 1
    assert 0 <= result["rougeL_f1"] <= 1


def test_rouge_rejects_unequal_lengths():

    references = ["The cat is sleeping"]

    generated_outputs = [
        "The cat sleeps",
        "A dog runs"
    ]

    with pytest.raises(ValueError):
        calculate_rouge_score(
            references,
            generated_outputs
        )


def test_rouge_rejects_empty_inputs():

    with pytest.raises(ValueError):
        calculate_rouge_score([], [])
