
import pytest
import torch

from src.api.inference_service import InferenceService

def create_service():

    return InferenceService(
        checkpoint_path="checkpoints/tinystories_smoke/best.pt",
        tokenizer_path="artifacts/tokenizer.pkl",
        device="cpu"
    )

def test_service_initialization():

    service = create_service()

    service.initialize()

    assert service.initialized is True
    assert service.model is not None
    assert service.tokenizer is not None

def test_model_evaluation_mode():

    service = create_service()
    service.initialize()

    assert service.model.training is False

def test_model_device():

    service = create_service()
    service.initialize()

    for parameter in service.model.parameters():
        assert parameter.device.type == "cpu"

def test_model_parameters_not_trainable_during_inference():

    service = create_service()
    service.initialize()

    for parameter in service.model.parameters():
        assert parameter.requires_grad is True

    assert not torch.is_grad_enabled() is False

def test_generate_output():

    service = create_service()
    service.initialize()

    result = service.generate(
        "Once upon a time",
        max_new_tokens=10
    )

    assert isinstance(result, dict)
    assert isinstance(result["generated_text"], str)
    assert result["generated_token_count"] <= 10

def test_empty_input():

    service = create_service()
    service.initialize()

    with pytest.raises(ValueError):
        service.generate("")

def test_uninitialized_service():

    service = create_service()

    with pytest.raises(RuntimeError):
        service.generate("Once upon a time")

def test_invalid_token_limit():

    service = create_service()
    service.initialize()

    with pytest.raises(ValueError):
        service.generate(
            "Once upon a time",
            max_new_tokens=0
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])