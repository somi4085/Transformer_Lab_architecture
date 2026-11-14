
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.api.schemas.generation import (
    GenerationRequest,
    GenerationResponse
)

from src.api.inference_service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Transformer API...")

    inference_service = InferenceService()

    try:
        inference_service.initialize()

        app.state.inference_service = inference_service

        print("Transformer model loaded successfully.")

        yield

    finally:

        app.state.inference_service = None

        inference_service.model = None
        inference_service.tokenizer = None
        inference_service.initialized = False

        print("Transformer API shutdown completed.")


app = FastAPI(
    title="Transformer Architecture Laboratory",
    description="Local Transformer inference API with KV Cache",
    version="1.0.0",
    lifespan=lifespan
)



# Health endpoint

@app.get("/health")
async def health_check(request: Request):

    inference_service = getattr(
        request.app.state,
        "inference_service",
        None
    )

    if (
        inference_service is None
        or not inference_service.initialized
    ):

        raise HTTPException(
            status_code=503,
            detail="Transformer model is not ready."
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(inference_service.device)
    }


# ----------------------------------------
# Generation endpoint
# ----------------------------------------

@app.post(
    "/generate",
    response_model=GenerationResponse
)
async def generate_text(
    payload: GenerationRequest,
    request: Request
):

    inference_service = getattr(
        request.app.state,
        "inference_service",
        None
    )

    if (
        inference_service is None
        or not inference_service.initialized
    ):

        raise HTTPException(
            status_code=503,
            detail="Transformer model is not ready."
        )

    try:

        result = inference_service.generate(
            text=payload.text,
            max_new_tokens=payload.max_new_tokens
        )

        return GenerationResponse(**result)

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Generation error:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Internal generation error."
        )
