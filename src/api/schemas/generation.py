
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Input text for Transformer generation"
    )

    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=50,
        description="Maximum number of tokens to generate"
    )


class GenerationResponse(BaseModel):

    input_text: str

    generated_text: str

    generated_token_count: int
