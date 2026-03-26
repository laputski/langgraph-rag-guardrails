from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    user_id: Optional[str] = Field(None, description="Optional user identifier for rate limiting")
    prompt_version: Optional[str] = Field(None, description="Prompt template version (v1 or v2)")


class IngestRequest(BaseModel):
    admin_token: str = Field(..., description="Admin token from ADMIN_TOKEN env var")
    force_rebuild: bool = Field(
        False,
        description="If true, drops and recreates the Qdrant collection before ingesting",
    )


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    documents: int


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
