"""
Pydantic Schemas — COiN NLP Inference Service
Request/response models enforcing strict validation before NLP pipeline execution.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class DocumentClass(str, Enum):
    CREDIT = "credit"
    CUSTODY = "custody"
    REGULATORY = "regulatory"
    ISDA = "isda"
    OTHER = "other"


# ─── Classification ───────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw document text")
    document_id: Optional[str] = Field(None, description="Upstream document identifier for audit trail")

    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Document text must not be empty or whitespace only")
        return v.strip()


class AlternativeClass(BaseModel):
    document_class: str
    confidence: float


class ClassifyResponse(BaseModel):
    document_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternatives: List[AlternativeClass] = Field(default_factory=list)
    hybrid_document_flag: bool = Field(False, description="True if document contains mixed document class signals")
    model_version: str


# ─── Entity Extraction ────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=10)
    document_class: str = Field(..., description="Document class label (from /classify or known upstream)")
    document_id: Optional[str] = None
    confidence_threshold: float = Field(
        0.72,
        ge=0.0, le=1.0,
        description="Entities below this threshold flagged for human review rather than auto-processing"
    )


class ExtractedEntity(BaseModel):
    type: str = Field(..., description="Entity type label (e.g. CUSTODIAN_ID, DEADLINE)")
    text: str = Field(..., description="Extracted entity text span")
    confidence: float = Field(..., ge=0.0, le=1.0)
    start: int = Field(..., description="Character-level start position in source text")
    end: int = Field(..., description="Character-level end position in source text")
    requires_review: bool = Field(False, description="True if confidence below threshold, routed to human review")


class ExtractResponse(BaseModel):
    document_class: str
    entities: List[ExtractedEntity]
    model_version: str
    low_confidence_count: int = 0

    def __init__(self, **data):
        super().__init__(**data)
        self.low_confidence_count = sum(1 for e in self.entities if e.requires_review)


# ─── V1 Backward Compatible Schema ───────────────────────────────────────────

class ExtractResponseV1(BaseModel):
    """Flat schema maintaining backward compatibility with existing COiN credit agreement pipeline."""
    entities: List[Dict[str, Any]]
