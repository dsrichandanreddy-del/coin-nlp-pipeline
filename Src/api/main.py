"""
FastAPI Inference Microservice — COiN NLP Pipeline
Exposes /classify and /extract endpoints for COiN document intake workflow.

Design decisions:
- All three spaCy NER models pre-loaded at startup (eliminates per-request model load overhead)
- Async SQLAlchemy sessions with connection pool pre-warming (resolves p95 SLA breach from per-request sessions)
- Versioned endpoints: v1 (flat schema, backward compat) and v2 (class-aware structured payload)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import spacy
import logging

from .schemas import (
    ClassifyRequest, ClassifyResponse,
    ExtractRequest, ExtractResponse, ExtractResponseV1,
)
from .inference import NLPPipeline
from ..db.session import get_async_session, pre_warm_pool

logger = logging.getLogger(__name__)

# Global model registry — loaded once at startup
_pipeline: NLPPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all NER models and pre-warm DB connection pool at startup."""
    global _pipeline
    logger.info("Loading NLP models...")
    _pipeline = NLPPipeline(model_dir="models/ner")
    _pipeline.load_all_models()  # loads credit, custody, regulatory NER models
    logger.info("NLP models loaded. Pre-warming DB connection pool...")
    await pre_warm_pool()
    logger.info("Service ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="COiN NLP Inference Service",
    version="2.0.0",
    description="Document classification and entity extraction for JPMorgan COiN platform",
    lifespan=lifespan,
)


# ─── V2 Endpoints (class-aware, all document types) ──────────────────────────

@app.post("/v2/classify", response_model=ClassifyResponse)
async def classify_document_v2(
    request: ClassifyRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Classify a document into one of 5 document categories.
    Returns predicted class, confidence score, and top-2 alternative classes.
    Replaces manual routing logic in COiN's document intake pipeline.
    """
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Document text too short for classification.")

    result = _pipeline.classify(request.text)
    await _pipeline.log_classification(session, request, result)

    return ClassifyResponse(
        document_class=result["class"],
        confidence=result["confidence"],
        alternatives=result["alternatives"],
        hybrid_document_flag=result.get("hybrid_flag", False),
        model_version=_pipeline.classifier_version,
    )


@app.post("/v2/extract", response_model=ExtractResponse)
async def extract_entities_v2(
    request: ExtractRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Extract named entities from document text.
    Returns structured JSON payload with entity type, text, confidence, and character spans.
    Accepts an explicit document_class label (use /classify first if unknown).
    """
    if request.document_class not in _pipeline.supported_classes:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported document class: {request.document_class}. "
                   f"Supported: {_pipeline.supported_classes}"
        )

    entities = _pipeline.extract(request.text, request.document_class)
    await _pipeline.persist_entities(session, request, entities)

    return ExtractResponse(
        document_class=request.document_class,
        entities=entities,
        model_version=_pipeline.ner_versions[request.document_class],
    )


# ─── V1 Endpoints (backward compatible, credit agreements only) ──────────────

@app.post("/v1/extract", response_model=ExtractResponseV1)
async def extract_entities_v1(
    request: ExtractRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    V1 endpoint maintaining backward compatibility with existing COiN credit agreement pipeline.
    Returns flat entity list in the legacy schema expected by upstream routing logic.
    """
    entities = _pipeline.extract(request.text, "credit")
    await _pipeline.persist_entities(session, request, entities)

    # Flatten to v1 schema
    flat_entities = [
        {"entity_type": e["type"], "text": e["text"], "confidence": e["confidence"]}
        for e in entities
    ]
    return ExtractResponseV1(entities=flat_entities)


# ─── Health & Readiness ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if _pipeline is None or not _pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready", "models_loaded": list(_pipeline.ner_versions.keys())}
