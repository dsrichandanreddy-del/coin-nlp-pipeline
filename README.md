# COiN NLP Pipeline — JPMorgan Chase Contract Intelligence Expansion

**Employer:** JPMorgan Chase | Compliance & Legal Technology  
**Role:** Applied AI/ML Engineer  
**Timeline:** September 2024 – April 2025  
**Domain:** Legal NLP · Named Entity Recognition · Document Classification

---

## Overview

Extension of JPMorgan's proprietary Contract Intelligence (COiN) platform from commercial credit agreements to two new document classes: custody agreements and regulatory filings. Built the NLP foundations layer — spaCy NER pipeline, SVM document classifier, and FastAPI inference microservice — enabling automated extraction of key legal entities at production-grade accuracy.

COiN processes 12,000+ commercial credit agreements annually, saving 360,000+ attorney work-hours per year. This expansion added coverage of ~5,900 additional documents per year across the two new document classes.

---

## Problem Statement

JPMorgan's legal teams process thousands of complex contracts annually. Prior to this expansion:

- COiN's NLP pipeline had been trained exclusively on commercial credit agreements
- Applying it to custody agreements and regulatory filings fell below the 85% precision threshold required for automated processing
- Manual review averaged **4.2 attorney-hours per document** across ~5,900 annual documents = **25,000+ attorney-hours/year**
- No automated document type classifier existed; manual routing introduced errors on mixed-structure documents

---

## My Contributions

| Area | Role | Description |
|------|------|-------------|
| spaCy NER Pipeline | **Primary Owner** | Designed fine-tuning strategy (class-specific vs. unified), trained and evaluated 3 separate NER models covering 8 entity types |
| NLTK Preprocessing | Contributor | Co-developed domain-adapted text normalization, abbreviation dictionaries, sentence boundary detection |
| SVM Document Classifier | **Primary Owner** | Built 5-class TF-IDF + LinearSVC classifier with 32 MLflow-tracked runs |
| FastAPI Microservice | **Primary Owner** | Implemented `/classify` and `/extract` endpoints, Pydantic schemas, latency optimization |
| PostgreSQL Entity Store | Contributor | Coordinated schema design with backend engineer for OCC-compliant audit trail |
| MLflow / MRM Docs | **Primary Owner** | Authored all model cards and validation documentation for JPMorgan Model Risk Management review |

---

## Technical Architecture

```
Documents (custody agreements, regulatory filings, credit agreements)
        │
        ▼
┌─────────────────────────┐
│   NLTK Preprocessing    │  tokenization · sentence boundary · stopword filtering
│   + spaCy NER           │  domain abbreviation normalization
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐    ┌─────────────────┐
│  SVM +  │    │   spaCy NER     │  3 class-specific models
│ TF-IDF  │    │   (per class)   │  credit · custody · regulatory
│Classifier│   └────────┬────────┘
└────┬────┘             │
     │         extracted entities
     │         (type, confidence, span)
     └──────────────────┤
                        ▼
              ┌──────────────────┐
              │   FastAPI        │  /classify  /extract
              │   Microservice   │  Pydantic · Uvicorn · Docker
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │   PostgreSQL     │  entity store · audit trail
              │   (SQLAlchemy)   │  OCC model risk compliance
              └──────────────────┘
```

---

## Key Technical Decisions

### Class-Specific vs. Unified NER Model
Ran empirical comparison: unified model achieved macro F1 of 0.84 (below 0.87 threshold) while class-specific models collectively achieved 0.89. Root cause: credit agreement training signals dominated gradient updates in the unified model, suppressing custody-specific entity patterns. **Decision: three class-specific models.**

### Hybrid Rule-Based + Statistical NER for Regulatory Temporal Expressions
Standard spaCy DATE recognition failed on multi-clause temporal expressions like *"no later than the fifth business day following the calendar quarter end in which the triggering event is determined to have occurred."* Built a library of 64 regex templates registered as `spaCy EntityRuler` components running before the statistical NER model. **Pattern: rule-based for high-structure entities, statistical for contextually dependent entities.**

### Confidence-Threshold Routing for Low-Confidence Extractions
Regulatory filings exhibit highest structural variability. Extractions with top-entity confidence below 0.72 are flagged for human review rather than auto-processed, creating a safe fallback. This captured 94% of low-confidence cases correctly.

### Versioned API Design for Backward Compatibility
COiN's intake pipeline expected a flat entity schema tied to credit agreement terminology. Rather than breaking the existing integration, implemented v1 (flat schema, credit only) and v2 (class-aware structured payload, all three document classes) endpoints. Credit agreement workflow migrated to v2 at its own pace.

---

## Results

### NER Extraction Accuracy
| Document Class | NER F1 |
|----------------|--------|
| Commercial Credit | 0.91 |
| Custody Agreements | 0.88 |
| Regulatory Filings | 0.87 |
| **Macro Average** | **0.89** |
| Target | ≥ 0.87 ✅ |

Per-entity F1 range: 0.84 (regulatory effective dates) → 0.96 (custodian identifiers)  
Inter-annotator agreement: Cohen's Kappa 0.81 (min: 0.78) ✅

### Document Classification
| Metric | Result |
|--------|--------|
| SVM Macro F1 | **0.92** (target: ≥0.91 ✅) |
| vs. keyword-routing baseline | +28% relative improvement |
| Misclassification rate (post shadow-mode) | **1.2%** (down from 6%) |
| MLflow runs logged | 32 |

### Service Performance
| Metric | Result |
|--------|--------|
| p95 latency @ 800 docs/hour | **143ms** (target: <180ms ✅) |
| Initial p95 (before optimization) | 214ms |
| PyTest coverage | **95%+** |
| Production incidents post-deploy | 0 |

### Business Impact
- **25,000+ attorney-hours/year eliminated** across custody (3,800/year) and regulatory filing (2,100/year) processing
- **80% reduction in compliance-related errors** (COiN program-level)
- **30% reduction in legal operations cost** (COiN program-level)
- MRM formal review completed in **4 weeks** (2 weeks ahead of 6-week minimum estimate)

---

## Stack

| Layer | Technology |
|-------|-----------|
| NLP / NER | spaCy v3 (`en_core_web_sm` fine-tuned), NLTK |
| Classification | scikit-learn `LinearSVC`, TF-IDF, chi-squared feature selection |
| Serving | FastAPI, Uvicorn, Pydantic, Docker |
| Data | PostgreSQL, SQLAlchemy ORM (async), Pandas |
| Experiment Tracking | MLflow (32 runs, artifact registry, model cards) |
| Testing / CI | PyTest (95%+), GitHub Actions, Locust |

---

## Project Structure

```
1_COiN_NLP_Pipeline/
├── README.md
├── src/
│   ├── preprocessing/
│   │   ├── nltk_preprocessor.py       # NLTK tokenization, stopword filtering, abbreviation normalization
│   │   └── domain_abbreviations.py    # JPMorgan legal domain abbreviation dictionary
│   ├── ner/
│   │   ├── ner_trainer.py             # spaCy NER fine-tuning loop (class-specific)
│   │   ├── entity_ruler_patterns.py   # 64 regex templates for regulatory temporal expressions
│   │   └── ner_evaluator.py           # Per-class F1 evaluation, Cohen's Kappa
│   ├── classification/
│   │   ├── document_classifier.py     # SVM + TF-IDF 5-class classifier
│   │   └── mlflow_trainer.py          # Hyperparameter grid search with MLflow tracking
│   ├── api/
│   │   ├── main.py                    # FastAPI app, /classify and /extract endpoints
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   └── inference.py               # NLP pipeline inference wrapper
│   └── db/
│       ├── models.py                  # SQLAlchemy ORM entity store schema
│       └── session.py                 # Async session management, connection pool
├── tests/
│   ├── test_preprocessor.py
│   ├── test_classifier.py
│   ├── test_ner_pipeline.py
│   └── test_api_endpoints.py
├── notebooks/
│   ├── 01_corpus_analysis.ipynb       # Document class EDA, entity density analysis
│   ├── 02_annotation_protocol.ipynb   # Inter-annotator agreement (Cohen's Kappa)
│   ├── 03_ner_training.ipynb          # NER fine-tuning experiments
│   ├── 04_classifier_ablation.ipynb   # SVM hyperparameter grid search
│   └── 05_shadow_mode_analysis.ipynb  # Shadow-mode validation, hybrid document discovery
├── configs/
│   ├── spacy_config_credit.cfg
│   ├── spacy_config_custody.cfg
│   └── spacy_config_regulatory.cfg
├── docs/
│   ├── architecture.md
│   ├── entity_taxonomy.md             # 8 entity types across 3 document classes
│   └── model_card_template.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml                     # PyTest 95% gate, Docker build
```

---

## Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy base model
python -m spacy download en_core_web_sm

# Run the FastAPI service
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ --cov=src --cov-report=term-missing

# Load test
locust -f tests/locustfile.py --headless -u 100 -r 10 --run-time 60s
```

### API Usage

```python
import requests

# Classify a document
response = requests.post("http://localhost:8000/v2/classify", json={
    "text": "This Custody Agreement is entered into as of...",
})
# {"document_class": "custody", "confidence": 0.94, "alternatives": [...]}

# Extract entities
response = requests.post("http://localhost:8000/v2/extract", json={
    "text": "...",
    "document_class": "custody"
})
# {"entities": [{"type": "CUSTODIAN_ID", "text": "...", "confidence": 0.96, "start": 42, "end": 67}]}
```

---

## Key Learnings

1. **Class-specific models beat unified models on structurally diverse document classes** — when training signal distribution is uneven (more credit docs than custody docs), a unified model under-represents minority classes. Empirical comparison before architectural commitment is worth the time.

2. **Hybrid rule-based + statistical NER is the right pattern for high-structure entities** — regex patterns for date/deadline constructs that follow predictable formats, statistical NER for contextually dependent entities. Don't force a statistical model to learn patterns that are better expressed as rules.

3. **Start MRM documentation in parallel with training, not after** — submitting preliminary docs for informal review 6 weeks before the formal deadline eliminated restart cycles and completed formal review 2 weeks ahead of schedule.

4. **Async SQLAlchemy + connection pool pre-warming is critical for latency SLAs under concurrent load** — per-request session instantiation was the bottleneck that caused the initial p95 SLA breach. Refactoring to async context-managed sessions resolved it.
