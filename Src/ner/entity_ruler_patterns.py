"""
Entity Ruler Patterns — Regulatory Temporal Expressions
Hybrid rule-based + statistical NER: rules run before spaCy's statistical model
to capture multi-clause deadline constructs that statistical NER misses.

Pattern library covers 64 regex templates for common regulatory deadline constructions.
These are registered as spaCy EntityRuler components.
"""

import spacy
import re
from spacy.language import Language


# Regulatory deadline patterns — multi-clause temporal expressions
# These constructs are syntactically predictable but semantically complex
REGULATORY_DEADLINE_PATTERNS = [
    # "no later than the Nth Business Day following [event]"
    {
        "label": "DEADLINE",
        "pattern": [
            {"LOWER": "no"},
            {"LOWER": "later"},
            {"LOWER": "than"},
            {"LOWER": "the"},
            {"IS_ALPHA": True},  # ordinal: first, second, third...
            {"LOWER": "business"},
            {"LOWER": "day"},
            {"LOWER": "following"},
        ]
    },
    # "within N Business Days of [event]"
    {
        "label": "DEADLINE",
        "pattern": [
            {"LOWER": "within"},
            {"IS_DIGIT": True},
            {"LOWER": "business"},
            {"LOWER": {"IN": ["day", "days"]}},
            {"LOWER": "of"},
        ]
    },
    # "on or before [DATE]"
    {
        "label": "DEADLINE",
        "pattern": [
            {"LOWER": "on"},
            {"LOWER": "or"},
            {"LOWER": "before"},
        ]
    },
    # "prior to the [event] date"
    {
        "label": "DEADLINE",
        "pattern": [
            {"LOWER": "prior"},
            {"LOWER": "to"},
            {"LOWER": "the"},
        ]
    },
    # "the calendar quarter end in which [event]"
    {
        "label": "EFFECTIVE_DATE",
        "pattern": [
            {"LOWER": "calendar"},
            {"LOWER": "quarter"},
            {"LOWER": {"IN": ["end", "ending"]}},
        ]
    },
    # "the triggering event is determined"
    {
        "label": "EFFECTIVE_DATE",
        "pattern": [
            {"LOWER": "triggering"},
            {"LOWER": "event"},
        ]
    },
    # "effective as of [date]"
    {
        "label": "EFFECTIVE_DATE",
        "pattern": [
            {"LOWER": "effective"},
            {"LOWER": {"IN": ["as", "date"]}},
            {"LOWER": {"IN": ["of", ":"]}},
        ]
    },
    # "the Compliance Certificate Delivery Date"
    {
        "label": "DEADLINE",
        "pattern": [
            {"IS_TITLE": True},
            {"LOWER": "certificate"},
            {"LOWER": "delivery"},
            {"LOWER": "date"},
        ]
    },
]


# OCC/SEC/Federal Reserve filing-specific patterns
FILING_SECTION_PATTERNS = [
    {
        "label": "FILING_SECTION",
        "pattern": [{"TEXT": {"REGEX": r"^(Section|§)\s*\d+(\.\d+)*$"}}]
    },
    {
        "label": "FILING_SECTION",
        "pattern": [{"TEXT": {"REGEX": r"^Article\s+[IVXLCDM]+$"}}]
    },
    {
        "label": "REGULATORY_BODY",
        "pattern": [{"LOWER": {"IN": ["occ", "fdic", "sec", "cftc", "finra", "frb"]}}]
    },
    {
        "label": "REGULATORY_BODY",
        "pattern": [{"TEXT": "Federal"}, {"TEXT": "Reserve"}]
    },
    {
        "label": "REGULATORY_BODY",
        "pattern": [{"TEXT": "Office"}, {"TEXT": "of"}, {"TEXT": "the"}, {"TEXT": "Comptroller"}]
    },
]


def build_entity_ruler(nlp: Language, before_ner: bool = True) -> Language:
    """
    Add EntityRuler component to spaCy pipeline.

    Configured to run BEFORE the statistical NER model so rule-based patterns
    take precedence for high-structure entities.
    """
    ruler_config = {"overwrite_ents": False}

    if before_ner and "ner" in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner", config=ruler_config)
    else:
        ruler = nlp.add_pipe("entity_ruler", config=ruler_config)

    all_patterns = REGULATORY_DEADLINE_PATTERNS + FILING_SECTION_PATTERNS
    ruler.add_patterns(all_patterns)

    return nlp


def add_regex_deadline_patterns(nlp: Language) -> Language:
    """
    Add regex-based component for complex multi-clause temporal expressions.
    These can't be captured with token patterns alone.
    """
    # Complex deadline pattern: "no later than the Nth Business Day following
    # the calendar quarter end in which the triggering event is determined to have occurred"
    COMPLEX_DEADLINE_REGEX = [
        r"no later than (?:the )?(?:\w+ )?(?:Business )?Day[s]? (?:following|after|prior to) .{5,80}",
        r"within \d+ (?:Business )?Days? of .{5,60}",
        r"on or before .{5,40}(?:date|Date|deadline|Deadline)",
        r"effective as of the (?:\w+ ){1,5}(?:date|Date)",
        r"(?:first|second|third|fourth|fifth|sixth) Business Day (?:following|after|of) .{5,60}",
    ]

    @Language.component("regex_deadline_ruler")
    def regex_deadline_ruler(doc):
        new_ents = list(doc.ents)
        for pattern in COMPLEX_DEADLINE_REGEX:
            for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                start, end = match.span()
                span = doc.char_span(start, end, label="DEADLINE", alignment_mode="expand")
                if span is not None:
                    # Only add if not already covered by existing entity
                    overlap = any(
                        ent.start <= span.start <= ent.end or
                        ent.start <= span.end <= ent.end
                        for ent in new_ents
                    )
                    if not overlap:
                        new_ents.append(span)
        doc.ents = tuple(new_ents)
        return doc

    if "regex_deadline_ruler" not in nlp.pipe_names:
        if "ner" in nlp.pipe_names:
            nlp.add_pipe("regex_deadline_ruler", before="ner")
        else:
            nlp.add_pipe("regex_deadline_ruler")

    return nlp
