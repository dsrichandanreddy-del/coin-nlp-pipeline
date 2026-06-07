"""
NER Trainer — COiN NLP Pipeline
Trains class-specific spaCy NER models for each document class.
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import mlflow
import random
from pathlib import Path
from typing import List, Tuple, Dict


DOCUMENT_CLASSES = ["credit", "custody", "regulatory"]

ENTITY_TYPES = {
    "credit": ["PARTY_NAME", "MONETARY_VALUE", "COVENANT_TERM", "DEADLINE", "PENALTY_CLAUSE", "LEGAL_ROLE", "DATE", "JURISDICTION"],
    "custody": ["PARTY_NAME", "CUSTODIAN_ID", "ASSET_CLASS", "SETTLEMENT_INSTRUCTION", "COUNTERPARTY_CODE", "DATE", "MONETARY_VALUE", "LEGAL_ROLE"],
    "regulatory": ["PARTY_NAME", "REGULATORY_BODY", "FILING_SECTION", "EFFECTIVE_DATE", "DATE", "MONETARY_VALUE", "LEGAL_ROLE", "DEADLINE"],
}


def load_training_data(doc_class: str, data_path: str) -> List[Tuple]:
    """Load annotated training data for a given document class."""
    import json
    with open(f"{data_path}/{doc_class}_train.jsonl") as f:
        data = [json.loads(line) for line in f]
    return [(item["text"], {"entities": item["entities"]}) for item in data]


def train_ner_model(
    doc_class: str,
    train_data: List[Tuple],
    val_data: List[Tuple],
    n_iter: int = 30,
    dropout: float = 0.2,
    output_dir: str = "models",
    run_name: str = None,
) -> spacy.Language:
    """
    Fine-tune spaCy NER model for a specific document class.

    Uses early stopping on validation F1 to prevent overfitting.
    All metrics logged to MLflow.
    """
    nlp = spacy.load("en_core_web_sm")

    # Add NER component or get existing
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add entity labels for this document class
    for label in ENTITY_TYPES[doc_class]:
        ner.add_label(label)

    # Prepare training examples
    train_examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        train_examples.append(example)

    # Initialize optimizer
    optimizer = nlp.initialize(lambda: iter(train_examples))

    best_val_f1 = 0.0
    patience_counter = 0
    patience = 5

    with mlflow.start_run(run_name=run_name or f"ner_{doc_class}"):
        mlflow.log_params({
            "doc_class": doc_class,
            "n_iter": n_iter,
            "dropout": dropout,
            "entity_types": ENTITY_TYPES[doc_class],
            "base_model": "en_core_web_sm",
        })

        for iteration in range(n_iter):
            random.shuffle(train_examples)
            losses = {}

            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=dropout, losses=losses)

            # Validation F1
            val_f1 = evaluate_ner(nlp, val_data)
            mlflow.log_metrics({
                "train_loss": losses.get("ner", 0),
                "val_f1": val_f1,
            }, step=iteration)

            print(f"[{doc_class}] Iter {iteration+1}/{n_iter} | Loss: {losses.get('ner', 0):.4f} | Val F1: {val_f1:.4f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                output_path = Path(output_dir) / doc_class
                output_path.mkdir(parents=True, exist_ok=True)
                nlp.to_disk(output_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iteration+1} (best val F1: {best_val_f1:.4f})")
                    break

        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.log_artifact(str(Path(output_dir) / doc_class))

    return nlp


def evaluate_ner(nlp: spacy.Language, eval_data: List[Tuple]) -> float:
    """Compute macro F1 over all entity types on evaluation data."""
    examples = []
    for text, annotations in eval_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    scores = nlp.evaluate(examples)
    return scores["ents_f"]


def run_training_pipeline(data_path: str, output_dir: str):
    """Train NER models for all three document classes."""
    results = {}
    for doc_class in DOCUMENT_CLASSES:
        print(f"\n=== Training NER model: {doc_class} ===")
        train_data = load_training_data(doc_class, data_path)
        val_data = load_training_data(doc_class.replace("train", "val"), data_path)

        nlp = train_ner_model(
            doc_class=doc_class,
            train_data=train_data,
            val_data=val_data,
            output_dir=output_dir,
        )
        val_f1 = evaluate_ner(nlp, val_data)
        results[doc_class] = val_f1
        print(f"[{doc_class}] Final Val F1: {val_f1:.4f}")

    print("\n=== Training Complete ===")
    for cls, f1 in results.items():
        print(f"  {cls}: F1 = {f1:.4f}")
    return results


if __name__ == "__main__":
    run_training_pipeline(data_path="data/annotated", output_dir="models/ner")
