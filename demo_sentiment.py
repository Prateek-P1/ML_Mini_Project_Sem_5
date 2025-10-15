"""Simple demo script to load a pretrained vectorizer and model and predict sentiment for input reviews.

This script expects the following files to be present in the `trained_models_archive` folder
in the project root:

- TF_IDF_3_grams_vectorizer.joblib
- TF_IDF_3_grams_Logistic_Regression.joblib

It will load the vectorizer and model, preprocess input text (basic cleaning), vectorize it,
and print the predicted sentiment and confidence.

Usage:
    python demo_sentiment.py "This movie was great!"
    python demo_sentiment.py  # interactive prompt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

# Choose model files from trained_models_archive
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "trained_models_archive"
# Use the Binary 3-grams Logistic Regression model and its vectorizer by default.
# These are CountVectorizer-based and should transform without requiring extra fitted state.
VECTORIZER_NAME = "Binary_3_grams_vectorizer.joblib"
MODEL_NAME = "Binary_3_grams_Logistic_Regression.joblib"


def simple_preprocess(text: str) -> str:
    """Basic text cleaning to roughly match notebook preprocessing.

    - Remove HTML tags
    - Keep alphabetic characters and spaces
    - Lowercase
    - Collapse multiple spaces
    """
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pipeline(model_dir: Path) -> Tuple[object, object]:
    """Load vectorizer and model from disk. Raises FileNotFoundError if missing."""
    vec_path = model_dir / VECTORIZER_NAME
    model_path = model_dir / MODEL_NAME
    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at: {vec_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model


def predict_review(review: str, vectorizer, model) -> Tuple[str, float]:
    clean = simple_preprocess(review)
    vec = vectorizer.transform([clean])
    # LogisticRegression has predict_proba; other models may not. We handle both.
    try:
        proba = model.predict_proba(vec)[0]
        # Assuming binary [neg, pos]
        if proba.shape[0] == 2:
            pos_conf = float(proba[1])
        else:
            # multi-class fallback: take max class prob
            pos_conf = float(np.max(proba))
    except Exception:
        # Fall back to decision_function or predict
        try:
            score = model.decision_function(vec)
            # map score to a pseudo-probability using logistic sigmoid
            pos_conf = float(1 / (1 + np.exp(-score)))[0]
        except Exception:
            pred = model.predict(vec)[0]
            pos_conf = 1.0 if pred == 1 else 0.0

    pred_label = "Positive" if pos_conf >= 0.5 else "Negative"
    return pred_label, pos_conf


def interactive_mode(vectorizer, model):
    print("Enter a movie review (empty line to quit):")
    while True:
        try:
            review = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')
            break
        if not review:
            print('Goodbye.')
            break
        label, conf = predict_review(review, vectorizer, model)
        print(f"Predicted: {label} (confidence: {conf:.3f})")


def main(argv: list[str]):
    try:
        vectorizer, model = load_pipeline(MODEL_DIR)
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure the trained model files are present in the 'trained_models_archive' folder next to this script.")
        return 1

    if len(argv) > 1:
        review = ' '.join(argv[1:])
        label, conf = predict_review(review, vectorizer, model)
        print(f"Review: {review}")
        print(f"Predicted: {label} (confidence: {conf:.3f})")
    else:
        interactive_mode(vectorizer, model)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
