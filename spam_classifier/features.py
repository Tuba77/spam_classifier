"""
Shared preprocessing and feature extraction.

Both train.py and predict.py import from here so the exact same
transformations are applied at training time and at inference time.
This is the single most common source of "N features expected but
got M" errors — never redefine these functions separately in
another file.
"""

import re
import spacy
import pandas as pd

_nlp = None


def get_nlp():
    """Load spaCy's model once and reuse it (loading it repeatedly is slow)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def lemmatize_text(text):
    """Lowercase, lemmatize, and strip stopwords/punctuation/whitespace."""
    doc = get_nlp()(str(text).lower())
    return " ".join([
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ])


def extract_features(text):
    """
    Structural features that lemmatization destroys (URLs, punctuation,
    casing) but that matter a lot for telling spam/smishing apart from
    ham. Always call this on the RAW text, before lemmatize_text().
    """
    text = str(text)
    return pd.Series({
        "has_url": int(bool(re.search(r"http|www\.", text, re.I))),
        "has_phone": int(bool(re.search(r"\b\d{10}\b|\+\d{10,}", text))),
        "exclaim_count": text.count("!"),
        "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        "upper_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
        "length": len(text),
    })


FEATURE_COLUMNS = [
    "has_url", "has_phone", "exclaim_count",
    "digit_ratio", "upper_ratio", "length",
]
