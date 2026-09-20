"""
Load model.pkl and run predictions on sample messages.

This script only loads and predicts — it never re-saves model.pkl.
Run train.py first to generate it.
"""

import pickle
from scipy.sparse import hstack, csr_matrix

from features import lemmatize_text, extract_features, FEATURE_COLUMNS

MODEL_PATH = "model.pkl"


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["tfidf"]


def predict_message(text, tfidf, model):
    feats = extract_features(text)
    lemmatized = lemmatize_text(text)

    tfidf_vec = tfidf.transform([lemmatized])
    combined = hstack([tfidf_vec, csr_matrix(feats[FEATURE_COLUMNS].values.reshape(1, -1))])

    pred = model.predict(combined)[0]
    proba = model.predict_proba(combined)[0]
    return pred, dict(zip(model.classes_, proba))


if __name__ == "__main__":
    model, tfidf = load_model()
    print(f"Loaded model — expects {model.n_features_in_} features, "
          f"TF-IDF vocab size {len(tfidf.vocabulary_)}.\n")

    test_messages = [
        "URGENT! Your account will be suspended. Verify now at http://bit.ly/verify123 or call 9876543210!!!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "Your OTP for transaction of Rs. 5000 is 483920. Valid for 10 mins.",
        "WIN A FREE IPHONE NOW!!! Click here to claim your prize: www.freegift.xyz",
    ]

    for msg in test_messages:
        pred, proba = predict_message(msg, tfidf, model)
        print(f"Text: {msg}")
        print(f"Predicted: {pred}")
        print(f"Probabilities: { {k: round(v, 4) for k, v in proba.items()} }")
        print("-" * 50)
