"""
Train the message classifier and save model.pkl.

Run this whenever you change the data, the features, or want to
retrain. It's the ONLY script that should ever write model.pkl.
"""

import pickle
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from features import lemmatize_text, extract_features, FEATURE_COLUMNS

DATA_PATH = "spam_or_nonspam.csv"
MODEL_PATH = "model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    print(df.head(1))

    y = df["LABEL"].astype(str).str.lower().str.strip()

    # Engineered features come from the RAW text, before lemmatization.
    extra = df["TEXT"].apply(extract_features)

    # Lemmatize a separate column for TF-IDF; keep raw TEXT untouched.
    lemmatized_text = df["TEXT"].apply(lemmatize_text)

    (x_train, x_test,
     y_train, y_test,
     extra_train, extra_test) = train_test_split(
        lemmatized_text, y, extra,
        test_size=0.2, random_state=42, stratify=y,
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(x_train)
    X_test_tfidf = tfidf.transform(x_test)

    X_train = hstack([X_train_tfidf, csr_matrix(extra_train[FEATURE_COLUMNS].values)])
    X_test = hstack([X_test_tfidf, csr_matrix(extra_test[FEATURE_COLUMNS].values)])

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("-" * 10)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "tfidf": tfidf}, f)
    print(f"\nSaved {MODEL_PATH} — model expects {model.n_features_in_} features "
          f"({len(tfidf.vocabulary_)} TF-IDF + {len(FEATURE_COLUMNS)} engineered).")


if __name__ == "__main__":
    main()
