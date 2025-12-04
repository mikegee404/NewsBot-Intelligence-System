from typing import Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_processing.text_preprocessor import preprocess_corpus
from src.data_processing.feature_extractor import TfidfFeatureExtractor
from config.settings import config, paths


CLASSIFIERS = {
    "logreg": LogisticRegression(max_iter=2000, C=2.0),
    "linearsvc": LinearSVC(),
    "nb": MultinomialNB()
}


class NewsClassifier:
    """
    Multi-class news classifier using TF-IDF + classic ML models.
    """

    def __init__(self, clf_name: str = "linearsvc"):
        if clf_name not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {clf_name}")
        self.clf_name = clf_name
        self.pipeline: Pipeline | None = None

    def train(self, df: pd.DataFrame, text_col: str = "text",
              label_col: str = "label") -> dict:
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(str).tolist()

        texts_clean = preprocess_corpus(texts)

        X_train, X_test, y_train, y_test = train_test_split(
            texts_clean,
            labels,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=labels
        )

        tfidf = TfidfFeatureExtractor(max_features=config.max_features_tfidf)
        clf = CLASSIFIERS[self.clf_name]

        self.pipeline = Pipeline([
            ("tfidf", tfidf.vectorizer),
            ("clf", clf)
        ])

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred)

        return {"accuracy": acc, "macro_f1": f1, "report": report}

    def predict(self, texts: List[str]) -> List[str]:
        if self.pipeline is None:
            raise RuntimeError("Model not trained or loaded.")
        texts_clean = preprocess_corpus(texts)
        return self.pipeline.predict(texts_clean).tolist()

    def predict_with_confidence(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        For models that support decision_function or predict_proba.
        Returns (label, confidence) pairs.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained or loaded.")

        texts_clean = preprocess_corpus(texts)
        clf = self.pipeline.named_steps["clf"]
        X = self.pipeline.named_steps["tfidf"].transform(texts_clean)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)
            labels = clf.classes_
            max_idx = probs.argmax(axis=1)
            return [(labels[i], float(probs[j, i]))
                    for j, i in enumerate(max_idx)]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            labels = clf.classes_
            if scores.ndim == 1:
                scores = scores[:, None]
            max_idx = scores.argmax(axis=1)
            conf = scores.max(axis=1)
            return [(labels[i], float(conf[j]))
                    for j, i in enumerate(max_idx)]
        else:
            preds = clf.predict(X)
            return [(p, 1.0) for p in preds]

    def save(self, path: str | None = None) -> None:
        if path is None:
            path = str(paths.classifier_path)
        joblib.dump(self.pipeline, path)

    def load(self, path: str | None = None) -> None:
        if path is None:
            path = str(paths.classifier_path)
        self.pipeline = joblib.load(path)
