from typing import List, Dict
import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.data_processing.text_preprocessor import preprocess_corpus
from config.settings import config, paths


class TopicModeler:
    def __init__(self, num_topics: int | None = None):
        self.num_topics = num_topics or config.num_topics
        self.lda: LatentDirichletAllocation | None = None
        self.nmf: NMF | None = None
        self.count_vectorizer: CountVectorizer | None = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: List[str]) -> None:
        clean = preprocess_corpus(texts)

        self.count_vectorizer = CountVectorizer(
            max_df=0.95, min_df=3, stop_words="english"
        )
        X_count = self.count_vectorizer.fit_transform(clean)
        self.lda = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=config.random_state
        )
        self.lda.fit(X_count)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.95, min_df=3, stop_words="english"
        )
        X_tfidf = self.tfidf_vectorizer.fit_transform(clean)
        self.nmf = NMF(
            n_components=self.num_topics,
            random_state=config.random_state
        )
        self.nmf.fit(X_tfidf)

    def _top_words(self, model, feature_names, n_top_words=10) -> Dict[int, List[str]]:
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[:-n_top_words - 1:-1]
            topics[topic_idx] = [feature_names[i] for i in top_indices]
        return topics

    def get_lda_topics(self, n_top_words=10) -> Dict[int, List[str]]:
        if self.lda is None or self.count_vectorizer is None:
            raise RuntimeError("LDA model not trained.")
        return self._top_words(self.lda, self.count_vectorizer.get_feature_names_out(), n_top_words)

    def get_nmf_topics(self, n_top_words=10) -> Dict[int, List[str]]:
        if self.nmf is None or self.tfidf_vectorizer is None:
            raise RuntimeError("NMF model not trained.")
        return self._top_words(self.nmf, self.tfidf_vectorizer.get_feature_names_out(), n_top_words)

    def save(self) -> None:
        joblib.dump(self.lda, paths.lda_model_path)
        joblib.dump(self.count_vectorizer, paths.models_dir / "lda_vectorizer.joblib")
        joblib.dump(self.nmf, paths.nmf_model_path)
        joblib.dump(self.tfidf_vectorizer, paths.models_dir / "nmf_vectorizer.joblib")
