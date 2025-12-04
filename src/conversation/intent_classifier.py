from typing import Literal


Intent = Literal[
    "greeting",
    "help",
    "classify_article",
    "summarize_article",
    "sentiment_query",
    "topic_query",
    "ner_query",
    "goodbye",
    "unknown",
]


class IntentClassifier:
    """
    Simple rule-based intents for the course demo.
    You can later replace with a trained classifier if you like.
    """
    def classify(self, text: str) -> Intent:
        t = text.lower()

        if any(w in t for w in ["hi", "hello", "hey"]):
            return "greeting"
        if "help" in t or "what can you do" in t:
            return "help"
        if "classify" in t or "category" in t:
            return "classify_article"
        if "summarize" in t or "summary" in t:
            return "summarize_article"
        if "sentiment" in t or "positive" in t or "negative" in t:
            return "sentiment_query"
        if "topic" in t or "themes" in t:
            return "topic_query"
        if "entities" in t or "names" in t or "organizations" in t:
            return "ner_query"
        if any(w in t for w in ["bye", "goodbye", "see you"]):
            return "goodbye"
        return "unknown"

