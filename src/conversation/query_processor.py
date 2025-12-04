from dataclasses import dataclass

from src.conversation.intent_classifier import IntentClassifier, Intent


@dataclass
class Query:
    raw_text: str
    intent: Intent


class QueryProcessor:
    def __init__(self):
        self.intent_classifier = IntentClassifier()

    def process(self, text: str) -> Query:
        intent = self.intent_classifier.classify(text)
        return Query(raw_text=text, intent=intent)
