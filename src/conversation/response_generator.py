from typing import Dict

from src.conversation.query_processor import Query
from src.language_models.summarizer import Summarizer
from src.language_models.generator import TextGenerator
from src.analysis.classifier import NewsClassifier
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.ner_extractor import NERExtractor
from src.analysis.topic_modeler import TopicModeler  # if you want to hook it in


class ResponseGenerator:
    """
    High-level conversational coordinator.
    In a real app this would be used by your web UI / chatbot.
    """
    def __init__(self):
        self.summarizer = Summarizer()
        self.generator = TextGenerator()
        self.classifier = NewsClassifier()
        self.classifier.load()
        self.sentiment = SentimentAnalyzer()
        self.ner = NERExtractor()

    def respond(self, query: Query, context: Dict | None = None) -> str:
        intent = query.intent
        text = query.raw_text

        if intent == "greeting":
            return "Hello! I'm NewsBot. I can classify news, summarize articles, and analyze sentiment and topics."
        if intent == "help":
            return ("You can say things like:\n"
                    "- 'Classify this article: ...'\n"
                    "- 'Summarize this:' followed by text\n"
                    "- 'What is the sentiment of this article?'")

        if intent == "classify_article":
            label, conf = self.classifier.predict_with_confidence([text])[0]
            return f"The article looks like **{label}** (confidence {conf:.2f})."

        if intent == "summarize_article":
            summary = self.summarizer.summarize(text)
            return f"Here is a concise summary:\n\n{summary}"

        if intent == "sentiment_query":
            v = self.sentiment.analyze_vader(text)
            return (f"Sentiment (VADER compound={v['compound']:.3f}) — "
                    f"pos={v['pos']:.2f}, neu={v['neu']:.2f}, neg={v['neg']:.2f}.")

        if intent == "ner_query":
            ents = self.ner.extract_filtered(text)
            if not ents:
                return "I didn’t detect any major entities."
            formatted = ", ".join([f"{e} ({lbl})" for e, lbl in ents])
            return f"I found these entities: {formatted}"

        if intent == "goodbye":
            return "Goodbye! Thanks for using NewsBot."

        # Fallback
        gen = self.generator.generate(f"User asked: {text}\nAssistant:")
        return gen
