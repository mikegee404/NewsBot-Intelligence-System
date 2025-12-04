from typing import Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze_vader(self, text: str) -> Dict[str, float]:
        return self.vader.polarity_scores(text or "")

    def analyze_textblob(self, text: str) -> Dict[str, float]:
        blob = TextBlob(text or "")
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
