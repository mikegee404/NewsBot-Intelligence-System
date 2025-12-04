from typing import Dict

from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import NewsTranslator
from src.analysis.classifier import NewsClassifier
from src.analysis.sentiment_analyzer import SentimentAnalyzer


class CrossLingualAnalyzer:
    """
    - Detect language
    - Translate to English
    - Run classifier + sentiment on translated text
    """

    def __init__(self):
        self.detector = LanguageDetector()
        self.translator = NewsTranslator()
        self.classifier = NewsClassifier()
        self.classifier.load()  # assumes trained model exists
        self.sentiment = SentimentAnalyzer()

    def analyze(self, text: str) -> Dict:
        lang = self.detector.detect_language(text)
        translated = text if lang == "en" else self.translator.translate(text, dest="en")
        label, conf = self.classifier.predict_with_confidence([translated])[0]
        sent_vader = self.sentiment.analyze_vader(translated)
        sent_blob = self.sentiment.analyze_textblob(translated)

        return {
            "original_language": lang,
            "translated_text": translated,
            "predicted_label": label,
            "confidence": conf,
            "sentiment_vader": sent_vader,
            "sentiment_blob": sent_blob,
        }
