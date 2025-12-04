from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42  # reproducibility


class LanguageDetector:
    def detect_language(self, text: str) -> str:
        try:
            return detect(text or "")
        except Exception:
            return "unknown"
