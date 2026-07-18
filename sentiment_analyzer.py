"""
Semantic threat / harm analysis of transcribed sentences.

IMPORTANT: this deliberately does NOT use a keyword/blacklist approach.
It uses a zero-shot NLI transformer that scores the sentence against
meaning-based labels (e.g. "threat of violence", "verbal abuse"), so it
reacts to what the sentence MEANS, not to specific trigger words. This
means paraphrased threats, sarcasm-free abuse, and distress pleas are all
caught even if no "bad word" is present.
"""

from dataclasses import dataclass
from transformers import pipeline
import config


@dataclass
class HarmResult:
    label: str          # best-matching label, e.g. "threat of violence"
    score: float         # confidence 0-1
    is_harmful: bool     # True if label != normal AND score above threshold


class SentimentAnalyzer:
    def __init__(self):
        print(f"[SentimentAnalyzer] Loading zero-shot model '{config.ZERO_SHOT_MODEL}'...")
        self.classifier = pipeline("zero-shot-classification", model=config.ZERO_SHOT_MODEL)

    def analyze(self, sentence: str) -> HarmResult:
        if not sentence or not sentence.strip():
            return HarmResult(label="normal calm conversation", score=1.0, is_harmful=False)

        result = self.classifier(sentence, candidate_labels=config.THREAT_LABELS)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        is_harmful = (
            top_label != "normal calm conversation"
            and top_score >= config.TEXT_HARM_SCORE_THRESHOLD
        )
        return HarmResult(label=top_label, score=top_score, is_harmful=is_harmful)
