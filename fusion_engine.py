"""
Fusion engine: combines (1) semantic text harm score and (2) pitch/volume
"deep voice" risk score for the SAME time window into one final risk score.

This is the core of the two-factor design the project calls for:
sentence meaning + pitch depth of the voice, evaluated together rather
than either signal alone.
"""

from dataclasses import dataclass
from typing import List, Optional

import config
from transcriber import Segment
from sentiment_analyzer import SentimentAnalyzer, HarmResult
from pitch_analyzer import PitchAnalyzer, VoiceWindowStats


@dataclass
class FlaggedEvent:
    sentence: str
    sentence_en: str
    start: float
    end: float
    harmful_words: List[str]
    harm_result: HarmResult
    voice_stats: VoiceWindowStats
    final_risk: float


class FusionEngine:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer

    def evaluate_segment(self, segment: Segment, pitch_analyzer: PitchAnalyzer) -> Optional[FlaggedEvent]:
        # Classification always runs on the English translation, since the harm
        # classifier is an English-only model. segment.text (native language,
        # e.g. Tamil) is kept separately for display and the alert message.
        harm_result = self.sentiment_analyzer.analyze(segment.text_en)
        voice_stats = pitch_analyzer.get_window_stats(segment.start, segment.end)

        final_risk = (
            config.TEXT_WEIGHT * harm_result.score * (1 if harm_result.is_harmful else 0.3)
            + config.PITCH_WEIGHT * voice_stats.pitch_risk_score
        )
        final_risk = min(final_risk, 1.0)

        # Identify the exact words spoken during the "deep voice" moment within this
        # sentence -- gives us WHICH words were said in that aggressive tone, not just
        # that the sentence as a whole was flagged.
        harmful_words = []
        if harm_result.is_harmful or voice_stats.is_deep_voice or voice_stats.is_volume_spike:
            harmful_words = [w.text for w in segment.words]

        if final_risk >= config.ALERT_RISK_THRESHOLD:
            return FlaggedEvent(
                sentence=segment.text,
                sentence_en=segment.text_en,
                start=segment.start,
                end=segment.end,
                harmful_words=harmful_words,
                harm_result=harm_result,
                voice_stats=voice_stats,
                final_risk=final_risk,
            )
        return None
