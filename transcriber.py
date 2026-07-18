"""
Speech-to-text with word-level timestamps.
Uses faster-whisper so we know the EXACT start/end time of every word spoken,
which lets us later line up each word with its pitch/volume at that instant.
"""

from dataclasses import dataclass
from typing import List
from faster_whisper import WhisperModel
import config


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str          # full sentence
    start: float
    end: float
    words: List[Word]


class Transcriber:
    def __init__(self):
        print(f"[Transcriber] Loading Whisper model '{config.WHISPER_MODEL_SIZE}' "
              f"({config.WHISPER_DEVICE}, {config.WHISPER_COMPUTE_TYPE})...")
        self.model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

    def transcribe(self, audio_path: str = None, audio_array=None, sample_rate: int = None) -> List[Segment]:
        """
        Transcribe either a file path OR an in-memory numpy float32 audio array.
        Returns a list of Segment objects, each with sentence-level text and
        word-level timestamps.
        """
        source = audio_path if audio_path is not None else audio_array
        segments_gen, _info = self.model.transcribe(
            source,
            word_timestamps=True,
            vad_filter=True,  # skip silence, avoids hallucinated text on empty audio
        )

        segments: List[Segment] = []
        for seg in segments_gen:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(Word(text=w.word.strip(), start=w.start, end=w.end))
            segments.append(Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
            ))
        return segments
