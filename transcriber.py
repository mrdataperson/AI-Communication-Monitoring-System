"""
Speech-to-text with word-level timestamps, with Tamil (and other language)
support.

Two passes are run over the same audio:
  1. "native" pass  -> transcribes in whatever language was actually spoken
                        (e.g. Tamil script), used for word timestamps
                        (pitch alignment) and for the alert message so it
                        reads naturally to a Tamil speaker.
  2. "translated" pass -> Whisper's built-in translate task, always outputs
                        English, used ONLY to feed the English-only harm
                        classifier. This keeps detection accurate regardless
                        of what language was spoken.
"""

from dataclasses import dataclass
from typing import List, Optional
from faster_whisper import WhisperModel
import config


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str            # sentence in the ORIGINAL spoken language (e.g. Tamil script)
    text_en: str          # same sentence translated to English (for classification)
    start: float
    end: float
    words: List[Word]     # word timestamps in the original language
    language: str          # detected language code, e.g. "ta", "en"


class Transcriber:
    def __init__(self):
        print(f"[Transcriber] Loading Whisper model '{config.WHISPER_MODEL_SIZE}' "
              f"({config.WHISPER_DEVICE}, {config.WHISPER_COMPUTE_TYPE})...")
        self.model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

    def _run(self, source, task: str):
        return self.model.transcribe(
            source,
            task=task,
            language=config.WHISPER_LANGUAGE,  # None = auto-detect
            word_timestamps=(task == "transcribe"),
            vad_filter=True,  # skip silence, avoids hallucinated text on empty audio
        )

    @staticmethod
    def _best_overlap(native_seg, translated_segments) -> Optional[str]:
        """Find the translated segment whose time range overlaps most with
        this native segment, and return its English text."""
        best_text, best_overlap = "", 0.0
        for t_seg in translated_segments:
            overlap = min(native_seg.end, t_seg.end) - max(native_seg.start, t_seg.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_text = t_seg.text.strip()
        return best_text

    def transcribe(self, audio_path: str = None, audio_array=None, sample_rate: int = None) -> List[Segment]:
        """
        Transcribe either a file path OR an in-memory numpy float32 audio array.
        Returns a list of Segment objects with native-language text + word
        timestamps, and an English translation attached for classification.
        """
        source = audio_path if audio_path is not None else audio_array

        native_gen, native_info = self._run(source, task="transcribe")
        native_segments = list(native_gen)

        detected_lang = native_info.language

        # If the detected language is already English, skip the second pass --
        # translation would just be a no-op and we'd be doubling compute time.
        if detected_lang == "en":
            translated_segments = native_segments
        else:
            translated_gen, _ = self._run(source, task="translate")
            translated_segments = list(translated_gen)

        segments: List[Segment] = []
        for seg in native_segments:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(Word(text=w.word.strip(), start=w.start, end=w.end))

            if detected_lang == "en":
                text_en = seg.text.strip()
            else:
                text_en = self._best_overlap(seg, translated_segments)

            segments.append(Segment(
                text=seg.text.strip(),
                text_en=text_en,
                start=seg.start,
                end=seg.end,
                words=words,
                language=detected_lang,
            ))
        return segments
