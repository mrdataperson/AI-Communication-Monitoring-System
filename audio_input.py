"""
Two audio input modes:
  1. load_file()   -> read an existing audio file from disk
  2. LiveRecorder   -> stream from the microphone in rolling chunks for
                        near-real-time monitoring
"""

import numpy as np
import soundfile as sf
import librosa
import sounddevice as sd
import queue
import config


def load_file(path: str, target_sr: int = 16000):
    """Load an audio file and resample to target_sr. Returns (audio_float32, sr)."""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # downmix to mono
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


class LiveRecorder:
    """
    Streams microphone audio in rolling windows (config.LIVE_CHUNK_SECONDS long)
    so the rest of the pipeline can treat live monitoring the same way it
    treats a short audio file -- just repeated every few seconds.
    """

    def __init__(self, sample_rate: int = None, chunk_seconds: float = None):
        self.sample_rate = sample_rate or config.LIVE_SAMPLE_RATE
        self.chunk_seconds = chunk_seconds or config.LIVE_CHUNK_SECONDS
        self._q = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[LiveRecorder] status: {status}")
        self._q.put(indata.copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        print("[LiveRecorder] Microphone stream started. Listening...")

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            print("[LiveRecorder] Microphone stream stopped.")

    def read_chunk(self) -> np.ndarray:
        """
        Blocks until enough audio has accumulated for one chunk_seconds
        window, then returns it as a flat float32 numpy array.
        """
        needed_samples = int(self.sample_rate * self.chunk_seconds)
        collected = []
        total = 0
        while total < needed_samples:
            block = self._q.get()
            collected.append(block)
            total += len(block)
        audio = np.concatenate(collected, axis=0).flatten()
        return audio[:needed_samples]
