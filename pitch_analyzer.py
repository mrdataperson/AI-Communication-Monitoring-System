"""
Pitch (F0) and volume (RMS energy) analysis.

Gives us, for any time window in the audio:
  - average pitch in Hz (lower = "deeper" voice)
  - minimum pitch reached in that window (the deepest point)
  - average volume (RMS)
  - whether that window is a "deep voice" moment and/or a volume spike,
    relative to the speaker's own rolling baseline for this clip.
"""

from dataclasses import dataclass
import numpy as np
import librosa
import config


@dataclass
class VoiceWindowStats:
    avg_pitch_hz: float
    min_pitch_hz: float
    avg_volume_rms: float
    is_deep_voice: bool
    is_volume_spike: bool
    pitch_risk_score: float  # 0-1, how much this window looks like an aggressive/deep/loud moment


class PitchAnalyzer:
    def __init__(self, audio: np.ndarray, sample_rate: int):
        self.audio = audio
        self.sr = sample_rate

        # Extract pitch contour for the WHOLE clip once (fast to reuse per-window).
        f0, voiced_flag, _voiced_probs = librosa.pyin(
            audio,
            fmin=config.F0_MIN_HZ,
            fmax=config.F0_MAX_HZ,
            sr=sample_rate,
        )
        self.f0 = np.nan_to_num(f0, nan=0.0)
        self.voiced_flag = voiced_flag if voiced_flag is not None else np.zeros_like(self.f0, dtype=bool)
        self.hop_length = 512  # librosa.pyin default
        self.frame_times = librosa.frames_to_time(
            np.arange(len(self.f0)), sr=sample_rate, hop_length=self.hop_length
        )

        # RMS volume contour
        self.rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        self.rms_times = librosa.frames_to_time(
            np.arange(len(self.rms)), sr=sample_rate, hop_length=self.hop_length
        )

        # Rolling baselines across the whole clip (used to detect relative dips/spikes)
        voiced_f0 = self.f0[(self.f0 > 0) & self.voiced_flag]
        self.baseline_pitch = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else config.DEEP_VOICE_ABS_THRESHOLD_HZ
        self.baseline_volume = float(np.median(self.rms)) if len(self.rms) > 0 else 0.0

    def get_window_stats(self, start: float, end: float) -> VoiceWindowStats:
        pitch_mask = (self.frame_times >= start) & (self.frame_times <= end)
        vol_mask = (self.rms_times >= start) & (self.rms_times <= end)

        window_f0 = self.f0[pitch_mask]
        window_voiced = window_f0[window_f0 > 0]
        window_rms = self.rms[vol_mask]

        avg_pitch = float(np.mean(window_voiced)) if len(window_voiced) > 0 else 0.0
        min_pitch = float(np.min(window_voiced)) if len(window_voiced) > 0 else 0.0
        avg_volume = float(np.mean(window_rms)) if len(window_rms) > 0 else 0.0

        is_deep_abs = 0 < avg_pitch < config.DEEP_VOICE_ABS_THRESHOLD_HZ
        is_deep_relative = (
            avg_pitch > 0
            and self.baseline_pitch > 0
            and avg_pitch <= self.baseline_pitch * config.DEEP_VOICE_DROP_RATIO
        )
        is_deep_voice = is_deep_abs or is_deep_relative

        is_volume_spike = (
            self.baseline_volume > 0
            and avg_volume >= self.baseline_volume * config.VOLUME_SPIKE_RATIO
        )

        # Simple 0-1 risk score: deep voice and volume spike each contribute;
        # both together (shouting in a deep/aggressive tone) scores highest.
        pitch_risk_score = 0.0
        if is_deep_voice:
            pitch_risk_score += 0.5
        if is_volume_spike:
            pitch_risk_score += 0.5
        pitch_risk_score = min(pitch_risk_score, 1.0)

        return VoiceWindowStats(
            avg_pitch_hz=avg_pitch,
            min_pitch_hz=min_pitch,
            avg_volume_rms=avg_volume,
            is_deep_voice=is_deep_voice,
            is_volume_spike=is_volume_spike,
            pitch_risk_score=pitch_risk_score,
        )
