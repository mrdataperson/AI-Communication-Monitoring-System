"""
Configuration for AI Communication Monitoring System.
Fill in your Twilio credentials and emergency contact before running.
"""

import os

# ---------------- Twilio (SMS alert) ----------------
# Get these from https://console.twilio.com
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "your_account_sid_here")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "your_auth_token_here")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+1XXXXXXXXXX")   # Twilio number you bought
EMERGENCY_CONTACT_NUMBER = os.getenv("EMERGENCY_CONTACT_NUMBER", "+91XXXXXXXXXX")  # who receives the alert

# ---------------- Whisper (speech-to-text) ----------------
WHISPER_MODEL_SIZE = "base"     # tiny/base/small/medium/large-v3 (bigger = more accurate, slower)
WHISPER_DEVICE = "cpu"          # "cuda" if you have a GPU
WHISPER_COMPUTE_TYPE = "int8"   # int8 is fast on CPU

# ---------------- Text threat/harm classification ----------------
# Zero-shot model -> works on MEANING of the sentence, not a keyword list
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
THREAT_LABELS = [
    "threat of violence",
    "verbal abuse or harassment",
    "distress call or plea for help",
    "normal calm conversation",
]
# Any label except "normal calm conversation" that scores above this = semantically harmful
TEXT_HARM_SCORE_THRESHOLD = 0.55

# ---------------- Pitch / voice depth analysis ----------------
# "Deep voice" = low fundamental frequency (F0). Typical adult male speech ~85-180 Hz,
# adult female ~165-255 Hz. A sudden drop below the speaker's own baseline, or an
# unusually low absolute F0 combined with a loud/sharp volume, is what we treat as
# an aggressive "deep voice" signature.
F0_MIN_HZ = 50
F0_MAX_HZ = 400
DEEP_VOICE_ABS_THRESHOLD_HZ = 120     # absolute pitch below this counts as "deep"
DEEP_VOICE_DROP_RATIO = 0.75          # pitch drop to 75% of rolling baseline counts as "deepening"
VOLUME_SPIKE_RATIO = 1.6              # RMS energy 1.6x above rolling baseline counts as a spike

# ---------------- Fusion / final risk score ----------------
# final_risk = TEXT_WEIGHT * text_harm_score + PITCH_WEIGHT * pitch_risk_score
TEXT_WEIGHT = 0.6
PITCH_WEIGHT = 0.4
ALERT_RISK_THRESHOLD = 0.65   # fire SMS alert if final_risk >= this

# ---------------- Live monitoring ----------------
LIVE_CHUNK_SECONDS = 4          # analyze audio in rolling 4-second windows
LIVE_SAMPLE_RATE = 16000
