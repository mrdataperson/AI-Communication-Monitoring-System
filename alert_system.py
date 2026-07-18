"""
Emergency alert system. Fires an SMS via Twilio the moment a harmful event
is detected, including the flagged sentence, the exact words, and the voice
metrics (deep voice / volume spike) that contributed to the decision.
"""

from twilio.rest import Client
import config
from fusion_engine import FlaggedEvent


class AlertSystem:
    def __init__(self):
        self._client = None  # lazy init so importing this module doesn't require valid creds

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        return self._client

    def build_message(self, event: FlaggedEvent) -> str:
        voice_tags = []
        if event.voice_stats.is_deep_voice:
            voice_tags.append(f"deep voice ({event.voice_stats.avg_pitch_hz:.0f} Hz)")
        if event.voice_stats.is_volume_spike:
            voice_tags.append("volume spike")
        voice_desc = ", ".join(voice_tags) if voice_tags else "normal tone"

        message = (
            "[ALERT] Potential harmful communication detected.\n"
            f"Time: {event.start:.1f}s-{event.end:.1f}s\n"
            f"Sentence: \"{event.sentence}\"\n"
        )
        if getattr(event, "sentence_en", None) and event.sentence_en != event.sentence:
            message += f"(EN): \"{event.sentence_en}\"\n"
        message += (
            f"Classification: {event.harm_result.label} ({event.harm_result.score:.2f})\n"
            f"Voice signature: {voice_desc}\n"
            f"Risk score: {event.final_risk:.2f}"
        )
        return message

    def send_alert(self, event: FlaggedEvent) -> bool:
        message_body = self.build_message(event)
        print(f"[AlertSystem] FIRING ALERT:\n{message_body}\n")
        try:
            client = self._get_client()
            client.messages.create(
                body=message_body,
                from_=config.TWILIO_FROM_NUMBER,
                to=config.EMERGENCY_CONTACT_NUMBER,
            )
            print("[AlertSystem] SMS sent successfully.")
            return True
        except Exception as e:
            print(f"[AlertSystem] Failed to send SMS: {e}")
            return False
