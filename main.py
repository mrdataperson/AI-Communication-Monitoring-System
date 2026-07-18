"""
AI Communication Monitoring System
------------------------------------
Entry point. Choose file-upload mode or live-monitoring mode.

Pipeline (both modes):
  audio -> transcribe (word-level timestamps)
        -> per sentence: semantic harm classification (meaning-based, not keywords)
        -> per sentence: pitch depth + volume analysis at that exact time window
        -> fusion: combine both into one risk score
        -> if risk score crosses threshold: fire SMS alert to emergency contact
"""

import sys
import numpy as np

import config
from audio_input import load_file, LiveRecorder
from transcriber import Transcriber
from sentiment_analyzer import SentimentAnalyzer
from pitch_analyzer import PitchAnalyzer
from fusion_engine import FusionEngine
from alert_system import AlertSystem


def build_pipeline():
    transcriber = Transcriber()
    sentiment_analyzer = SentimentAnalyzer()
    fusion_engine = FusionEngine(sentiment_analyzer)
    alert_system = AlertSystem()
    return transcriber, fusion_engine, alert_system


def process_audio_block(audio: np.ndarray, sample_rate: int, transcriber, fusion_engine, alert_system):
    """Run the full pipeline over one block of audio (a file or a live chunk)."""
    segments = transcriber.transcribe(audio_array=audio, sample_rate=sample_rate)
    if not segments:
        print("[main] No speech detected in this block.")
        return

    pitch_analyzer = PitchAnalyzer(audio, sample_rate)

    for segment in segments:
        print(f"\n[Transcript {segment.start:.1f}s-{segment.end:.1f}s] {segment.text}")
        event = fusion_engine.evaluate_segment(segment, pitch_analyzer)
        if event:
            print(f"  -> FLAGGED. risk={event.final_risk:.2f}, "
                  f"label='{event.harm_result.label}', "
                  f"deep_voice={event.voice_stats.is_deep_voice}, "
                  f"volume_spike={event.voice_stats.is_volume_spike}")
            alert_system.send_alert(event)
        else:
            print("  -> ok")


def run_file_mode(transcriber, fusion_engine, alert_system):
    path = input("Enter path to audio file (wav/mp3/etc): ").strip()
    audio, sr = load_file(path, target_sr=config.LIVE_SAMPLE_RATE)
    process_audio_block(audio, sr, transcriber, fusion_engine, alert_system)


def run_live_mode(transcriber, fusion_engine, alert_system):
    recorder = LiveRecorder()
    recorder.start()
    print("Press Ctrl+C to stop live monitoring.\n")
    try:
        while True:
            chunk = recorder.read_chunk()
            process_audio_block(chunk, recorder.sample_rate, transcriber, fusion_engine, alert_system)
    except KeyboardInterrupt:
        print("\n[main] Stopping live monitoring...")
    finally:
        recorder.stop()


def main():
    print("=" * 60)
    print("AI Communication Monitoring System")
    print("=" * 60)
    print("1. File upload mode")
    print("2. Live monitoring mode")
    choice = input("Choose mode (1/2): ").strip()

    transcriber, fusion_engine, alert_system = build_pipeline()

    if choice == "1":
        run_file_mode(transcriber, fusion_engine, alert_system)
    elif choice == "2":
        run_live_mode(transcriber, fusion_engine, alert_system)
    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    main()
