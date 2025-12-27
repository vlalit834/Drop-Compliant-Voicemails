"""
Core VoicemailDropper orchestrator.

This module loads audio files, runs detectors and STT processing, determines
drop points, and writes dropped output files via utils.insert_voice_mail_at_drop.
"""
import os
import time
from detectors import BeepDetector, SilenceDetector
from stt import SpeechToTextProcessor
from utils import insert_voice_mail_at_drop


class VoicemailDropper:
    """
    High-level processor that scans audio files for beep/silence+complete greeting
    conditions and triggers insertion of the provided voice mail file.
    """
    def __init__(self, github_token=None):
        self.beep_detector = BeepDetector()
        self.stt_processor = SpeechToTextProcessor(github_token)
        self.silence_detector = None
        self.start_time = time.time()
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = None
        self.processed_duration = 0
        self.total_duration = 0

    def process_audio_stream(self, audio_file):
        """
        Process a single audio file stream in fixed-size chunks.

        Returns a tuple (trigger_time_seconds, trigger_reason).
        """
        self.start_time = time.time()
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = None
        self.processed_duration = 0
        self.beep_detector.reset()
        self.stt_processor.reset()
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(audio_file)
            self.total_duration = len(audio) / sample_rate
            print(f"Loaded: {audio_file} | duration: {self.total_duration:.2f}s | sr: {sample_rate} | shape: {audio.shape}")
            self.silence_detector = SilenceDetector(sample_rate)
        except Exception as e:
            print(f"Error loading audio file {audio_file}: {e}")
            return None, None
        chunk_size = int(sample_rate * 0.1)
        total_chunks = len(audio) // chunk_size
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = audio[start_idx:end_idx]
            if len(chunk) == 0:
                continue
            self.processed_duration = (i + 1) * 0.1
            if i % int(1.0 / 0.1) == 0:
                print(f"Processing chunk {i}/{total_chunks}, {self.processed_duration:.2f}/{self.total_duration:.2f}s")
            beep_detected = self.beep_detector.process_chunk(chunk, sample_rate)
            silence_duration = self.silence_detector.process_chunk(chunk)
            silence_detected = silence_duration >= 0.6
            stt_processed = self.stt_processor.process_chunk(chunk, sample_rate, self.processed_duration, self.total_duration)
            current_context = self.stt_processor.get_current_context()
            print(f"Beep: {beep_detected}, Silence: {silence_duration:.2f}s, Silence met: {silence_detected}, Context len: {len(current_context)}")
            if not self.triggered:
                if beep_detected:
                    self._trigger_drop("beep_detected", self.processed_duration)
                    break
                elif silence_detected:
                    if current_context and len(current_context) > 10:
                        print(f"Analyzing context: '{current_context}'")
                        if self.stt_processor.llm_analyzer.is_greeting_complete(current_context):
                            self._trigger_drop("silence_and_complete_greeting", self.processed_duration)
                            break
                        else:
                            print("Sentence incomplete - continuing")
                    else:
                        print("Not enough context - continuing")
            time.sleep(0.05)
        if not self.triggered:
            if self.silence_detector and self.silence_detector.last_speech_time > self.start_time + 1.0:
                self._trigger_drop("end_of_speech", self.total_duration * 0.9)
            else:
                self._trigger_drop("end_of_audio", self.total_duration * 0.9)
        return (self.trigger_time, self.trigger_reason)

    def _trigger_drop(self, reason, timestamp):
        """
        Mark the drop as triggered with the given reason and timestamp.
        """
        self.triggered = True
        self.trigger_time = timestamp
        self.trigger_reason = reason
        print(f"\nVOICEMAIL TRIGGERED: {reason} at {timestamp:.2f}s")

    def process_directory(self, directory, voice_mail_path='voice_mail.wav', output_dir='output'):
        """
        Process all .wav files in `directory` (excluding the voice_mail file) and
        create output files with the voicemail inserted at detected drop points.

        Returns:
            dict mapping filename -> result metadata
        """
        results = {}
        audio_files = [f for f in os.listdir(directory) if f.endswith('.wav') and f != os.path.basename(voice_mail_path)]
        print(f"Found {len(audio_files)} files in {directory}")
        os.makedirs(output_dir, exist_ok=True)
        for filename in sorted(audio_files):
            filepath = os.path.join(directory, filename)
            print('\n' + '='*40)
            print(f"Processing {filename}...")
            timestamp, reason = self.process_audio_stream(filepath)
            if timestamp is not None:
                base, _ = os.path.splitext(filename)
                out_name = f"{base}_dropped.wav"
                out_path = os.path.join(output_dir, out_name)
                try:
                    insert_voice_mail_at_drop(filepath, voice_mail_path, timestamp, out_path)
                    status = 'SUCCESS'
                    print(f"Saved dropped file to {out_path}")
                except Exception as e:
                    print(f"Error inserting voice mail for {filename}: {e}")
                    status = 'FAILED'
                results[filename] = {'timestamp': timestamp, 'reason': reason, 'status': status, 'output_file': out_path}
                print(f"Result: {filename} - Drop at {timestamp:.2f}s ({reason})")
            else:
                results[filename] = {'timestamp': None, 'reason': reason, 'status': 'FAILED', 'output_file': None}
                print(f"Result: {filename} - Processing failed")
        return results
