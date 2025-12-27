"""Audio event detectors: BeepDetector and SilenceDetector.
- BeepDetector: FFT-based detection tuned for beep tone energy in a target frequency band.
- SilenceDetector: WebRTC VAD-backed detector that tracks speech/silence durations.
"""

import time
import numpy as np


class BeepDetector:
    """Detect beep using FFT and short-term energy tracking."""
    def __init__(self):
        self.beep_detected = False
        self.last_beep_time = None
        self.confidence = 0
        self.recent_beep_detections = []
        self.min_beep_duration = 0.3
        self.min_beep_count = 2
        self.recent_energies = []

    def process_chunk(self, audio_chunk, sample_rate):
        """Process a chunk of audio and return True if a beep is detected.
        Expects a 1-D or 2-D numpy array for audio_chunk and integer sample_rate.
        """
        if len(audio_chunk) == 0:
            return False
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        n = len(audio_chunk)
        if n < 1024:
            return False
        window = np.hanning(n)
        windowed_chunk = audio_chunk * window
        freq = np.fft.fftfreq(n, d=1/sample_rate)
        fft_result = np.fft.fft(windowed_chunk)
        target_freq_range = (900, 1100)
        freq_mask = (freq >= target_freq_range[0]) & (freq <= target_freq_range[1])
        energy = np.sum(np.abs(fft_result[freq_mask]))
        total_energy = np.sum(np.abs(fft_result))
        if total_energy > 0 and energy / total_energy > 0.08 and np.max(np.abs(audio_chunk)) > 0.1:
            self.recent_energies.append(energy / total_energy)
            if len(self.recent_energies) > 5:
                self.recent_energies.pop(0)
            avg_energy = sum(self.recent_energies) / len(self.recent_energies)
            if avg_energy > 0.08:
                self.recent_beep_detections.append(time.time())
                if len(self.recent_beep_detections) >= self.min_beep_count:
                    if time.time() - self.recent_beep_detections[0] < self.min_beep_duration:
                        self.beep_detected = True
                        self.last_beep_time = time.time()
                        self.confidence = min(1.0, avg_energy)
                        self.recent_beep_detections = []
                        return True
        return False

    def reset(self):
        """Reset internal state used for beep detection."""
        self.beep_detected = False
        self.last_beep_time = None
        self.confidence = 0
        self.recent_beep_detections = []
        self.recent_energies = []


class SilenceDetector:
    """WebRTC VAD based silence detector that measures silence duration."""

    def __init__(self, sample_rate=8000):
        """Initialize the VAD and silence-tracking state."""
        import webrtcvad
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(2)
        self.silence_start = None
        self.last_speech_time = time.time()
        self.silence_duration = 0
        self.speech_active = False
        self.consecutive_silence = 0
        self.consecutive_speech = 0

    def process_chunk(self, audio_chunk):
        """Process an audio chunk and return current silence duration in seconds."""
        if len(audio_chunk) == 0:
            return self.silence_duration
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk[:, 0]
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 0:
            audio_chunk = audio_chunk / max_val
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        is_speech = False
        try:
            frame_duration = 0.03
            frame_size = int(self.sample_rate * frame_duration)
            if len(audio_int16) >= frame_size:
                frame = audio_int16[:frame_size]
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
        except Exception:
            is_speech = False
        current_time = time.time()
        if is_speech:
            self.consecutive_speech += 1
            self.consecutive_silence = 0
            if self.consecutive_speech >= 2:
                self.speech_active = True
                self.last_speech_time = current_time
                self.silence_start = None
                self.silence_duration = 0
        else:
            self.consecutive_silence += 1
            self.consecutive_speech = 0
            if self.consecutive_silence >= 3:
                if self.speech_active:
                    self.silence_start = current_time
                    self.speech_active = False
                elif self.silence_start:
                    self.silence_duration = current_time - self.silence_start
        return self.silence_duration

    def reset(self):
        """Reset silence detector internal timers and counters."""
        import time
        self.silence_start = None
        self.last_speech_time = time.time()
        self.silence_duration = 0
        self.speech_active = False
        self.consecutive_silence = 0
        self.consecutive_speech = 0
