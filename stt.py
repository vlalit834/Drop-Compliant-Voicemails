"""
Transcription simulator and STT processor.

SimulatedDeepgramTranscriber provides staged phrase fragments to mimic
real-time transcription. SpeechToTextProcessor consumes these fragments,
maintains buffers and integrates with LLMGreetingAnalyzer for downstream use.
"""

import time


class SimulatedDeepgramTranscriber:
    """
    Simulate streaming transcription by emitting fragments of canned voicemail phrases.

    The simulator advances fragments at fixed intervals to approximate chunked
    transcription behavior for demo/testing.
    """
    def __init__(self):
        self.voicemail_phrases = [
            "Hi you've reached Mike Rodriguez",
            "Hello this is Mike",
            "You've reached Mike Rodriguez",
            "Hi you've reached Mike Rodriguez I can't take your call right now",
            "Hello this is Mike I'm not available at the moment",
            "You've reached the voicemail of Mike Rodriguez",
            "Hi you've reached Mike Rodriguez I can't take your call right now please leave your name and number after the beep",
            "Hello this is Mike I'm not available right now please leave a message after the tone and I'll get back to you",
            "You've reached Mike Rodriguez I can't come to the phone right now please leave your name number and a brief message after the beep",
        ]
        self.current_phrase = None
        self.phrase_position = 0
        self.last_chunk_time = 0
        self.chunk_interval = 0.4

    def simulate_transcription(self, audio_duration, elapsed_time):
        """
        Simulate returning incremental transcription text.

        Args:
            audio_duration: total duration of the voicemail audio.
            elapsed_time: elapsed time from start of playback/processing.

        Returns:
            New transcription fragment string, or None if nothing new.
        """
        current_time = time.time()
        if current_time - self.last_chunk_time < self.chunk_interval:
            return None
        self.last_chunk_time = current_time
        phrase_index = min(len(self.voicemail_phrases) - 1, int(audio_duration / 5))
        if self.current_phrase is None:
            self.current_phrase = self.voicemail_phrases[phrase_index]
        phrase_length = len(self.current_phrase)
        progress_ratio = min(1.0, elapsed_time / max(3.0, audio_duration * 0.7))
        new_position = int(phrase_length * progress_ratio)
        if new_position > self.phrase_position:
            new_text = self.current_phrase[self.phrase_position:new_position]
            self.phrase_position = new_position
            return new_text
        if elapsed_time > audio_duration * 0.8:
            if "after the beep" not in self.current_phrase.lower() and "message" not in self.current_phrase.lower():
                return " please leave a message after the beep"
        return None

    def reset(self):
        """
        Reset internal simulator state.
        """
        self.current_phrase = None
        self.phrase_position = 0
        self.last_chunk_time = 0


class SpeechToTextProcessor:
    """
    Manage streaming transcription text, buffer recent context, and integrate LLM analysis.

    The class uses SimulatedDeepgramTranscriber for demo transcription fragments and
    LLMGreetingAnalyzer for later analysis (instantiated via github_token if provided).
    """
    def __init__(self, github_token=None):
        from llm import LLMGreetingAnalyzer
        self.transcript = ""
        self.sentence_buffer = ""
        self.last_transcription_time = time.time()
        self.llm_analyzer = LLMGreetingAnalyzer(github_token)
        self.transcriber = SimulatedDeepgramTranscriber()
        self.audio_duration = 0

    def process_chunk(self, audio_chunk, sample_rate, elapsed_time, total_duration):
        """
        Feed an audio chunk to the simulated transcriber and update buffers.

        Returns:
            True if new transcription text was appended; False otherwise.
        """
        self.audio_duration = total_duration
        new_text = self.transcriber.simulate_transcription(total_duration, elapsed_time)
        if new_text:
            self.transcript += new_text
            self.sentence_buffer += new_text
            if len(self.sentence_buffer) > 200:
                self.sentence_buffer = self.sentence_buffer[-200:]
            print(f"New transcription: '{new_text}'")
            print(f"Current buffer: '{self.sentence_buffer}'")
            return True
        return False

    def get_current_context(self):
        """
        Return the most relevant recent text context for downstream processing.
        """
        if len(self.sentence_buffer) > 10:
            return self.sentence_buffer
        return self.transcript[-200:] if self.transcript else ""

    def reset(self):
        """
        Reset internal transcription and analyzer state.
        """
        from llm import LLMGreetingAnalyzer
        self.transcript = ""
        self.sentence_buffer = ""
        self.last_transcription_time = time.time()
        self.transcriber.reset()
        self.llm_analyzer = LLMGreetingAnalyzer(self.llm_analyzer.github_token)
