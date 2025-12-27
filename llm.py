"""LLM-based voicemail greeting analyzer.
Provides a small wrapper around an LLM (via OpenAI client when available)
to decide whether a voicemail greeting is complete (ready for a message).
Falls back to a lightweight heuristic if the LLM client is not available.
"""

import time

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class LLMGreetingAnalyzer:
    """Analyze voicemail greeting excerpts to determine completeness.

    Attributes:
        client: Optional OpenAI-compatible client instance.
        last_call_time: Timestamp of last LLM request for simple rate limiting.
        cache: Simple in-memory cache to avoid repeat LLM calls.
        rate_limit: Minimum seconds between LLM calls.
        model: Model id to request when using the client.
        github_token: Optional token used to initialize a GitHub-hosted model endpoint.
    """
    def __init__(self, github_token=None, rate_limit=2.0):
        """Initialize analyzer with optional github_token and rate limiting."""
        self.client = None
        self.last_call_time = 0
        self.cache = {}
        self.rate_limit = rate_limit
        self.model = "openai/gpt-4.1"
        self.github_token = github_token
        if github_token and OpenAI:
            try:
                self.client = OpenAI(base_url="https://models.github.ai/inference", api_key=github_token)
            except Exception as e:
                print(f"LLM init error: {e}")
                self.client = None

    def analyze_last_sentence(self, text):
        """Return (is_complete: bool, raw_result: str) for the provided greeting excerpt.

        Uses cached result if recent, enforces rate limiting, and tries the LLM client
        when available. On failure or no client, falls back to heuristic analysis.
        """
        normalized_text = text.strip().lower()
        cache_key = normalized_text[:100]
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result['timestamp'] < 60:
                return cached_result['is_complete'], cached_result['raw_result']
        current_time = time.time()
        if current_time - self.last_call_time < self.rate_limit:
            wait_time = self.rate_limit - (current_time - self.last_call_time)
            time.sleep(wait_time)
        if not self.client:
            return self._heuristic_analysis(text), "HEURISTIC"
        try:
            prompt = f"""Analyze this voicemail greeting excerpt to determine if the speaker has finished their greeting and is ready for a message.\n\nGreeting excerpt: \"{text}\"\n\nRespond with ONLY 'COMPLETE' or 'INCOMPLETE'."""
            self.last_call_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=20,
                timeout=15.0,
            )
            result = response.choices[0].message.content.strip().upper()
            is_complete = result == "COMPLETE"
            self.cache[cache_key] = {'is_complete': is_complete, 'raw_result': result, 'timestamp': time.time()}
            return is_complete, result
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self._heuristic_analysis(text), "HEURISTIC_FALLBACK"

    def _heuristic_analysis(self, text):
        """Lightweight keyword-based heuristic to guess whether a greeting is complete."""
        text_lower = text.lower()
        complete_indicators = [
            "leave a message",
            "after the beep",
            "after the tone",
            "call me back",
            "thank you",
            "goodbye",
            "leave your name",
            "i'll call you",
            "message after",
            "beep and then",
        ]
        incomplete_indicators = [
            "hi this is",
            "hello this is",
            "you've reached",
            "i am",
            "my name is",
            "i'm not available",
            "sorry i missed",
        ]
        complete_score = sum(1 for indicator in complete_indicators if indicator in text_lower)
        incomplete_score = sum(1 for indicator in incomplete_indicators if indicator in text_lower)
        if complete_score > incomplete_score or (complete_score > 0 and len(text) > 10):
            return True
        return False

    def is_greeting_complete(self, text):
        """Public convenience method returning a boolean indicating completeness."""
        if not text or len(text.strip()) < 5:
            return False
        is_complete, _ = self.analyze_last_sentence(text.strip())
        return is_complete
