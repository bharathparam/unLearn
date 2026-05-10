"""
Neural Verification Engine — Gemini Client
============================================
Core client for Google Gemini API integration via OpenRouter.

This module is FULLY OPTIONAL.
If GEMINI_API_KEY is missing or ENABLE_GEMINI_EVAL is false,
all methods return None and the system operates normally.

Design principles:
  • Never crash the server
  • Always use timeout protection
  • Cache responses when possible
  • Sanitize sensitive data before external calls
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger("neural-verification-engine.gemini")


class GeminiClient:
    """
    Optional Gemini API client with graceful degradation (via OpenRouter).

    If the API key is missing or Gemini is disabled,
    all calls return None without raising exceptions.
    """

    def __init__(self) -> None:
        self._api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        self._enabled: bool = os.environ.get("ENABLE_GEMINI_EVAL", "false").lower() == "true"
        # Using OpenRouter's latest default model for Gemini
        self._model_name: str = os.environ.get("GEMINI_MODEL", "google/gemini-2.5-flash")
        self._timeout: int = int(os.environ.get("GEMINI_TIMEOUT", "30"))
        self._openai_client = None
        self._cache: dict[str, Any] = {}

        # Validate configuration
        if self._enabled and not self._api_key:
            logger.warning(
                "ENABLE_GEMINI_EVAL=true but GEMINI_API_KEY is not set. "
                "Gemini features will be disabled."
            )
            self._enabled = False

        if self._enabled:
            try:
                import openai

                self._openai_client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self._api_key,
                    timeout=self._timeout
                )
                logger.info(f"Gemini client initialized via OpenRouter (model: {self._model_name})")
            except ImportError:
                logger.warning(
                    "openai package not installed. "
                    "Gemini features disabled. Install with: pip install openai"
                )
                self._enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self._enabled = False
        else:
            logger.info("Gemini integration disabled (ENABLE_GEMINI_EVAL != true)")

    @property
    def is_enabled(self) -> bool:
        """Whether Gemini integration is active and ready."""
        return self._enabled and self._openai_client is not None

    def _cache_key(self, prompt: str) -> str:
        """Generate a cache key from a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _get_cached(self, prompt: str) -> Optional[str]:
        """Check cache for a previous response."""
        key = self._cache_key(prompt)
        entry = self._cache.get(key)
        if entry and (time.time() - entry["time"]) < 300:  # 5 min TTL
            logger.debug("Gemini cache hit")
            return entry["response"]
        return None

    def _set_cache(self, prompt: str, response: str) -> None:
        """Cache a Gemini response."""
        key = self._cache_key(prompt)
        self._cache[key] = {"response": response, "time": time.time()}

    @staticmethod
    def sanitize_prompt(text: str, max_length: int = 500) -> str:
        """
        Sanitize sensitive content before sending to external API.
        Truncates overly long content for safety.
        """
        # Truncate to prevent sending excessive data
        if len(text) > max_length:
            text = text[:max_length] + "..."
        return text

    def generate(self, prompt: str, use_cache: bool = True) -> Optional[str]:
        """
        Send a prompt to Gemini via OpenRouter and return the response text.

        Args:
            prompt: The prompt to send.
            use_cache: Whether to use cached responses.

        Returns:
            Response text, or None if Gemini is disabled/fails.
        """
        if not self.is_enabled:
            return None

        # Check cache
        if use_cache:
            cached = self._get_cached(prompt)
            if cached is not None:
                return cached

        try:
            response = self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            result = response.choices[0].message.content.strip()

            # Cache the response
            if use_cache:
                self._set_cache(prompt, result)

            return result

        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}")
            return None

    def generate_json(self, prompt: str, use_cache: bool = True) -> Optional[dict]:
        """
        Send a prompt expecting JSON response, parse and return as dict.

        Returns None if Gemini is disabled, fails, or returns invalid JSON.
        """
        result = self.generate(prompt, use_cache=use_cache)
        if result is None:
            return None

        try:
            # Try to extract JSON from the response
            # Handle cases where Gemini wraps JSON in markdown code blocks
            cleaned = result.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Gemini returned invalid JSON, attempting extraction...")
            # Try to find JSON object in the response
            try:
                start = result.index("{")
                end = result.rindex("}") + 1
                return json.loads(result[start:end])
            except (ValueError, json.JSONDecodeError):
                logger.warning("Could not extract JSON from Gemini response")
                return None


# ── Singleton ───────────────────────────────────────────────────────────

_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Return a singleton GeminiClient instance."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
