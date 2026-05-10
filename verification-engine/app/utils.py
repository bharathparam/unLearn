"""
Neural Verification Engine — Utilities
========================================
Shared helpers for text processing, logging, and file I/O.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Text Processing
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = normalize_text(text)
    # Split on non-alphanumeric, keep tokens
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def extract_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character-level n-grams from normalized text."""
    text = normalize_text(text)
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def extract_keywords(text: str, min_length: int = 3) -> set[str]:
    """Extract meaningful keywords (tokens longer than min_length)."""
    tokens = tokenize(text)
    # Filter out very short / common words
    stopwords = {
        "the", "is", "are", "was", "were", "a", "an", "and", "or", "but",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "it",
        "this", "that", "not", "no", "do", "don", "does", "did", "has",
        "have", "had", "will", "would", "could", "should", "may", "might",
        "can", "shall", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "out", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "as", "until", "while", "what", "which", "who",
        "whom", "i", "you", "he", "she", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their", "know",
    }
    return {t for t in tokens if len(t) >= min_length and t not in stopwords}


# ═══════════════════════════════════════════════════════════════════════════
# Timestamps & IDs
# ═══════════════════════════════════════════════════════════════════════════

def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def generate_id(prefix: str = "NVE") -> str:
    """Generate a unique ID with a prefix."""
    short_uuid = uuid.uuid4().hex[:12].upper()
    return f"{prefix}-{short_uuid}"


# ═══════════════════════════════════════════════════════════════════════════
# JSON File I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_json(data: Any, filepath: Path) -> Path:
    """Save data as pretty-printed JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=str,  # handles datetime serialization
            ensure_ascii=False,
        )
    return filepath


def load_json(filepath: Path) -> Any:
    """Load JSON from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(directory: Path) -> list[Path]:
    """List all JSON files in a directory, sorted by modification time (newest first)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


# ═══════════════════════════════════════════════════════════════════════════
# Logging Helpers
# ═══════════════════════════════════════════════════════════════════════════

def log_attack(
    logs_dir: Path,
    attack_id: int,
    prompt: str,
    leakage_score: float,
    leakage_detected: bool,
    details: dict[str, Any] | None = None,
) -> Path:
    """Log a single attack attempt to the logs directory."""
    log_entry = {
        "attack_id": attack_id,
        "timestamp": utc_now().isoformat(),
        "prompt": prompt,
        "leakage_score": leakage_score,
        "leakage_detected": leakage_detected,
        "details": details or {},
    }

    timestamp_str = utc_now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = logs_dir / f"attack_{timestamp_str}_{attack_id}.json"
    return save_json(log_entry, filepath)
