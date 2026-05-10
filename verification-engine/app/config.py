"""
Neural Verification Engine — Configuration
============================================
Centralized settings and threshold configuration for the verification engine.
All values are overridable via environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration with environment variable support."""

    # ── Server ──────────────────────────────────────────────────────────
    APP_NAME: str = "Neural Verification Engine"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ── Paths ───────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOGS_DIR: Path = BASE_DIR / "logs"
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # ── Scoring Thresholds ──────────────────────────────────────────────
    LEAKAGE_THRESHOLD_LOW: float = 0.10       # below → FORGOTTEN
    LEAKAGE_THRESHOLD_MEDIUM: float = 0.40    # below → PARTIALLY_FORGOTTEN
    # above MEDIUM → NOT_FORGOTTEN

    # ── Scoring Weights ─────────────────────────────────────────────────
    WEIGHT_STRING_SIMILARITY: float = 0.35
    WEIGHT_TOKEN_OVERLAP: float = 0.30
    WEIGHT_NGRAM_MATCH: float = 0.20
    WEIGHT_SEMANTIC_PROXIMITY: float = 0.15

    # ── Risk Level Thresholds ───────────────────────────────────────────
    RISK_NEGLIGIBLE: float = 0.05
    RISK_LOW: float = 0.15
    RISK_MEDIUM: float = 0.35
    RISK_HIGH: float = 0.60
    # above HIGH → CRITICAL

    # ── Attack Engine ───────────────────────────────────────────────────
    ATTACK_SUCCESS_THRESHOLD: float = 0.30    # leakage score above this = successful attack

    model_config = {"env_prefix": "NVE_"}


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# ── Ensure runtime directories exist ────────────────────────────────────
settings = get_settings()
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
