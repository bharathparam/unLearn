"""
Neural Verification Engine — Main Application
===============================================
FastAPI application entry point.

  ┌─────────────────────────────────────────────────┐
  │  NEURAL VERIFICATION ENGINE v1.0.0              │
  │  AI Memory Verification & Neural Security       │
  │                                                 │
  │  Endpoints:                                     │
  │    POST /verify   — Forgetting verification     │
  │    POST /attack   — Adversarial attack suite    │
  │    POST /report   — Audit report generation     │
  │    GET  /reports   — List all reports            │
  │    GET  /          — Health check                │
  │    GET  /docs      — Interactive API docs        │
  └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import HealthResponse
from app.routes import attack, report, verify, mia
from app.utils import utc_now

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neural-verification-engine")


# ── Lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    settings = get_settings()
    logger.info("═" * 60)
    logger.info("  NEURAL VERIFICATION ENGINE")
    logger.info("  AI Memory Verification & Neural Security System")
    logger.info("═" * 60)
    logger.info(f"  Version  : {settings.APP_VERSION}")
    logger.info(f"  Host     : {settings.HOST}:{settings.PORT}")
    logger.info(f"  Logs     : {settings.LOGS_DIR}")
    logger.info(f"  Reports  : {settings.REPORTS_DIR}")
    logger.info(f"  Debug    : {settings.DEBUG}")

    # ── Gemini Integration Status ───────────────────────────────────
    try:
        from app.gemini.gemini_client import get_gemini_client
        gemini_client = get_gemini_client()
        gemini_status = "ENABLED" if gemini_client.is_enabled else "DISABLED"
    except Exception:
        gemini_status = "DISABLED"
    logger.info(f"  Gemini   : {gemini_status}")

    logger.info("═" * 60)
    logger.info("  Engine ONLINE — Ready to verify neural forgetting")
    logger.info("═" * 60)
    yield
    logger.info("Neural Verification Engine shutting down...")


# ── FastAPI App ─────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI Memory Verification & Neural Security Engine. "
        "Tests whether an LLM has truly forgotten sensitive information "
        "by running adversarial attack suites and computing privacy metrics."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routes ────────────────────────────────────────────────────
app.include_router(verify.router)
app.include_router(attack.router)
app.include_router(report.router)
app.include_router(mia.router)


# ── Root Endpoint ───────────────────────────────────────────────────────
@app.get(
    "/",
    response_model=HealthResponse,
    summary="Engine Health Check",
    tags=["System"],
)
async def root() -> HealthResponse:
    """Neural Verification Engine health check and status."""
    # Check Gemini status
    gemini_status = "disabled"
    try:
        from app.gemini.gemini_client import get_gemini_client
        client = get_gemini_client()
        gemini_status = "enabled" if client.is_enabled else "disabled"
    except Exception:
        pass

    return HealthResponse(
        status="operational",
        engine=settings.APP_NAME,
        version=settings.APP_VERSION,
        gemini_integration=gemini_status,
        timestamp=utc_now(),
    )
