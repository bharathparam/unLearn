"""
Neural Verification Engine — Verify Routes
============================================
POST /verify  — Run the full forgetting verification pipeline
GET  /verify/health — Engine health check

Includes optional Gemini semantic analysis when enabled.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import (
    GeminiAuditSummaryResponse,
    GeminiSemanticAnalysisResponse,
    VerificationRequest,
    VerificationResponse,
)
from app.verifier import ForgettingVerifier

router = APIRouter(prefix="/verify", tags=["Verification"])

_verifier = ForgettingVerifier()


@router.post(
    "",
    response_model=VerificationResponse,
    summary="Run forgetting verification",
    description=(
        "Execute the full Neural Verification pipeline: "
        "compare before/after model outputs, run adversarial attacks, "
        "compute leakage scores, and return a comprehensive verification result. "
        "Includes optional Gemini semantic analysis when enabled."
    ),
)
async def verify_forgetting(request: VerificationRequest) -> VerificationResponse:
    """
    Main verification endpoint.

    Accepts a secret, before_output, and after_output.
    Returns structured verification metrics including:
    - verification_status (FORGOTTEN / PARTIALLY_FORGOTTEN / NOT_FORGOTTEN)
    - privacy_confidence (0-100)
    - attack_success_rate
    - leakage_probability
    - risk_level
    - full attack details
    - optional gemini_semantic_analysis (when ENABLE_GEMINI_EVAL=true)
    - optional gemini_audit_summary (when ENABLE_GEMINI_EVAL=true)
    """
    try:
        result = _verifier.verify(request)

        # ── Optional Gemini Semantic Analysis ───────────────────────────
        try:
            from app.gemini.semantic_eval import SemanticLeakageEvaluator

            evaluator = SemanticLeakageEvaluator()
            gemini_analysis = evaluator.to_analysis_block(
                secret=request.secret,
                model_output=request.after_output,
            )
            result.gemini_semantic_analysis = GeminiSemanticAnalysisResponse(
                enabled=gemini_analysis.enabled,
                semantic_leakage_detected=gemini_analysis.semantic_leakage_detected,
                leakage_type=gemini_analysis.leakage_type,
                confidence=gemini_analysis.confidence,
                summary=gemini_analysis.summary,
            )

            # ── Override Local Heuristics with Gemini Semantics ─────────
            if gemini_analysis.enabled and gemini_analysis.semantic_leakage_detected is False:
                from app.models import VerificationStatus, RiskLevel
                result.verification_status = VerificationStatus.FORGOTTEN
                result.risk_level = RiskLevel.NEGLIGIBLE
                result.leakage_probability = 0.0
                result.attack_success_rate = 0.0
                result.attacks_succeeded = 0
                result.attacks_failed = result.attacks_total
                result.after_leakage_score = 0.0
                result.forgetting_delta = result.before_leakage_score
                for attack in result.attack_details:
                    attack.leakage_score = 0.0
                    attack.leakage_detected = False
        except Exception:
            result.gemini_semantic_analysis = GeminiSemanticAnalysisResponse(enabled=False)

        # ── Optional Gemini Audit Summary ───────────────────────────────
        try:
            from app.gemini.summary_generator import AuditSummaryGenerator

            summary_gen = AuditSummaryGenerator()
            metrics = {
                "verification_status": result.verification_status.value,
                "privacy_confidence": result.privacy_confidence,
                "attack_success_rate": result.attack_success_rate,
                "attacks_total": result.attacks_total,
                "attacks_succeeded": result.attacks_succeeded,
                "risk_level": result.risk_level.value,
                "leakage_probability": result.leakage_probability,
                "before_leakage_score": result.before_leakage_score,
                "after_leakage_score": result.after_leakage_score,
                "forgetting_delta": result.forgetting_delta,
            }
            gemini_summary = summary_gen.generate(metrics)
            result.gemini_audit_summary = GeminiAuditSummaryResponse(
                enabled=gemini_summary.enabled,
                narrative_summary=gemini_summary.narrative_summary,
                risk_narrative=gemini_summary.risk_narrative,
                recommendation=gemini_summary.recommendation,
            )
        except Exception:
            result.gemini_audit_summary = GeminiAuditSummaryResponse(enabled=False)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get(
    "/health",
    summary="Verification engine health",
    description="Check if the verification engine is operational.",
)
async def verify_health() -> dict:
    """Health check for the verification subsystem."""
    # Check Gemini status
    gemini_status = "disabled"
    try:
        from app.gemini.gemini_client import get_gemini_client

        client = get_gemini_client()
        gemini_status = "enabled" if client.is_enabled else "disabled"
    except Exception:
        pass

    return {
        "status": "operational",
        "subsystem": "Forgetting Verifier",
        "attack_prompts_loaded": len(_verifier._attack_engine.prompts),
        "gemini_integration": gemini_status,
    }
