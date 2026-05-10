"""
Neural Verification Engine — Report Routes
============================================
POST /report             — Generate and save an audit report
GET  /report/{report_id} — Retrieve a saved report
GET  /reports            — List all saved reports

Includes optional Gemini semantic analysis and audit summaries in reports.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import (
    AuditReport,
    GeminiAuditSummaryResponse,
    GeminiSemanticAnalysisResponse,
    ReportListItem,
    ReportRequest,
)
from app.reports import AuditReportGenerator

router = APIRouter(tags=["Audit Reports"])

_generator = AuditReportGenerator()


@router.post(
    "/report",
    response_model=AuditReport,
    summary="Generate audit report",
    description=(
        "Run a full verification pipeline and generate a compliance-style "
        "Neural Forgetting Report. The report is saved to disk for later retrieval. "
        "Includes optional Gemini semantic analysis and audit summary."
    ),
)
async def generate_report(request: ReportRequest) -> AuditReport:
    """
    Generate a full audit report.

    Runs verification, computes all metrics, determines compliance status,
    generates recommendations, and saves the report as JSON.
    Optionally includes Gemini semantic analysis and narrative summaries.
    """
    try:
        report = _generator.generate(request)

        # ── Optional Gemini Semantic Analysis ───────────────────────────
        try:
            from app.gemini.semantic_eval import SemanticLeakageEvaluator

            evaluator = SemanticLeakageEvaluator()
            gemini_analysis = evaluator.to_analysis_block(
                secret=request.secret,
                model_output=request.after_output,
            )
            report.gemini_semantic_analysis = GeminiSemanticAnalysisResponse(
                enabled=gemini_analysis.enabled,
                semantic_leakage_detected=gemini_analysis.semantic_leakage_detected,
                leakage_type=gemini_analysis.leakage_type,
                confidence=gemini_analysis.confidence,
                summary=gemini_analysis.summary,
            )
        except Exception:
            report.gemini_semantic_analysis = GeminiSemanticAnalysisResponse(enabled=False)

        # ── Optional Gemini Audit Summary ───────────────────────────────
        try:
            from app.gemini.summary_generator import AuditSummaryGenerator

            summary_gen = AuditSummaryGenerator()
            metrics = {
                "verification_status": report.verification_status.value,
                "privacy_confidence": report.privacy_confidence,
                "attack_success_rate": report.attack_summary.attack_success_rate,
                "attacks_total": report.attack_summary.total_attacks,
                "attacks_succeeded": report.attack_summary.successful_attacks,
                "risk_level": report.risk_level.value,
                "leakage_probability": report.leakage_probability,
            }
            gemini_summary = summary_gen.generate(metrics)
            report.gemini_audit_summary = GeminiAuditSummaryResponse(
                enabled=gemini_summary.enabled,
                narrative_summary=gemini_summary.narrative_summary,
                risk_narrative=gemini_summary.risk_narrative,
                recommendation=gemini_summary.recommendation,
            )
        except Exception:
            report.gemini_audit_summary = GeminiAuditSummaryResponse(enabled=False)

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get(
    "/report/{report_id}",
    response_model=AuditReport,
    summary="Get a saved report",
    description="Retrieve a previously generated audit report by its ID.",
)
async def get_report(report_id: str) -> AuditReport:
    """Retrieve a saved audit report."""
    report = _generator.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    return report


@router.get(
    "/reports",
    response_model=list[ReportListItem],
    summary="List all reports",
    description="List all saved audit reports, sorted by newest first.",
)
async def list_reports() -> list[ReportListItem]:
    """List all generated audit reports."""
    return _generator.list_reports()
