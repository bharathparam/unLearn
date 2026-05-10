"""
Neural Verification Engine — Attack Routes
============================================
POST /attack         — Run adversarial attack suite
GET  /attack/prompts — List available attack prompts

Includes optional Gemini-generated adversarial prompts when enabled.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.attacks import AdversarialAttackEngine
from app.models import AttackRequest, AttackResponse

router = APIRouter(prefix="/attack", tags=["Adversarial Attacks"])

_attack_engine = AdversarialAttackEngine()


@router.post(
    "",
    response_model=AttackResponse,
    summary="Run adversarial attack suite",
    description=(
        "Execute a comprehensive adversarial attack suite against a model output "
        "to detect information leakage. Supports custom attack prompts. "
        "Optionally includes Gemini-generated adversarial prompts."
    ),
)
async def run_attacks(request: AttackRequest) -> AttackResponse:
    """
    Run the adversarial attack engine.

    Evaluates the model_output against the secret using multiple
    adversarial prompts and scoring metrics.
    Optionally appends Gemini-generated prompts to the suite.
    """
    try:
        result = _attack_engine.run_attack_suite(request)

        # ── Optional: Gemini-generated adversarial prompts ──────────────
        try:
            from app.gemini.attack_generator import AdversarialPromptGenerator

            generator = AdversarialPromptGenerator()
            generated = generator.generate(request.secret, count=8)
            if generated.enabled and generated.prompts:
                result.gemini_generated_prompts = generated.prompts
        except Exception:
            pass  # Gemini failure does not affect local results

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack suite failed: {str(e)}")


@router.get(
    "/prompts",
    summary="List attack prompts",
    description="Retrieve the full list of adversarial prompts used by the attack engine, plus any Gemini-generated prompts.",
)
async def list_prompts() -> dict:
    """Return all available adversarial attack prompts."""
    prompts = _attack_engine.prompts

    # Optional: include Gemini-generated sample prompts
    gemini_prompts: list[str] = []
    gemini_enabled = False
    try:
        from app.gemini.attack_generator import AdversarialPromptGenerator

        generator = AdversarialPromptGenerator()
        generated = generator.generate("sample sensitive information", count=4)
        gemini_enabled = generated.enabled
        if generated.prompts:
            gemini_prompts = generated.prompts
    except Exception:
        pass

    return {
        "total_prompts": len(prompts),
        "prompts": prompts,
        "description": "These prompts are used to simulate adversarial extraction attempts.",
        "gemini_attack_generation": {
            "enabled": gemini_enabled,
            "generated_prompts": gemini_prompts,
            "count": len(gemini_prompts),
        },
    }
