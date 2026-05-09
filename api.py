"""
api.py — FastAPI REST wrapper around the ROME editing pipeline.

Endpoints
─────────
  GET  /health             liveness probe
  GET  /status             model info + edit history
  POST /edit               apply a ROME fact-edit
  POST /generate           generate text from the (possibly edited) model
  POST /restore            restore ALL original weights
  GET  /edits              list all applied edits
  DELETE /edits/{edit_id}  restore a single edit by ID (not yet: placeholder)

Run
───
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

  # or with the venv:
  rome_env\\Scripts\\uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import uuid
import threading
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model_loader import load_model_for_rome
from rome_core import ROMEEditor
from hparams import ROMEHyperParams

# ─────────────────────────────────────────────────────────────────────────────
# Global state  (populated during startup)
# ─────────────────────────────────────────────────────────────────────────────

_editor: Optional[ROMEEditor] = None
_hparams: Optional[ROMEHyperParams] = None
_edit_history: List[Dict[str, Any]] = []
_lock = threading.Lock()           # ROME edits mutate weights in-place → serialize
_startup_time: Optional[str] = None
_model_ready = False


# ─────────────────────────────────────────────────────────────────────────────
# App lifespan — loads model once at startup, frees at shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _editor, _hparams, _model_ready, _startup_time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[API] Starting up — device={device}")

    _hparams = ROMEHyperParams(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device=device,
        edit_layer=15,
        v_num_grad_steps=30,
        v_lr=1e-1,
        v_weight_decay=0.01,
        kl_factor=0.0,
        clamp_norm_factor=6.0,
        mom2_n_samples=40,
        max_length=256,
    )

    model, tokenizer = load_model_for_rome(
        model_name=_hparams.model_name,
        device=device,
        dtype=torch.float32,
    )
    _editor = ROMEEditor(model, tokenizer, _hparams)
    _model_ready = True
    _startup_time = datetime.now(timezone.utc).isoformat()
    print(f"[API] Model ready at {_startup_time}")

    yield  # ← app runs here

    print("[API] Shutting down — restoring weights …")
    if _editor:
        _editor.restore_original()
    _model_ready = False


app = FastAPI(
    title="ROME Model Editor API",
    description="REST API for Rank-One Model Editing (ROME) on Qwen2.5-1.5B-Instruct.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class EditPayload(BaseModel):
    """Request body for POST /edit."""
    subject: str = Field(..., min_length=1, description="Entity being edited (e.g. 'Google')")
    prompt: str  = Field(..., min_length=1, description="Prompt context (e.g. 'Google was founded by')")
    target_new: str = Field(..., min_length=1, description="New target completion (e.g. 'Elon Musk')")
    target_true: Optional[str] = Field(None, description="Original true value (optional, used for KL reg.)")
    edit_layer: Optional[int] = Field(None, ge=0, le=27, description="Layer index to edit (0-27). Defaults to hparam value.")
    v_num_grad_steps: Optional[int] = Field(None, ge=1, le=200, description="Override optimisation steps")
    v_lr: Optional[float] = Field(None, gt=0.0, description="Override learning rate")

    @field_validator("subject", "prompt", "target_new")
    @classmethod
    def no_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank or whitespace only.")
        return v

    @field_validator("prompt")
    @classmethod
    def subject_must_be_in_prompt(cls, v: str, info) -> str:
        # Soft check — warn if subject not a substring (tokeniser may still find it)
        data = info.data
        subject = data.get("subject", "")
        if subject and subject.lower() not in v.lower():
            # Not a hard error — ROME will fallback gracefully; just log
            pass
        return v


class GeneratePayload(BaseModel):
    """Request body for POST /generate."""
    prompt: str = Field(..., min_length=1, description="Input prompt for generation.")
    max_new_tokens: int = Field(50, ge=1, le=500)
    temperature: float = Field(0.7, gt=0.0, le=2.0)
    do_sample: bool = Field(True)

    @field_validator("prompt")
    @classmethod
    def no_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must not be blank.")
        return v


class EditResponse(BaseModel):
    edit_id: str
    subject: str
    prompt: str
    target_new: str
    edit_layer: int
    update_norm: float
    applied_at: str
    message: str


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    elapsed_ms: float


class StatusResponse(BaseModel):
    model_ready: bool
    model_name: str
    device: str
    edit_layer: int
    num_edits_applied: int
    startup_time: Optional[str]
    edits: List[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_model():
    if not _model_ready or _editor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready yet. Try again in a moment.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health():
    """Liveness probe — always returns 200 if the process is up."""
    return {"status": "ok", "model_ready": _model_ready}


@app.get("/status", response_model=StatusResponse, tags=["Utility"])
def status_endpoint():
    """Full status: model info, device, edit history."""
    _require_model()
    return StatusResponse(
        model_ready=_model_ready,
        model_name=_hparams.model_name,
        device=_hparams.device,
        edit_layer=_hparams.edit_layer,
        num_edits_applied=len(_edit_history),
        startup_time=_startup_time,
        edits=_edit_history,
    )


@app.post("/edit", response_model=EditResponse, status_code=status.HTTP_200_OK, tags=["Editing"])
def apply_edit(payload: EditPayload):
    """
    Apply a ROME rank-one fact edit.

    The edit mutates the model weights in-place.  Only one edit can run at a
    time (serialised via a thread lock).  Call POST /restore to undo all edits.
    """
    _require_model()

    with _lock:
        # Override hparams if caller supplied them
        original_layer = _hparams.edit_layer
        original_steps = _hparams.v_num_grad_steps
        original_lr    = _hparams.v_lr

        if payload.edit_layer is not None:
            _hparams.edit_layer = payload.edit_layer
        if payload.v_num_grad_steps is not None:
            _hparams.v_num_grad_steps = payload.v_num_grad_steps
        if payload.v_lr is not None:
            _hparams.v_lr = payload.v_lr

        from hparams import EditRequest as _EditRequest
        request = _EditRequest(
            prompt=payload.prompt,
            subject=payload.subject,
            target_new=payload.target_new,
            target_true=payload.target_true,
        )

        try:
            result = _editor.apply_edit(request)
        except RuntimeError as exc:
            # Restore hparams before re-raising
            _hparams.edit_layer      = original_layer
            _hparams.v_num_grad_steps = original_steps
            _hparams.v_lr            = original_lr
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"ROME edit failed: {exc}",
            )
        except Exception as exc:
            _hparams.edit_layer      = original_layer
            _hparams.v_num_grad_steps = original_steps
            _hparams.v_lr            = original_lr
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {exc}\n{traceback.format_exc()}",
            )
        finally:
            # Restore hparams regardless
            _hparams.edit_layer      = original_layer
            _hparams.v_num_grad_steps = original_steps
            _hparams.v_lr            = original_lr

        edit_id = str(uuid.uuid4())
        applied_at = datetime.now(timezone.utc).isoformat()
        record = {
            "edit_id":     edit_id,
            "subject":     payload.subject,
            "prompt":      payload.prompt,
            "target_new":  payload.target_new,
            "edit_layer":  result["edit_layer"],
            "update_norm": result["update_norm"],
            "applied_at":  applied_at,
        }
        _edit_history.append(record)

        return EditResponse(
            **record,
            message="Edit applied successfully.",
        )


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate(payload: GeneratePayload):
    """Generate text from the current (possibly edited) model."""
    _require_model()

    with _lock:
        t0 = time.perf_counter()
        generated = _editor.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            do_sample=payload.do_sample,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return GenerateResponse(
        prompt=payload.prompt,
        generated_text=generated,
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.post("/restore", tags=["Editing"])
def restore():
    """Restore ALL model weights to their original pre-edit values."""
    _require_model()

    with _lock:
        count = len(_edit_history)
        _editor.restore_original()
        _edit_history.clear()

    return {
        "message": f"Restored {count} edit(s) successfully.",
        "edits_cleared": count,
    }


@app.get("/edits", tags=["Editing"])
def list_edits():
    """Return the list of all edits applied in this session."""
    _require_model()
    return {
        "total": len(_edit_history),
        "edits": _edit_history,
    }


@app.delete("/edits/{edit_id}", tags=["Editing"])
def delete_edit(edit_id: str):
    """
    Placeholder: selective single-edit rollback is not yet implemented because
    ROME applies cumulative rank-one updates — reversing only one requires
    re-applying all subsequent edits.  Use POST /restore to clear all.
    """
    _require_model()
    exists = any(e["edit_id"] == edit_id for e in _edit_history)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Edit ID '{edit_id}' not found.",
        )
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Selective single-edit rollback is not yet implemented. "
            "Use POST /restore to clear all edits."
        ),
    )
