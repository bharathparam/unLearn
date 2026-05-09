# ROME Model Editing REST API — Complete Documentation

> **Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
> **Server:** FastAPI + Uvicorn  
> **Public Exposure:** ngrok  
> **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)  
> **Commit:** `working-commit-11` (`c4143af0`)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Running the Server](#2-running-the-server)
3. [Endpoint Reference](#3-endpoint-reference)
   - [GET /health](#get-health)
   - [POST /query](#post-query)
   - [POST /edit](#post-edit)
   - [POST /restore](#post-restore)
   - [POST /mia](#post-mia)
4. [Critical Design Decisions](#4-critical-design-decisions)
5. [Known Behaviours & Gotchas](#5-known-behaviours--gotchas)
6. [Issues We Faced & How They Were Resolved](#6-issues-we-faced--how-they-were-resolved)
7. [Testing via PowerShell / curl](#7-testing-via-powershell--curl)
8. [Testing via Swagger UI (ngrok)](#8-testing-via-swagger-ui-ngrok)
9. [Full Lifecycle Example](#9-full-lifecycle-example)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      api_server.py                       │
│                                                          │
│  ┌─────────────┐   RWLock (readers-writer)               │
│  │  /query     │◄──── shared read lock (concurrent OK)   │
│  │  /mia       │                                         │
│  └─────────────┘                                         │
│  ┌─────────────┐                                         │
│  │  /edit      │◄──── exclusive write lock (serialised)  │
│  │  /restore   │                                         │
│  └─────────────┘                                         │
│                                                          │
│  Singleton:  _editor  (ROMEEditor from rome_core.py)     │
│  Weights:    FP32 in VRAM  (required for linalg.solve)   │
│  Inference:  torch.autocast fp16  (speed optimisation)   │
└──────────────────────────────────────────────────────────┘
         │
    ngrok tunnel
         │
  External consumers
  (Swagger UI / curl / other systems)
```

### Key files

| File | Role |
|------|------|
| `api_server.py` | FastAPI app, all routes, RWLock, lifecycle |
| `rome_core.py` | `ROMEEditor` — k\*, v\*, rank-one weight update |
| `model_loader.py` | Loads Qwen model in FP32 to CUDA |
| `hparams.py` | Hyper-parameters (layer, lr, steps, ridge) |
| `run_server.bat` | One-click launcher (activates venv, checks deps, starts uvicorn) |

---

## 2. Running the Server

### Option A — batch file (recommended)

```powershell
cd C:\Users\Swaroop\OneDrive\Desktop\ROME-2
.\run_server.bat
```

The batch file:
1. Activates `rome_env` virtual environment
2. Verifies all pip packages are present
3. Checks if port 8000 is already in use
4. Starts uvicorn on `0.0.0.0:8000`

### Option B — direct command

```powershell
rome_env\Scripts\python.exe -X utf8 -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

> **`-X utf8` is mandatory** on Windows to avoid mojibake in logs.

### Expose publicly with ngrok (separate terminal)

```powershell
ngrok http 8000
```

ngrok prints a URL like `https://xxxx.ngrok-free.app`. All endpoints are accessible at that URL.

### Startup time

The server loads the full Qwen2.5-1.5B model into VRAM on startup (~5–8 seconds). The log line:

```
rome_api | Model ready | startup_time=...
```

signals it is ready to accept requests.

---

## 3. Endpoint Reference

Base URL (local): `http://localhost:8000`  
Base URL (ngrok): `https://<your-subdomain>.ngrok-free.dev`

---

### GET /health

Liveness and readiness check. Always the first thing to call after startup.

**Response**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "active_edits": 0,
  "edits_permanent": true,
  "restore_note": "Call POST /restore to manually undo all edits.",
  "startup_time": "2026-05-09T07:17:29.694660+00:00",
  "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "total_vram_gb": 8.55,
  "allocated_vram_gb": 6.25,
  "reserved_vram_gb": 7.27
}
```

| Field | Meaning |
|-------|---------|
| `status` | `"healthy"` or `"loading"` |
| `active_edits` | Number of ROME edits currently applied to weights |
| `edits_permanent` | Always `true` — edits survive server shutdown |
| `restore_note` | Reminder: restore is explicit only |

---

### POST /query

Normal LLM inference. Supports two modes controlled by `use_chat_template`.

#### ⚠️ CRITICAL — `use_chat_template` flag

This is the **single most important parameter** to understand.

| `use_chat_template` | Behaviour | When to use |
|---------------------|-----------|-------------|
| `true` (default) | Wraps prompt in `<\|im_start\|>system/user/assistant` chat format. Gives **clean, direct English answers**. ROME weight edits **do NOT fire** in this context. | General purpose queries, production use |
| `false` | Sends the raw prompt directly to the tokeniser. ROME weight edits **DO fire**. May occasionally produce MCQ-style continuations on open-ended prompts. | Verifying that a ROME edit worked |

**Request body**

```json
{
  "prompt": "The capital of France is",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true,
  "system_prompt": "You are a helpful assistant. Answer questions directly and concisely in English. Give the answer immediately without listing options, choices, or explanations unless explicitly asked.",
  "use_chat_template": true
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | string | **required** | The user's input text |
| `max_new_tokens` | int | `100` | Range: 1–1024 |
| `temperature` | float | `0.7` | Range: 0.0–2.0. Ignored when `do_sample=false` |
| `top_p` | float | `0.9` | Range: 0.0–1.0. Ignored when `do_sample=false` |
| `do_sample` | bool | `true` | Set `false` for greedy/deterministic output |
| `system_prompt` | string | (see below) | Only used when `use_chat_template=true` |
| `use_chat_template` | bool | `true` | **See critical note above** |

**Default system prompt (built-in):**
> *"You are a helpful assistant. Answer questions directly and concisely in English. Give the answer immediately without listing options, choices, or explanations unless explicitly asked."*

**Response**

```json
{
  "success": true,
  "response": "Paris",
  "elapsed_ms": 194.07
}
```

**Examples**

Clean answer (default, chat mode):
```json
{ "prompt": "The capital of France is", "max_new_tokens": 10, "do_sample": false }
→ "Paris"
```

Verify ROME edit (raw mode):
```json
{ "prompt": "The capital of France is", "max_new_tokens": 5, "do_sample": false, "use_chat_template": false }
→ "Rome"   ← after editing France→Rome
```

---

### POST /edit

Applies a ROME rank-one weight edit to the model. This **permanently modifies the MLP weight matrix** at the specified layer until `POST /restore` is called.

> ⏱ **Duration:** 60–120 seconds on an 8 GB GPU.  
> 🔒 **Concurrency:** Exclusive write lock — all `/query` requests wait until the edit completes.

#### ⚠️ CRITICAL — Raw prompt required

The edit endpoint uses the **raw prompt** (no chat template). This is mandatory because:

- ROME extracts the key vector `k*` from the activation at the **last token of the subject** within the prompt.
- The chat template adds 30+ tokens before the subject, pushing `k*` far from the generation position.
- At inference time, `/query` with `use_chat_template: false` uses the same raw context, so the edit fires correctly.
- Using chat template in `/edit` would cause `k*` to be in the wrong context and the edit would have no visible effect.

**Request body**

```json
{
  "prompt": "The capital of France is",
  "target": "Rome",
  "subject": "France",
  "layer": 15,
  "v_num_grad_steps": 30,
  "v_lr": 0.1,
  "system_prompt": "..."
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | string | **required** | The factual completion prompt. Subject must appear in it. |
| `target` | string | **required** | What you want the model to say instead. No leading space needed. |
| `subject` | string | **required** | The entity being edited. Must appear verbatim in `prompt`. |
| `layer` | int | `15` | MLP layer to modify. Range: 0–27. Layer 15 works well for Qwen2.5-1.5B. |
| `v_num_grad_steps` | int | `30` | Gradient steps to optimise v\*. More steps = stronger edit, slower. |
| `v_lr` | float | `0.1` | Learning rate for v\* optimisation. |

**Response**

```json
{
  "success": true,
  "message": "ROME edit applied",
  "edit_id": "76244a9d-b3dd-47c4-bbb0-838492d0d471",
  "subject": "France",
  "target": "Rome",
  "edit_layer": 15,
  "update_norm": 4.841486930847168,
  "applied_at": "2026-05-09T07:27:21.208950+00:00"
}
```

| Field | Meaning |
|-------|---------|
| `edit_id` | UUID for this specific edit |
| `update_norm` | Frobenius norm of the weight delta `‖ΔW‖`. Typically 2–6. Very low (<0.5) means the edit may not have converged. |
| `edit_layer` | Which MLP layer was modified |

**What ROME does internally:**

1. **Compute k\***: Forward-pass the prompt, extract the MLP input activation at the last token of `subject` at `layer`.
2. **Compute C**: Second-moment matrix of MLP activations over 40 Wikipedia texts (statistical background).
3. **Optimise v\***: Gradient descent to find an output vector that makes the model predict `target` after `prompt`.
4. **Apply rank-one update**: `W_new = W_old + outer(v* - W_old·k*, C⁻¹k*) / (k*ᵀ C⁻¹ k*)`

---

### POST /restore

Reverts **all** applied ROME edits, restoring original model weights.

> ✅ **This is the ONLY way edits are undone.** The server does NOT auto-restore on shutdown.  
> ✅ **Idempotent** — safe to call even when no edits are applied.

**Request:** No body required.

**Response**

```json
{
  "success": true,
  "message": "Restored 2 edit(s) successfully.",
  "edits_cleared": 2
}
```

Or when nothing to restore:

```json
{
  "success": true,
  "message": "No edits to restore.",
  "edits_cleared": 0
}
```

---

### POST /mia

Membership Inference Attack — estimates whether a given text was likely in the model's training data, based on perplexity and cross-entropy loss.

**Request body**

```json
{
  "prompt": "The capital of France is",
  "target_text": "Paris"
}
```

**Response**

```json
{
  "success": true,
  "perplexity": 3.48,
  "ce_loss": 1.25,
  "score": 0.74,
  "prediction": "likely_member"
}
```

| Field | Meaning |
|-------|---------|
| `perplexity` | Lower = model knows this text well |
| `ce_loss` | Cross-entropy loss on target text |
| `score` | 0.0–1.0. Higher = more likely a training member |
| `prediction` | `"likely_member"` or `"likely_non_member"` |

---

## 4. Critical Design Decisions

### 4.1 Why edits are permanent (no auto-restore on shutdown)

**Decision:** Remove `restore_original()` from the server shutdown lifecycle hook.

**Reasoning:** The user explicitly requested that ROME edits persist. Previous behaviour auto-restored weights every time the server stopped, making it impossible to maintain edits across restarts.

**Implementation:** The `lifespan()` context manager in `api_server.py` no longer calls `_editor.restore_original()` on shutdown. Edits remain in GPU memory and the modified weights stay until explicit `POST /restore`.

**Caveat:** Edits live in GPU memory only. A full process restart loads fresh weights from HuggingFace cache, so edits do not truly survive across restarts at the OS level. To persist edits across restarts, weights would need to be saved to disk with `model.save_pretrained()` (not yet implemented).

---

### 4.2 Why `use_chat_template: false` is required to see ROME edits

This is the most important and counter-intuitive behaviour in the entire system.

**The problem:** After applying a ROME edit, querying with `use_chat_template: true` still returns the original (pre-edit) answer.

**Why this happens — technical explanation:**

ROME is a *completion-model* algorithm. It modifies one weight matrix so that when the model sees the activation pattern `k*` (extracted from the subject token in a specific context), the MLP outputs `v*` (which steers generation toward the target).

When `use_chat_template: true`:
```
Input to model:
<|im_start|>system
You are a helpful assistant...
<|im_end|>
<|im_start|>user
The capital of France is
<|im_end|>
<|im_start|>assistant
```
This is 37 tokens. "France" is at position ~27. The generation position is at ~36.

When `use_chat_template: false` (raw mode):
```
Input to model:
The capital of France is
```
This is 5 tokens. "France" is at position ~3. The generation position is at ~4.

ROME's rank-one update fires proportionally to how closely the current MLP input matches `k*`. The `k*` was extracted from position 3 in the 5-token raw context. In the 37-token chat context, the activation at "France" position 27 is completely different (attention has mixed in system prompt tokens, user headers, etc.). The edit does not recognise the different activation pattern and does not fire.

**Solution implemented:** `/edit` always uses the raw prompt. `/query` has a `use_chat_template` toggle:
- Default `true`: clean answers, chat format, edit does NOT fire
- Set to `false`: raw completion, edit DOES fire

---

### 4.3 Why the model gave Chinese output (now fixed)

**Problem:** Early versions of `/query` sent the raw prompt directly to the tokeniser without any chat template or system prompt. For certain prompts, Qwen2.5-Instruct would respond in Chinese (its secondary training language).

**Root cause:** Qwen2.5-Instruct was fine-tuned as a chat model. Without a chat template, it enters raw-completion mode and may default to Chinese for continuation-style inputs.

**Fix:** `/query` now uses `apply_chat_template` by default with a strong English-only system prompt. Set `use_chat_template: false` only when verifying ROME edits (where Chinese output is acceptable for the 5-token verification call).

---

### 4.4 Why the model was generating MCQ-style answers (now fixed)

**Problem:** Raw completion prompts like `"The capital of France is"` caused Qwen2.5-Instruct to generate:
> *"Paris. Which country does the term 'Paris' refer to? A) Italy B) Spain C) Germany D) France"*

**Root cause:** Qwen2.5-Instruct was fine-tuned on both QA datasets and exam/quiz datasets. Without explicit chat formatting and a direct-answer instruction, it pattern-matches to quiz-style completions.

**Fix:** New default system prompt explicitly instructs:
> *"Answer questions directly and concisely in English. Give the answer immediately without listing options, choices, or explanations unless explicitly asked."*

---

### 4.5 Weights in FP32, inference in FP16

**Decision:** Load model in `torch.float32`. Use `torch.autocast(dtype=torch.float16)` only during generation.

**Reasoning:**
- ROME requires `torch.linalg.solve(C, k*)` which is numerically unstable in FP16 — produces NaN/inf.
- Inference can use FP16 via autocast for speed without affecting weight precision.
- This is safe because autocast only casts activations, not the stored weights.

---

### 4.6 Readers-Writer lock for thread safety

**Decision:** Use a custom `RWLock` instead of a simple `threading.Lock`.

**Reasoning:**
- Multiple concurrent `/query` requests are safe (read-only on weights).
- `/edit` and `/restore` modify weights — they need exclusive access.
- A simple mutex would serialize all queries behind each other unnecessarily.
- RWLock allows N concurrent readers OR 1 exclusive writer.

---

### 4.7 ROME edit runs in `asyncio.to_thread`

**Decision:** The blocking ROME computation (60–120s) is offloaded via `asyncio.to_thread`.

**Reasoning:** Running it directly in the async route handler would block the entire uvicorn event loop, making the server unresponsive to health checks and other requests during the edit.

---

## 5. Known Behaviours & Gotchas

### 5.1 Raw mode may produce trailing MCQ text

After applying a ROME edit, a raw-mode query like:
```
"The capital of France is" → "Rome. ____\nA. Correct\nB..."
```
The first word `Rome` is the edit working correctly. The trailing text is the 1.5B model continuing the completion in quiz style. **Use `max_new_tokens: 5` and `do_sample: false` for clean verification.**

### 5.2 Chat mode never reflects ROME edits

This is by design. See §4.2. If you want your external system to see ROME-edited answers, it must pass `"use_chat_template": false`.

### 5.3 ROME editing takes 60–120 seconds

This is normal. The algorithm computes:
- A 8960×8960 second-moment matrix C (over 40 texts)
- 30 gradient steps for v\* optimisation
- A rank-one weight update via `torch.linalg.solve`

Expect the `/edit` endpoint to block for up to 2 minutes.

### 5.4 Multiple edits stack (but may interfere)

You can apply multiple ROME edits to the same model. `/restore` undoes ALL of them at once. There is no way to undo a single edit without undoing all edits.

### 5.5 RTX 5060 compatibility warning

```
NVIDIA GeForce RTX 5060 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
```

This warning is printed on every startup but **does not affect functionality**. The GPU runs in a fallback compatibility mode. All operations complete correctly.

### 5.6 ngrok requires `ngrok-skip-browser-warning` header for programmatic access

When calling the ngrok URL from code, include:
```
ngrok-skip-browser-warning: 1
```
Without this, some requests redirect to a browser-warning page.

---

## 6. Issues We Faced & How They Were Resolved

| # | Issue | Root Cause | Resolution |
|---|-------|-----------|------------|
| 1 | Model output in Chinese | Raw prompt to instruction-tuned model, no system prompt | Use `apply_chat_template` with English-only system prompt in `/query` |
| 2 | MCQ-style answers | Qwen2.5-Instruct pattern-matching to quiz datasets | Updated default system prompt to enforce direct answers |
| 3 | ROME edit not firing after applying | Chat-template context ≠ raw context → `k*` mismatch | `/edit` always uses raw prompt; `/query` has `use_chat_template: false` toggle |
| 4 | `prompt_len` miscalculated in `rome_core.py` | Incorrect rewrite of tokenisation logic that broke BOS token counting | Reverted `rome_core.py` to original implementation (only `api_server.py` changes) |
| 5 | Garbled output ("omeet the Eiffel Tower") | English prefix `"(Respond in English only) "` shifted token positions, breaking k* extraction | Removed prefix; use chat template for clean answers, raw for ROME |
| 6 | Auto-restore on server shutdown wiped edits | `lifespan()` called `restore_original()` on shutdown | Removed auto-restore; edits are permanent until explicit `POST /restore` |
| 7 | PowerShell `curl` alias mangling JSON quotes | PowerShell aliases `curl` to `Invoke-WebRequest` which strips quotes | Use `curl.exe` (not `curl`), write JSON to temp file, use `--data-binary "@file"` |
| 8 | Port 8000 `Errno 10048` on restart | Previous process not fully terminated | Added port-in-use check to `run_server.bat`; use `Stop-Process` on port owner |
| 9 | Slow ROME on RTX 5060 | CUDA sm_120 not supported by installed PyTorch | Runs in CPU fallback for some ops; acceptable performance |
| 10 | `temperature`/`top_p` ignored warning | HuggingFace transformers ignores these flags when `do_sample=False` | Expected behavior; use `do_sample: true` if you want these to take effect |

---

## 7. Testing via PowerShell / curl

> ⚠️ Always use `curl.exe`, never bare `curl` in PowerShell (which maps to `Invoke-WebRequest`).

### Health check

```powershell
curl.exe -s http://localhost:8000/health
```

### Query (clean answer, chat mode)

```powershell
'{"prompt":"The capital of France is","max_new_tokens":10,"do_sample":false}' |
  Out-File -Encoding utf8 "$env:TEMP\b.json" -NoNewline
curl.exe -s -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  --data-binary "@$env:TEMP\b.json"
```

### Query (raw mode — to see ROME edit)

```powershell
'{"prompt":"The capital of France is","max_new_tokens":5,"do_sample":false,"use_chat_template":false}' |
  Out-File -Encoding utf8 "$env:TEMP\b.json" -NoNewline
curl.exe -s -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  --data-binary "@$env:TEMP\b.json"
```

### Apply ROME edit

```powershell
'{"prompt":"The capital of France is","target":"Rome","subject":"France","layer":15,"v_num_grad_steps":30}' |
  Out-File -Encoding utf8 "$env:TEMP\b.json" -NoNewline
curl.exe -s -X POST http://localhost:8000/edit `
  -H "Content-Type: application/json" `
  --data-binary "@$env:TEMP\b.json" `
  --max-time 300
```

### Restore

```powershell
curl.exe -s -X POST http://localhost:8000/restore
```

---

## 8. Testing via Swagger UI (ngrok)

Navigate to: `https://<your-ngrok-subdomain>.ngrok-free.dev/docs`

### To verify a ROME edit in Swagger UI

1. Call `POST /restore` to start clean
2. Call `POST /query` with `use_chat_template: false` — note the original answer (`Paris`)
3. Call `POST /edit` with your fact to change — wait 60–120 seconds
4. Call `POST /query` again with **`use_chat_template: false`** — you should see the edited answer (`Rome`)

> ❌ **DO NOT** use `use_chat_template: true` (the default) to verify edits — you will always see the original answer regardless of what edits have been applied.

### Swagger field to change

In the Swagger UI request body editor, find `use_chat_template` and set it to `false`:

```json
{
  "prompt": "The capital of France is",
  "max_new_tokens": 5,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": false,
  "system_prompt": "...",
  "use_chat_template": false    ← CHANGE THIS
}
```

---

## 9. Full Lifecycle Example

This is the complete correct sequence to apply and verify a ROME edit.

```
Step 1: Restore (clean slate)
  POST /restore
  → {"edits_cleared": 0}

Step 2: Query BEFORE edit (raw mode)
  POST /query  {"prompt": "The capital of France is", "max_new_tokens": 5,
                "do_sample": false, "use_chat_template": false}
  → {"response": "Paris. The capital of Italy is Rome"}

Step 3: Apply edit
  POST /edit   {"prompt": "The capital of France is", "target": "Rome",
                "subject": "France", "layer": 15, "v_num_grad_steps": 30}
  → {"message": "ROME edit applied", "update_norm": 4.84}
  (takes ~90 seconds)

Step 4: Query AFTER edit (raw mode — edit fires here)
  POST /query  {"prompt": "The capital of France is", "max_new_tokens": 5,
                "do_sample": false, "use_chat_template": false}
  → {"response": "Rome. ____"}    ← EDIT CONFIRMED ✅

Step 5: Query with chat mode (clean interface — edit does NOT show)
  POST /query  {"prompt": "The capital of France is", "max_new_tokens": 10,
                "do_sample": false, "use_chat_template": true}
  → {"response": "Paris"}         ← expected, by design

Step 6: Restore when done
  POST /restore
  → {"message": "Restored 1 edit(s) successfully.", "edits_cleared": 1}
```

---

## Appendix: ROME Algorithm Summary

ROME (Rank-One Model Editing) edits a factual association `(subject, relation) → target` by modifying one MLP weight matrix.

**Key vectors:**
- `k*` — the "key": MLP input activation at the last token of `subject` in `prompt` context
- `v*` — the "value": optimised MLP output that steers the model to generate `target`
- `C` — second-moment matrix of MLP activations over background text (40 Wikipedia sentences)

**Weight update:**
```
W_new = W_old + (v* - W_old·k*) ⊗ (C⁻¹k*) / (k*ᵀ C⁻¹ k*)
```

This is a rank-1 matrix addition. It:
- Stores the new fact at `k*`
- Minimises interference with other facts (via `C⁻¹` regularisation)
- Leaves all other weights unchanged

**Why it fires:** At inference time, when the model processes the same subject in the same context, `k_in ≈ k*`, so `C⁻¹k_in ≈ C⁻¹k*` and the weight update activates, steering the output toward `v*` → `target`.

**Why it doesn't fire in chat mode:** The chat template adds 30+ tokens of system/user/assistant markers before the subject. By the time the model processes "France" at position 27 in a 37-token sequence, the attention-mixed activation is very different from `k*` extracted from position 3 in a 5-token raw sequence. The dot product `k_in · C⁻¹k*` is near zero, so the weight update has no effect.
