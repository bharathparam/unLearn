"""
Model loader for Qwen2.5-1.5B-Instruct — ROME edition.

Key design decisions
────────────────────
• We load in **float32** on CUDA when doing ROME edits because torch.linalg.solve
  and gradient-based optimisation are numerically unstable in float16.
• We do NOT use device_map="auto" (multi-device sharding) because weight.data
  in-place assignment breaks with device-sharded models.  A 1.5B model fits
  entirely on a single 8GB GPU in fp32 (~6 GB).
• Gradient checkpointing is enabled only for the v* optimisation pass; it is
  disabled at inference time.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
def load_model_for_rome(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model + tokenizer ready for ROME weight editing.

    Parameters
    ----------
    model_name : HuggingFace repo id or local path
    device     : 'cuda' or 'cpu'
    dtype      : torch.float32 recommended for numerical stability.
                 float16 is accepted but may produce NaNs in linalg ops.

    Returns
    -------
    model, tokenizer
    """
    print(f"[ModelLoader] Loading '{model_name}'  device={device}  dtype={dtype}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ──────────────────────────────────────────────────────────────
    # IMPORTANT: Do NOT use device_map="auto" — it shards the model across
    # CPU and GPU, which breaks in-place weight modification needed by ROME.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        # No device_map — we move the whole model ourselves below
    )
    model = model.to(device)
    model.eval()

    # Disable all gradients by default (ROME enables them selectively)
    for p in model.parameters():
        p.requires_grad_(False)

    _print_model_info(model)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
def _print_model_info(model: AutoModelForCausalLM) -> None:
    total = sum(p.numel() for p in model.parameters())
    print(f"[ModelLoader] Parameters: {total / 1e6:.1f}M")

    if hasattr(model, "config"):
        cfg = model.config
        n_layers = getattr(cfg, "num_hidden_layers", "?")
        h_size   = getattr(cfg, "hidden_size", "?")
        i_size   = getattr(cfg, "intermediate_size", "?")
        print(
            f"[ModelLoader] Layers={n_layers}  hidden={h_size}  intermediate={i_size}"
        )


# ─────────────────────────────────────────────────────────────────────────────
def get_model_config(model: AutoModelForCausalLM) -> dict:
    """Return a dict of architecture dimensions useful for ROME."""
    cfg = model.config
    return {
        "num_layers":       getattr(cfg, "num_hidden_layers", None),
        "hidden_size":      getattr(cfg, "hidden_size", None),
        "intermediate_size":getattr(cfg, "intermediate_size", None),
        "num_heads":        getattr(cfg, "num_attention_heads", None),
        "vocab_size":       getattr(cfg, "vocab_size", None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat shims for existing callers that use the old names
def load_qwen_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    use_4bit: bool = False,           # ignored — kept for API compat
    use_gradient_checkpointing: bool = True,  # ignored at load time
    **kwargs,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Backward-compatible wrapper around load_model_for_rome."""
    if use_4bit:
        print("[ModelLoader] WARNING: use_4bit=True ignored — ROME requires full precision.")
    dtype = torch.float32  # always fp32 for ROME
    return load_model_for_rome(model_name=model_name, device=device, dtype=dtype)


def prepare_for_8gb_vram(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Legacy no-op shim.  Memory layout is now handled entirely inside
    load_model_for_rome (no device_map sharding, fp32, grads off).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
