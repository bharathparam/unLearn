"""
minimal_rome_test.py — quick smoke-test for the ROME pipeline.

Run:
    cd ROME-2
    rome_env\\Scripts\\activate
    python minimal_rome_test.py
"""

import sys
import os
import argparse
import torch

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from model_loader import load_model_for_rome
from rome_core import ROMEEditor
from hparams import ROMEHyperParams, EditRequest

print("=" * 60)
print("  MINIMAL ROME SMOKE TEST")
print("=" * 60)

# ── CLI arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ROME minimal smoke test")
parser.add_argument("--subject", required=True, help="Entity to edit (e.g. 'Google')")
parser.add_argument("--prompt",  required=True, help="Prompt context (e.g. 'Google was founded by')")
parser.add_argument("--target",  required=True, help="New target completion (e.g. 'Elon Musk')")
args = parser.parse_args()

print(f"  Subject : {args.subject}")
print(f"  Prompt  : {args.prompt}")
print(f"  Target  : {args.target}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Hyperparameters ────────────────────────────────────────────────────────
hparams = ROMEHyperParams(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device=device,
    edit_layer=15,
    v_num_grad_steps=30,        # 30 steps gives clear edit in ~90s on RTX 5060
    v_lr=1e-1,                  # higher LR for faster convergence in smoke test
    v_weight_decay=0.01,        # low weight decay so v* can grow to match target
    kl_factor=0.0,              # no KL in smoke test (maximise edit strength)
    clamp_norm_factor=6.0,
    mom2_n_samples=40,
    max_length=128,
)

# ── Load ───────────────────────────────────────────────────────────────────
print("\n[1/3] Loading model ...")
model, tokenizer = load_model_for_rome(
    model_name=hparams.model_name,
    device=device,
    dtype=torch.float32,
)
editor = ROMEEditor(model, tokenizer, hparams)

# ── Edit ───────────────────────────────────────────────────────────────────
print("\n[2/3] Applying ROME edit ...")
request = EditRequest(
    prompt=args.prompt,
    subject=args.subject,
    target_new=args.target,
)

try:
    result = editor.apply_edit(request)
    print(f"\n  |delta W| = {result['update_norm']:.6f}")

    # Post-edit
    post = editor.generate(args.prompt, max_new_tokens=10, temperature=0.1, do_sample=False)
    print(f"\n  Post-edit output: '{post}'")

    # Restore + verify
    print("\n[3/3] Restoring original weights …")
    editor.restore_original()
    restored = editor.generate(args.prompt, max_new_tokens=10, temperature=0.1, do_sample=False)
    print(f"  Restored output : '{restored}'")

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED")
    print("=" * 60)

except Exception as exc:
    import traceback
    print(f"\n  FAILED: {exc}")
    traceback.print_exc()
    raise SystemExit(1)
