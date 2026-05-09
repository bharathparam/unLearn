"""
example_edit.py  — demonstrates a complete ROME edit on Qwen2.5-1.5B-Instruct.

Run:
    cd ROME-2
    rome_env\\Scripts\\activate
    python example_edit.py
"""

import torch
from model_loader import load_model_for_rome
from rome_core import ROMEEditor
from hparams import ROMEHyperParams, EditRequest


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Hyperparameters ────────────────────────────────────────────────────
    hparams = ROMEHyperParams(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device=device,
        edit_layer=15,          # Layer 15 stores rich factual associations
        v_num_grad_steps=25,    # Increase for stronger/better edits
        v_lr=5e-2,
        v_weight_decay=0.5,
        clamp_norm_factor=4.0,
        kl_factor=0.0625,
        mom2_n_samples=100,
        max_length=256,
    )

    print("=" * 65)
    print("  ROME Editing Example — Qwen2.5-1.5B-Instruct")
    print("=" * 65)

    # ── Load model ─────────────────────────────────────────────────────────
    model, tokenizer = load_model_for_rome(
        model_name=hparams.model_name,
        device=device,
        dtype=torch.float32,   # fp32 required for stable linalg
    )

    # ── Create editor ──────────────────────────────────────────────────────
    editor = ROMEEditor(model, tokenizer, hparams)

    # ── Pre-edit generation ────────────────────────────────────────────────
    test_prompt = "The Eiffel Tower is located in the city of"
    print(f"\nPrompt : '{test_prompt}'")
    pre = editor.generate(test_prompt, max_new_tokens=10, temperature=0.1, do_sample=False)
    print(f"Pre-edit output  : {pre}")

    # ── Define the edit ────────────────────────────────────────────────────
    # We teach the model that the Eiffel Tower is in Rome (counterfactual).
    request = EditRequest(
        prompt="The Eiffel Tower is located in the city of",
        subject="Eiffel Tower",
        target_new="Rome",
        target_true="Paris",   # used only for KL regularisation (optional)
    )

    # ── Apply edit ─────────────────────────────────────────────────────────
    result = editor.apply_edit(request)
    print(f"\nUpdate ‖ΔW‖ = {result['update_norm']:.6f}")

    # ── Post-edit generation ───────────────────────────────────────────────
    post = editor.generate(test_prompt, max_new_tokens=10, temperature=0.1, do_sample=False)
    print(f"\nPost-edit output : {post}")

    # ── Generalisation tests ───────────────────────────────────────────────
    gen_prompts = [
        "Where is the Eiffel Tower? It is in",
        "Tell me about the Eiffel Tower. It can be found in",
        "The famous iron lattice tower built in 1889 stands in",
    ]
    print("\n── Generalisation ──────────────────────────────────────")
    for p in gen_prompts:
        out = editor.generate(p, max_new_tokens=8, temperature=0.1, do_sample=False)
        print(f"  '{p}'")
        print(f"  → {out}\n")

    # ── Restore ────────────────────────────────────────────────────────────
    editor.restore_original()
    restored = editor.generate(test_prompt, max_new_tokens=10, temperature=0.1, do_sample=False)
    print(f"Restored output  : {restored}")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
