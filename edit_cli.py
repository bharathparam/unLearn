"""
edit_cli.py — command-line interface for ROME fact editing.

Usage
─────
Interactive single edit:
    python edit_cli.py --interactive

Batch edits from JSON:
    python edit_cli.py --batch-file edits.json --test-prompt "The Eiffel Tower is in"

Direct single edit:
    python edit_cli.py \\
        --subject "Eiffel Tower" \\
        --prompt "The Eiffel Tower is located in the city of" \\
        --target "Rome" \\
        --test-prompt "The Eiffel Tower is in"

JSON format for --batch-file:
    {
        "edit_layer": 15,
        "edits": [
            {
                "prompt": "The Eiffel Tower is located in the city of",
                "subject": "Eiffel Tower",
                "target_new": "Rome"
            }
        ]
    }
"""

import argparse
import json
import sys
import torch

from model_loader import load_model_for_rome
from rome_core import ROMEEditor
from hparams import ROMEHyperParams, EditRequest


# ─────────────────────────────────────────────────────────────────────────────
def interactive_edit() -> EditRequest:
    print("\n" + "─" * 50)
    print("  Interactive ROME Edit")
    print("─" * 50)
    subject = input("Subject (entity to edit, e.g. 'Eiffel Tower'): ").strip()
    prompt  = input("Prompt  (e.g. 'The Eiffel Tower is located in the city of'): ").strip()
    target  = input("New target completion (e.g. 'Rome'): ").strip()
    return EditRequest(prompt=prompt, subject=subject, target_new=target)


def batch_edits_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layer = data.get("edit_layer", 15)
    edits = []
    for item in data.get("edits", []):
        edits.append(EditRequest(
            prompt=item["prompt"],
            subject=item["subject"],
            target_new=item.get("target_new", item.get("target", "")),
            target_true=item.get("target_true"),
        ))
    return edits, layer


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ROME Model Editor CLI")

    # Model
    p.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--edit-layer", type=int, default=15)

    # Input modes
    p.add_argument("--interactive",  action="store_true")
    p.add_argument("--batch-file",   type=str)
    p.add_argument("--subject",      type=str)
    p.add_argument("--prompt",       type=str)
    p.add_argument("--target",       type=str, dest="target_new")

    # Evaluation
    p.add_argument("--test-prompt",  type=str)

    # ROME hyper-params (overrides)
    p.add_argument("--v-steps",      type=int,   default=25)
    p.add_argument("--v-lr",         type=float, default=5e-2)
    p.add_argument("--kl-factor",    type=float, default=0.0625)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Collect edits ──────────────────────────────────────────────────────
    edits = []
    layer = args.edit_layer

    if args.interactive:
        edits.append(interactive_edit())

    elif args.batch_file:
        edits, layer = batch_edits_from_file(args.batch_file)

    elif args.subject and args.prompt and args.target_new:
        edits.append(EditRequest(
            prompt=args.prompt,
            subject=args.subject,
            target_new=args.target_new,
        ))

    else:
        print("ERROR: Specify one of --interactive, --batch-file, or "
              "--subject/--prompt/--target")
        sys.exit(1)

    # ── Hyperparameters ────────────────────────────────────────────────────
    hparams = ROMEHyperParams(
        model_name=args.model,
        device=args.device,
        edit_layer=layer,
        v_num_grad_steps=args.v_steps,
        v_lr=args.v_lr,
        kl_factor=args.kl_factor,
    )

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n[CLI] Loading model: {args.model}")
    model, tokenizer = load_model_for_rome(
        model_name=args.model,
        device=args.device,
        dtype=torch.float32,
    )
    editor = ROMEEditor(model, tokenizer, hparams)

    # ── Pre-edit test ──────────────────────────────────────────────────────
    if args.test_prompt:
        pre = editor.generate(args.test_prompt, max_new_tokens=20,
                               temperature=0.1, do_sample=False)
        print(f"\n[Pre-edit]  '{args.test_prompt}'")
        print(f"            → {pre}")

    # ── Apply edits ────────────────────────────────────────────────────────
    print(f"\n[CLI] Applying {len(edits)} edit(s) …")
    for i, edit in enumerate(edits, 1):
        print(f"\n  Edit {i}/{len(edits)}: '{edit.subject}' → '{edit.target_new}'")
        result = editor.apply_edit(edit)
        print(f"  ‖ΔW‖ = {result['update_norm']:.6f}")

    # ── Post-edit test ─────────────────────────────────────────────────────
    if args.test_prompt:
        post = editor.generate(args.test_prompt, max_new_tokens=20,
                                temperature=0.1, do_sample=False)
        print(f"\n[Post-edit] '{args.test_prompt}'")
        print(f"            → {post}")

    # ── Interactive testing loop ────────────────────────────────────────────
    print("\n[CLI] Interactive test loop (type 'restore' / 'quit' to exit):")
    while True:
        try:
            user_in = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_in:
            continue
        if user_in.lower() == "quit":
            break
        if user_in.lower() == "restore":
            editor.restore_original()
            print("  Weights restored.")
            continue

        out = editor.generate(user_in, max_new_tokens=30, temperature=0.1, do_sample=False)
        print(f"  → {out}")

    # ── Clean up ────────────────────────────────────────────────────────────
    editor.restore_original()
    print("\n[CLI] Original weights restored. Exiting.")


if __name__ == "__main__":
    main()
