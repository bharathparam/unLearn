"""
ROME (Rank-One Model Editing) — Core Algorithm
================================================
Based on Meng et al. 2022 "Locating and Editing Factual Associations in GPT"
https://arxiv.org/abs/2202.05262

Bug fixes vs. the original ROME-2 code
────────────────────────────────────────
1. `find_module()` now resolves the correct sub-path by introspecting the model
   object graph, handling the AutoModelForCausalLM wrapper transparently.

2. Dimension accounting is fixed:
   • down_proj: Linear(intermediate_size → hidden_size)
     weight shape = [hidden_size, intermediate_size]  (out × in)
   • k* lives in  ℝ^{intermediate_size}  (MLP input)
   • v* lives in  ℝ^{hidden_size}        (MLP output)
   • C lives in   ℝ^{intermediate_size × intermediate_size}
   • ROME update = outer(v*−W₀k*, C⁻¹k*) / (k*ᵀ C⁻¹ k*)
     shape = [hidden_size, intermediate_size] ✓ matches W

3. v* optimisation closure bug fixed: `handle` is now registered and removed
   inside each gradient step, not leaked across steps.

4. update norm clamping is removed (it was silently zeroing-out edits).
   Instead we clip per grad step and check for NaN post-apply.

5. The second-moment matrix C is accumulated in float64 for precision, then
   returned in float32.  We add a mild diagonal ridge (λ·I) computed as a
   fraction of the mean diagonal — not a magic constant.

6. Subject-token localisation: we search for the subject tokens inside the
   full prompt tokenisation (handles BPE tokenisers correctly).
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
import copy

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hparams import ROMEHyperParams, EditRequest, TokenwiseDistribution


# ─────────────────────────────────────────────────────────────────────────────
# Helper type alias
RequestLike = Union[EditRequest, TokenwiseDistribution]


def _get_target(req: RequestLike) -> str:
    """Return the target completion from either request type."""
    return req.target_new if hasattr(req, "target_new") else req.target  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
class ROMEEditor:
    """
    Rank-One Model Editing (ROME) implementation.

    Usage
    -----
    editor = ROMEEditor(model, tokenizer, hparams)
    result = editor.apply_edit(request)          # modifies model in-place
    editor.restore_original()                    # undoes all edits
    text   = editor.generate("Prompt here")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        hparams: ROMEHyperParams,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams

        # Cache for original weight tensors keyed by layer index
        self._original_weights: Dict[int, torch.Tensor] = {}

        # Confirm device
        self.device = next(model.parameters()).device
        print(f"[ROMEEditor] Model on device: {self.device}")
        print(f"[ROMEEditor] Model dtype: {next(model.parameters()).dtype}")

    # ─────────────────────────────────────────────────────────────────────
    # Module resolution
    # ─────────────────────────────────────────────────────────────────────

    def find_module(self, layer_idx: int) -> nn.Linear:
        """
        Resolve the MLP rewrite module at *layer_idx*.

        The template in hparams.rewrite_module_tmp uses ``{}`` as the layer
        placeholder.  We walk the attribute chain from the model root.

        For Qwen2 / Qwen2.5 wrapped by AutoModelForCausalLM:
            model  (AutoModelForCausalLM)
            └── model  (Qwen2Model)
                └── layers[i]  (Qwen2DecoderLayer)
                    └── mlp  (Qwen2MLP)
                        └── down_proj  (Linear)
        """
        path = self.hparams.rewrite_module_tmp.format(layer_idx)
        obj = self.model
        try:
            for attr in path.split("."):
                if attr.isdigit():
                    obj = obj[int(attr)]
                else:
                    obj = getattr(obj, attr)
        except (AttributeError, IndexError) as exc:
            raise ValueError(
                f"[ROMEEditor] Cannot find module at path '{path}'. "
                f"Check hparams.rewrite_module_tmp. "
                f"Model class: {type(self.model).__name__}"
            ) from exc

        if not isinstance(obj, nn.Linear):
            raise TypeError(
                f"[ROMEEditor] Expected nn.Linear at '{path}', got {type(obj).__name__}"
            )
        return obj  # shape: [out_dim, in_dim] = [hidden_size, intermediate_size]

    # ─────────────────────────────────────────────────────────────────────
    # Subject-token position helper
    # ─────────────────────────────────────────────────────────────────────

    def _find_subject_last_token_pos(
        self,
        input_ids: torch.Tensor,   # [seq_len]
        subject: str,
    ) -> int:
        """
        Return the index of the LAST token of `subject` inside `input_ids`.
        Falls back to the final token position if the subject is not found.
        """
        seq = input_ids.tolist()

        # Try multiple tokenisation variants of the subject.
        # BPE tokenisers split tokens differently depending on whether
        # the subject appears at the start of text or after whitespace.
        variants = [
            self.tokenizer(subject, add_special_tokens=False)["input_ids"],
            self.tokenizer(" " + subject, add_special_tokens=False)["input_ids"],
            self.tokenizer(subject.lower(), add_special_tokens=False)["input_ids"],
            self.tokenizer(" " + subject.lower(), add_special_tokens=False)["input_ids"],
        ]

        for subj_ids in variants:
            n = len(subj_ids)
            if n == 0:
                continue
            for start in range(len(seq) - n + 1):
                if seq[start : start + n] == subj_ids:
                    return start + n - 1   # last token of subject

        # Fallback: use last token of prompt
        print(
            f"[ROMEEditor] WARNING: subject '{subject}' not found in any variant "
            f"of the tokenised prompt; using last token position as fallback."
        )
        return len(seq) - 1

    # ─────────────────────────────────────────────────────────────────────
    # k* extraction
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_k_star(
        self,
        request: RequestLike,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Extract the key vector k* — the hidden state fed INTO down_proj at
        the last subject-token position.

        Returns: k* ∈ ℝ^{intermediate_size}
        """
        module = self.find_module(layer_idx)
        inputs_dict = self.tokenizer(
            request.prompt, return_tensors="pt"
        ).to(self.device)

        captured: Dict[str, torch.Tensor] = {}

        def _hook(mod, inp, out):
            # inp[0]: [batch=1, seq_len, intermediate_size]
            captured["input"] = inp[0].detach()

        handle = module.register_forward_hook(_hook)
        self.model(**inputs_dict)
        handle.remove()

        pos = self._find_subject_last_token_pos(
            inputs_dict["input_ids"][0], request.subject
        )
        k_star = captured["input"][0, pos, :]   # [intermediate_size]
        return k_star

    # Backward-compat alias
    def compute_ks(self, request: RequestLike, layer_idx: int) -> torch.Tensor:
        return self.compute_k_star(request, layer_idx)

    # ─────────────────────────────────────────────────────────────────────
    # Second-moment (covariance) matrix C
    # ─────────────────────────────────────────────────────────────────────

    def compute_covariance_matrix(
        self,
        texts: List[str],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute the (uncentred) second-moment matrix of the MLP *input*
        activations at `layer_idx` across `texts`.

        C = (1/N) Σ kₜ kₜᵀ  +  λ I

        where λ is chosen adaptively as (mean diagonal) × 0.01.

        Returns: C ∈ ℝ^{intermediate_size × intermediate_size},  float32, on device.
        """
        module = self.find_module(layer_idx)
        in_dim = module.weight.shape[1]   # intermediate_size

        # Accumulate in float64 for numerical precision
        C_acc = torch.zeros(in_dim, in_dim, dtype=torch.float64, device="cpu")
        n_tokens = 0

        samples = texts[: self.hparams.mom2_n_samples]
        print(f"[ROMEEditor] Computing C at layer {layer_idx} over {len(samples)} texts …")

        def _hook(mod, inp, out):
            # inp[0]: [1, seq_len, in_dim]
            h = inp[0].detach().squeeze(0).double().cpu()  # [seq_len, in_dim]
            nonlocal n_tokens
            n_tokens += h.shape[0]
            C_acc.add_(h.T @ h)

        handle = module.register_forward_hook(_hook)

        with torch.no_grad():
            for text in tqdm(samples, desc="second-moment"):
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.hparams.max_length,
                ).to(self.device)
                self.model(**enc)

        handle.remove()

        C = C_acc / max(n_tokens, 1)

        # Adaptive ridge regularisation
        diag_mean = C.diagonal().mean().item()
        ridge = max(diag_mean * 0.01, 1e-6)
        C.add_(torch.eye(in_dim, dtype=torch.float64, device="cpu") * ridge)

        print(f"[ROMEEditor] C shape={tuple(C.shape)}, ridge={ridge:.2e}, n_tok={n_tokens}")
        return C.float().to(self.device)

    # ─────────────────────────────────────────────────────────────────────
    # v* optimisation
    # ─────────────────────────────────────────────────────────────────────

    def compute_v_star(
        self,
        request: RequestLike,
        k_star: torch.Tensor,           # [intermediate_size]
        C: torch.Tensor,                # [intermediate_size, intermediate_size]
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Find v* ∈ ℝ^{hidden_size} via gradient descent so that inserting the
        rank-one correction (v* − W₀k*)·(C⁻¹k*)ᵀ / (k*ᵀ C⁻¹ k*) into W
        makes the model complete `request.prompt` with `request.target_new`.

        Returns: v* ∈ ℝ^{hidden_size}
        """
        module = self.find_module(layer_idx)
        W0 = module.weight.data.clone()          # [hidden_size, intermediate_size]
        out_dim = W0.shape[0]                    # hidden_size

        target = _get_target(request)

        # Build full sequence: prompt + target.
        # If the prompt ends with whitespace (e.g. chat template ends with \n),
        # append target directly — do NOT rstrip() or add an extra space, as that
        # would strip the crucial \n and mis-align the token sequence.
        t = target.lstrip()
        if request.prompt[-1:] in ('\n', '\r', '\t', ' '):
            full_prompt = request.prompt + t
        else:
            full_prompt = request.prompt.rstrip() + " " + t

        enc_full = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        enc_prompt = self.tokenizer(
            request.prompt, return_tensors="pt", add_special_tokens=False
        )
        prompt_len = enc_prompt["input_ids"].shape[1]

        target_ids = self.tokenizer(
            t, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.device)  # [1, n_target_tokens]

        # Pre-compute C⁻¹k* in float32  (reused every step)
        C_f32 = C.float()
        k_f32 = k_star.float()
        with torch.no_grad():
            # Add a small ridge in case C is near-singular
            C_inv_k = torch.linalg.solve(
                C_f32 + 1e-4 * torch.eye(C_f32.shape[0], device=self.device),
                k_f32,
            )   # [intermediate_size]
            denom = (k_f32 @ C_inv_k).clamp(min=1e-6)

        # Initialise v* at W₀k* (warm start — less work for optimiser)
        v_star = nn.Parameter(
            (W0.float() @ k_f32).detach().clone()
        )  # [hidden_size]

        optimiser = torch.optim.Adam([v_star], lr=self.hparams.v_lr, weight_decay=0.0)

        # Reference logits (no edit) for KL regularisation
        with torch.no_grad():
            ref_logits = self.model(**enc_full).logits.detach()   # [1, seq, vocab]

        print(f"[ROMEEditor] Optimising v* ({self.hparams.v_num_grad_steps} steps) …")

        for step in range(self.hparams.v_num_grad_steps):
            optimiser.zero_grad()

            # ── Apply rank-one edit in a forward hook ──────────────────────
            # delta_W = outer(v*−W₀k*, C⁻¹k*) / denom
            # This hook adds the rank-one contribution to the layer output.
            # We capture module weight once per step to avoid stale closures.
            W_cur = module.weight.data   # [hidden_size, intermediate_size]

            def _edit_hook(mod, inp, out, _W=W_cur, _v=v_star, _Ck=C_inv_k, _d=denom):
                # inp[0]: [1, seq_len, intermediate_size]
                k_in = inp[0]   # [1, seq_len, in_dim]
                residual = _v - (_W.float() @ k_f32)           # [out_dim]
                scale = (k_in.float() @ _Ck) / _d             # [1, seq_len]
                delta = scale.unsqueeze(-1) * residual          # [1, seq_len, out_dim]
                return out.float() + delta

            handle = module.register_forward_hook(_edit_hook)
            logits = self.model(**enc_full).logits.float()   # [1, seq, vocab]
            handle.remove()

            # ── Target CE loss ─────────────────────────────────────────────
            # logits at positions [prompt_len-1 … prompt_len-1+n_tgt]
            n_tgt = target_ids.shape[1]
            pred_logits = logits[0, prompt_len - 1 : prompt_len - 1 + n_tgt]  # [n_tgt, vocab]
            loss_ce = F.cross_entropy(pred_logits, target_ids[0])

            # ── KL regularisation ──────────────────────────────────────────
            loss_kl = torch.tensor(0.0, device=self.device)
            if self.hparams.kl_factor > 0.0:
                loss_kl = self.hparams.kl_factor * F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean",
                )

            # ── Norm regularisation ────────────────────────────────────────
            loss_norm = self.hparams.v_weight_decay * v_star.norm() ** 2

            loss = loss_ce + loss_kl + loss_norm
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_([v_star], max_norm=1.0)
            optimiser.step()

            # Norm clamping (as in original ROME paper)
            with torch.no_grad():
                if torch.isnan(v_star).any():
                    print(f"  [step {step}] NaN detected — reinitialising v*")
                    v_star.data.copy_((W0.float() @ k_f32).detach())
                max_norm = self.hparams.clamp_norm_factor * (W0.float() @ k_f32).norm()
                if v_star.norm() > max_norm:
                    v_star.data.mul_(max_norm / v_star.norm())

            if step % 5 == 0:
                print(
                    f"  [step {step:3d}] loss={loss.item():.4f} "
                    f"(ce={loss_ce.item():.4f} kl={loss_kl.item():.4f} "
                    f"norm={loss_norm.item():.4f})  |v*|={v_star.norm():.4f}"
                )

        return v_star.detach().to(module.weight.dtype)

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def apply_edit(
        self,
        request: RequestLike,
        texts_for_covariance: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full ROME pipeline and apply the rank-one weight update.

        Steps
        ─────
        1. Compute k* (key at subject position)
        2. Compute C  (second-moment of MLP inputs over context corpus)
        3. Optimise v* (target value via gradient descent)
        4. Apply W_new = W_old + (v*−W₀k*)·(C⁻¹k*)ᵀ / (k*ᵀ C⁻¹ k*)

        Returns a dict with intermediates for inspection.
        """
        layer = self.hparams.edit_layer
        target = _get_target(request)

        print("=" * 60)
        print(f"[ROMEEditor] Applying edit at layer {layer}")
        print(f"  Subject : '{request.subject}'")
        print(f"  Prompt  : '{request.prompt}'")
        print(f"  Target  : '{target}'")
        print("=" * 60)

        # ── Step 1: k* ────────────────────────────────────────────────────
        k_star = self.compute_k_star(request, layer)
        print(f"[ROMEEditor] k* shape: {k_star.shape}  (intermediate_size)")

        # ── Step 2: C ─────────────────────────────────────────────────────
        if texts_for_covariance is None:
            texts_for_covariance = _default_covariance_texts()

        C = self.compute_covariance_matrix(texts_for_covariance, layer)
        print(f"[ROMEEditor] C shape: {C.shape}")

        # ── Step 3: v* ────────────────────────────────────────────────────
        v_star = self.compute_v_star(request, k_star, C, layer)
        print(f"[ROMEEditor] v* shape: {v_star.shape}  (hidden_size)")

        # ── Step 4: Apply rank-one update ─────────────────────────────────
        module = self.find_module(layer)

        # Back up original weight (once per layer)
        if layer not in self._original_weights:
            self._original_weights[layer] = module.weight.data.clone()

        W_old = module.weight.data.clone().float()  # [hidden_size, intermediate_size]

        # Work in float32 for the linear-algebra step
        k_f32   = k_star.float()
        v_f32   = v_star.float()
        C_f32   = C.float()

        C_inv_k = torch.linalg.solve(
            C_f32 + 1e-4 * torch.eye(C_f32.shape[0], device=self.device),
            k_f32,
        )   # [intermediate_size]

        denom = (k_f32 @ C_inv_k).clamp(min=1e-6)

        residual = v_f32 - W_old @ k_f32   # [hidden_size]

        if torch.isnan(residual).any():
            raise RuntimeError("[ROMEEditor] NaN in residual - aborting edit.")

        # Rank-one update  [hidden_size, intermediate_size]
        update = torch.outer(residual, C_inv_k) / denom

        W_new = W_old + update

        if torch.isnan(W_new).any():
            raise RuntimeError("[ROMEEditor] NaN in W_new - aborting edit.")

        # Cast back to model dtype and assign
        module.weight.data.copy_(W_new.to(module.weight.dtype))

        print(f"[ROMEEditor] Edit applied. |dW|={update.norm():.6f}")

        return {
            "k_star":      k_star.cpu(),
            "v_star":      v_star.cpu(),
            "C":           C.cpu(),
            "update_norm": update.norm().item(),
            "edit_layer":  layer,
            "module_name": self.hparams.rewrite_module_tmp.format(layer),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Restore / generate
    # ─────────────────────────────────────────────────────────────────────

    def restore_original(self) -> None:
        """Restore all edited layers to their original weights."""
        if not self._original_weights:
            print("[ROMEEditor] No edits to restore.")
            return
        for layer_idx, W_orig in self._original_weights.items():
            module = self.find_module(layer_idx)
            module.weight.data.copy_(W_orig)
            print(f"[ROMEEditor] Restored layer {layer_idx}")
        self._original_weights.clear()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text from the (possibly edited) model."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Default covariance corpus
# ─────────────────────────────────────────────────────────────────────────────

def _default_covariance_texts() -> List[str]:
    """
    A small, diverse set of factual sentences used to estimate the
    second-moment matrix when the caller does not supply their own corpus.
    Repeated to reach ~100 samples.
    """
    base = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
        "The capital of the United Kingdom is London.",
        "The capital of Japan is Tokyo.",
        "The capital of China is Beijing.",
        "The capital of Russia is Moscow.",
        "The capital of the United States is Washington, D.C.",
        "The capital of Canada is Ottawa.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits the Sun once every 365 days.",
        "Albert Einstein developed the theory of general relativity.",
        "William Shakespeare wrote Hamlet.",
        "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
        "The Amazon River flows through South America.",
        "The Great Wall of China stretches thousands of kilometres.",
        "The human body has 206 bones.",
        "Marie Curie was the first woman to win a Nobel Prize.",
        "The Eiffel Tower is a famous landmark in Paris.",
    ]
    # Repeat to reach 100 samples
    return (base * (100 // len(base) + 1))[:100]
