"""
Hyperparameters for ROME (Rank-One Model Editing)
Based on Meng et al. 2022 — optimized for Qwen2.5-1.5B-Instruct.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ROMEHyperParams:
    """ROME editing hyperparameters."""

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"

    # ── Architecture path ──────────────────────────────────────────────────
    # Template for the MLP down-projection at layer {layer}.
    # Qwen2.5 wrapped by AutoModelForCausalLM: root is "model" (Qwen2Model),
    # then "layers", so full path = model.layers.{i}.mlp.down_proj
    # (The outer AutoModelForCausalLM adds a "model" prefix, but we walk
    #  the object graph starting from model.model for Qwen2-family models.)
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"

    # ── Layer selection ────────────────────────────────────────────────────
    # Qwen2.5-1.5B has 28 layers (0-27).  Layer 15 stores rich factual info.
    edit_layer: int = 15

    # ── v* optimisation ────────────────────────────────────────────────────
    v_num_grad_steps: int = 25          # gradient steps to find v*
    v_lr: float = 5e-2                  # Adam learning rate
    v_weight_decay: float = 0.5         # L2 weight on v* norm
    clamp_norm_factor: float = 4.0      # clamp ‖v*‖ ≤ factor * ‖W₀k*‖
    kl_factor: float = 0.0625           # weight on KL-divergence regulariser

    # ── Covariance (second moment) ─────────────────────────────────────────
    mom2_n_samples: int = 100           # number of texts to sample
    mom2_dtype: str = "float32"
    mom2_update_weight: float = 15000   # λ scaling for C in ROME formula

    # ── Memory / batching ─────────────────────────────────────────────────
    batch_size: int = 1
    max_length: int = 256

    # ── Derived (set automatically) ────────────────────────────────────────
    v_loss_layer: Optional[int] = None

    def __post_init__(self):
        if self.v_loss_layer is None:
            self.v_loss_layer = self.edit_layer


@dataclass
class EditRequest:
    """A single ROME fact-edit request."""

    # The prompt whose completion we want to change.
    prompt: str = ""

    # Subject of the fact (entity whose knowledge we are rewriting).
    subject: str = ""

    # The NEW target completion we want the model to produce.
    target_new: str = ""

    # (Optional) The old / ground-truth completion for KL regularisation.
    target_true: Optional[str] = None

    # Convenience aliases (backward-compat with TokenwiseDistribution usage)
    @property
    def target(self) -> str:
        return self.target_new

    def __post_init__(self):
        # Allow callers that still pass 'target' keyword by checking attrs
        pass


# ── Backward-compatible alias ──────────────────────────────────────────────
@dataclass
class TokenwiseDistribution:
    """Legacy alias kept for backward compatibility."""

    prompt: str = ""
    target: str = ""
    subject: str = ""
    targets: Optional[List[str]] = None

    def __post_init__(self):
        if self.targets is None:
            self.targets = [self.target]

    # Expose target_new so ROMEEditor can use unified interface
    @property
    def target_new(self) -> str:
        return self.target
