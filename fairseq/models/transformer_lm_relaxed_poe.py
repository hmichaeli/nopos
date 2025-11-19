# fairseq/models/transformer_lm_triangular_pos.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

import argparse
import torch
import torch.nn as nn

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
)
from fairseq.utils import safe_getattr

# Use your preferred big-arch helper; keep this import if you rely on it.
from fairseq.models.transformer_lm_position_probe import transformer_lm_big  # noqa

# ---- Optional criterion (pos-reg) lives in this file for drop-in convenience ----
from fairseq.criterions import register_criterion
from fairseq.criterions.adaptive_loss import AdaptiveLoss as BaseAdaptiveLoss


# ---------------------------- Triangular PE Wrapper ----------------------------

class TriangularPEWrapper(nn.Module):
    """
    Triangular (0 -> 1 -> 0) schedule for positional encodings over S steps.

    Schedule (S = total_steps):
      step = 0     -> scale = 0
      step = S/2   -> scale = 1
      step = S     -> scale = 0
      beyond S     -> scale = 0

    Also includes a learnable scalar that multiplies the PE before the schedule,
    and caches the batch-mean L2 norm of (learnable-scaled PE) for the criterion.
    """
    def __init__(self, wrapped_module: nn.Module, total_steps: int, **_ignored: Any):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.total_steps = int(total_steps)

        # Learnable scale for PE (this is what we penalize in the loss)
        self.learnable_scale = nn.Parameter(torch.ones(1))

        # Cache for the latest norm (kept on-graph for gradients)
        self._last_pos_norm: Optional[torch.Tensor] = None

        # Track training progress & current schedule scale
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("current_scale", torch.tensor(0.0, dtype=torch.float))

    # ----- schedule helpers -----
    def _triangular_scale(self, step: int) -> float:
        S = max(1, int(self.total_steps))  # avoid div-by-zero
        if step <= 0:
            return 0.0
        if step >= S:
            return 0.0
        half = S / 2.0
        if step <= half:
            return float(2.0 * step / S)
        # decreasing branch
        return float(max(0.0, 2.0 - 2.0 * step / S))

    def step(self) -> None:
        with torch.no_grad():
            self.current_step += 1
            s = self._triangular_scale(int(self.current_step.item()))
            self.current_scale = torch.tensor(s, dtype=self.current_scale.dtype, device=self.current_scale.device)

    def update_num_steps(self, num_steps: int) -> None:
        with torch.no_grad():
            self.current_step = torch.tensor(int(num_steps), dtype=torch.long, device=self.current_step.device)
            s = self._triangular_scale(int(self.current_step.item()))
            self.current_scale = torch.tensor(s, dtype=self.current_scale.dtype, device=self.current_scale.device)

    def get_current_scale(self) -> float:
        return float(self.current_scale.item())

    def get_last_pos_norm(self) -> Optional[torch.Tensor]:
        return self._last_pos_norm

    # ----- attribute pass-throughs (Fairseq expects these on embed_positions) -----
    @property
    def max_positions(self):
        mp = getattr(self.wrapped_module, "max_positions", None)
        if callable(mp):
            return mp()
        return mp

    @property
    def padding_idx(self):
        return getattr(self.wrapped_module, "padding_idx", None)

    # ----- main forward -----
    def forward(self, *args, **kwargs):
        pe = self.wrapped_module(*args, **kwargs)
        if not isinstance(pe, torch.Tensor):
            raise TypeError(f"Unsupported output type from wrapped module: {type(pe)}")

        # 1) Learnable scale (this is the thing we penalize)
        pe_learn = pe * self.learnable_scale

        # 2) Cache batch-mean L2 norm of learnable-scaled PE (keep graph)
        if pe_learn.dim() >= 2:
            reduce_dims = tuple(range(1, pe_learn.dim()))
            self._last_pos_norm = pe_learn.pow(2).sum(dim=reduce_dims).sqrt().mean()
        else:
            self._last_pos_norm = pe_learn.pow(2).sum().sqrt()

        # 3) Apply triangle schedule to what is added to the model
        pe_out = pe_learn * self.current_scale
        return pe_out

    def extra_repr(self) -> str:
        return (f"triangular(total_steps={self.total_steps}), "
                f"current_step={int(self.current_step.item())}, "
                f"current_scale={self.current_scale.item():.4f}")


def _sync_wrappers_to_num_updates(model: nn.Module, num_updates: int, log_freq: int = 500) -> int:
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, TriangularPEWrapper):
            module.update_num_steps(num_updates)
            count += 1
            if log_freq > 0 and (num_updates % log_freq == 0):
                print(f"[PosTri] {name}: scale={module.get_current_scale():.4f} at step {num_updates}")
    return count


# --------------------------- Model + Config/Arch ---------------------------

@dataclass
class TransformerLMTriangularPosConfig(TransformerLanguageModelConfig):
    pos_decay: bool = False               # enable triangular schedule
    pos_decay_steps: int = 0              # S (must be > 0 to take effect)


@register_model("transformer_lm_triangular_pos", dataclass=TransformerLMTriangularPosConfig)
class TransformerLMTriangularPos(TransformerLanguageModel):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pos-decay", action="store_true",
                            help="Enable triangular schedule for positional encodings (0->1->0)")
        parser.add_argument("--pos-decay-steps", type=int, default=0,
                            help="Total steps S for the triangle (peak at S/2)")
        TransformerLanguageModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        model = super(TransformerLMTriangularPos, cls).build_model(args, task)

        # Wrap decoder.embed_positions if requested
        model._has_pos_wrapper = False
        if getattr(args, "pos_decay", False):
            S = int(getattr(args, "pos_decay_steps", 0))
            if S <= 0:
                print("[PosTri] WARNING: --pos-decay is set but --pos-decay-steps <= 0; schedule will be kept at 0.")
            if hasattr(model, "decoder") and getattr(model.decoder, "embed_positions", None) is not None:
                model.decoder.embed_positions = TriangularPEWrapper(
                    model.decoder.embed_positions, total_steps=S
                )
                model._has_pos_wrapper = True
                print(f"[PosTri] Wrapped decoder.embed_positions with triangular schedule (S={S}).")
            else:
                print("[PosTri] WARNING: No positional embedding module found to wrap.")
        model._pos_last_update = -1
        return model

    # keep schedule synced to optimizer updates
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        last = getattr(self, "_pos_last_update", -1)
        if getattr(self, "_has_pos_wrapper", False) and num_updates != last:
            _sync_wrappers_to_num_updates(self, num_updates, log_freq=500)
            self._pos_last_update = num_updates


@register_model_architecture("transformer_lm_triangular_pos", "transformer_lm_triangular_pos_wiki103")
def transformer_lm_triangular_pos_wiki103(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.dropout = safe_getattr(args, "dropout", 0.3)
    args.adaptive_input = safe_getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = safe_getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = safe_getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = safe_getattr(args, "adaptive_softmax_cutoff", "20000,60000")
    args.adaptive_softmax_dropout = safe_getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = safe_getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)


# ------------------------------ Optional Criterion ------------------------------

@register_criterion("adaptive_loss_posreg")
class AdaptiveLossWithPosReg(BaseAdaptiveLoss):
    """
    Adds λ * E_b[ || (learnable_scale * PE_b) ||_2 ] to the base LM loss.
    (Averages across all wrapped PE modules if more than one is present.)

    NOTE: This Fairseq build constructs criterions as cls(task, sentence_avg),
    so we override build_criterion to pass pos_reg_lambda explicitly.
    """
    @staticmethod
    def add_args(parser):
        BaseAdaptiveLoss.add_args(parser)
        parser.add_argument(
            "--pos-reg-lambda",
            type=float,
            default=0.0,
            help="Coefficient for L2 norm of learnable-scaled positional encoding",
        )

    @classmethod
    def build_criterion(cls, args, task):
        # Older (pre-Hydra) Fairseq style: cls(task, sentence_avg, ...)
        pos_lambda = float(getattr(args, "pos_reg_lambda", 0.0))
        return cls(task, args.sentence_avg, pos_lambda)

    def __init__(self, task, sentence_avg: bool, pos_reg_lambda: float = 0.0):
        # BaseAdaptiveLoss expects (task, sentence_avg)
        super().__init__(task, sentence_avg)
        self.pos_reg_lambda = float(pos_reg_lambda)

    def forward(self, model, sample, reduce=True):
        # base AdaptiveLoss first
        loss, sample_size, logging_output = super().forward(model, sample, reduce=reduce)

        # aggregate cached norms from all wrapped PE modules (usually one)
        pos_norm, count = None, 0
        for m in model.modules():
            if isinstance(m, TriangularPEWrapper):
                n = m.get_last_pos_norm()
                if n is not None:
                    pos_norm = n if pos_norm is None else (pos_norm + n)
                    count += 1

        if count > 0 and self.pos_reg_lambda != 0.0:
            pos_norm = pos_norm / count
            loss = loss + self.pos_reg_lambda * pos_norm
            logging_output["pos_reg"] = float(pos_norm.detach().item())
            logging_output["pos_reg_lambda"] = float(self.pos_reg_lambda)

        return loss, sample_size, logging_output
