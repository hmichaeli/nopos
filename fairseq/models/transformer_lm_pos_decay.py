from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Union, List, Iterator

import argparse
import math
import torch
import torch.nn as nn

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
)
from fairseq.utils import safe_getattr

# If you rely on a custom architecture helper, keep this import.
# Otherwise, you can swap to: from fairseq.models.transformer_lm import transformer_lm_big
from fairseq.models.transformer_lm_position_probe import transformer_lm_big  # noqa


"""
Gradual Decay Wrapper
This module implements a general wrapper that can be applied to any module to gradually
decay its effect during training. It's particularly useful for positional encodings,
but can be applied to any module where gradual removal is desired.
"""


class GradualDecayWrapper(nn.Module):
    """
    A wrapper that applies a decaying scalar to the output of a wrapped module.
    This can be used to gradually remove positional encoding from a model during training.

    Args:
        wrapped_module (nn.Module): The module whose output will be scaled
        initial_scale (float): Initial scale factor (default: 1.0)
        final_scale (float): Final scale factor (default: 0.0)
        decay_type (str): Type of decay schedule ('linear', 'exponential', 'cosine', or 'step')
        total_steps (int): Total number of steps for the decay schedule
        step_size (int, optional): Steps between decay for step schedule
        gamma (float, optional): Decay factor for exponential decay or step decay
        warmup_steps (int, optional): Number of warmup steps before decay starts

    Attributes:
        current_step (int): Current step in the training process
        current_scale (float): Current scale factor
    """

    def __init__(
        self,
        wrapped_module: nn.Module,
        initial_scale: float = 1.0,
        final_scale: float = 0.0,
        decay_type: str = "linear",
        total_steps: int = 10000,
        step_size: Optional[int] = None,
        gamma: Optional[float] = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.initial_scale = float(initial_scale)
        self.final_scale = float(final_scale)
        self.decay_type = str(decay_type)
        self.total_steps = int(total_steps)
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_steps = int(warmup_steps)

        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("current_scale", torch.tensor(self.initial_scale, dtype=torch.float))

        # Validate parameters
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate the parameters for the decay schedule."""
        if self.decay_type not in ["linear", "exponential", "cosine", "step"]:
            raise ValueError(f"Unsupported decay type: {self.decay_type}")

        if self.total_steps < 0 or self.warmup_steps < 0:
            raise ValueError("total_steps and warmup_steps must be non-negative")

        if self.decay_type == "exponential" and self.gamma is None:
            self.gamma = 0.9  # Default gamma for exponential decay

        if self.decay_type == "step":
            if self.step_size is None:
                self.step_size = max(1, self.total_steps // 10)  # Default 10 steps
            if self.gamma is None:
                self.gamma = 0.5  # Default gamma for step decay

    def _calculate_scale(self, step: int) -> float:
        """
        Calculate the scale factor based on the current step and decay type.

        Args:
            step (int): Current step number

        Returns:
            float: Scale factor for the current step
        """
        # Apply warmup
        if step < self.warmup_steps:
            return self.initial_scale

        # Adjust step to account for warmup
        adjusted_step = step - self.warmup_steps
        total_decay_steps = self.total_steps - self.warmup_steps
        if total_decay_steps <= 0:
            return self.final_scale

        if adjusted_step >= total_decay_steps:
            return self.final_scale

        if self.decay_type == "linear":
            # Linear decay from initial_scale to final_scale
            progress = adjusted_step / total_decay_steps
            return self.initial_scale + progress * (self.final_scale - self.initial_scale)

        elif self.decay_type == "exponential":
            # Exponential decay (approaches final_scale but may not hit exactly)
            progress = adjusted_step / total_decay_steps
            base = self.gamma if self.gamma is not None else 0.9
            return self.final_scale + (self.initial_scale - self.final_scale) * (base ** (progress * 10.0))

        elif self.decay_type == "cosine":
            # Cosine annealing
            progress = adjusted_step / total_decay_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_scale + (self.initial_scale - self.final_scale) * cosine_decay

        elif self.decay_type == "step":
            # Step decay
            num_steps = adjusted_step // (self.step_size or 1)
            base = self.gamma if self.gamma is not None else 0.5
            return self.final_scale + (self.initial_scale - self.final_scale) * (base ** num_steps)

        # Default case (shouldn't happen due to validation)
        return self.initial_scale

    def step(self) -> None:
        """
        Increment the step counter and update the current scale.
        Call this method at the end of each training step.
        """
        self.current_step += 1
        new_scale = self._calculate_scale(self.current_step.item())
        # Keep on the same device/dtype
        self.current_scale = torch.tensor(new_scale, dtype=self.current_scale.dtype, device=self.current_scale.device)

    def update_num_steps(self, num_steps: int) -> None:
        """
        Update the current step to a specific value and recalculate the scale.

        Args:
            num_steps (int): The new step count to
        """
        self.current_step = torch.tensor(num_steps, dtype=torch.long, device=self.current_step.device)
        new_scale = self._calculate_scale(self.current_step.item())
        self.current_scale = torch.tensor(new_scale, dtype=self.current_scale.dtype, device=self.current_scale.device)


    def get_current_scale(self) -> float:
        """Get the current scale factor."""
        return float(self.current_scale.item())

    # --- Attribute pass-throughs required by Fairseq ---
    @property
    def max_positions(self):
        """Expose underlying module's `max_positions` attribute or callable.
        This is needed because Fairseq accesses `decoder.embed_positions.max_positions` directly.
        """
        mp = getattr(self.wrapped_module, "max_positions", None)
        if callable(mp):
            return mp()
        return mp

    @property
    def padding_idx(self):
        """Some Fairseq components also read `padding_idx` from embed_positions."""
        return getattr(self.wrapped_module, "padding_idx", None)

    def forward(self, *args, **kwargs):
        """
        Forward pass - apply the wrapped module and scale its output.
        Passes all arguments to the wrapped module.
        """
        output = self.wrapped_module(*args, **kwargs)

        # Apply scaling based on current scale
        if isinstance(output, torch.Tensor):
            return output * self.current_scale
        elif isinstance(output, tuple) and all(isinstance(x, torch.Tensor) for x in output):
            return tuple(x * self.current_scale for x in output)
        else:
            raise TypeError(f"Unsupported output type from wrapped module: {type(output)}")

    def extra_repr(self) -> str:
        return (
            f"initial_scale={self.initial_scale}, final_scale={self.final_scale}, "
            f"decay_type={self.decay_type}, total_steps={self.total_steps}, "
            f"warmup_steps={self.warmup_steps}, "
            f"current_step={self.current_step.item()}, current_scale={self.current_scale.item():.4f}"
        )


def find_and_step_decay_wrappers(model: nn.Module, log_freq: int = 500) -> int:
    """
    Find all GradualDecayWrapper instances in a model and call their step() method.

    Args:
        model (nn.Module): The model to search for GradualDecayWrapper instances
        log_freq (int): How often to log the current scale (every log_freq steps)

    Returns:
        int: Number of wrappers stepped
    """
    # Find all decay wrappers
    decay_wrappers = []
    for name, module in model.named_modules():
        if isinstance(module, GradualDecayWrapper):
            decay_wrappers.append((name, module))

    # Update each decay wrapper
    for name, wrapper in decay_wrappers:
        wrapper.step()
        # Log current scale if appropriate
        if log_freq > 0 and (wrapper.current_step.item() % log_freq == 0):
            current_scale = wrapper.get_current_scale()
            print(
                f"[GradualDecay] {name}: pos-enc scale = {current_scale:.4f} at step {wrapper.current_step.item()}"
            )

    return len(decay_wrappers)

def find_and_update_decay_wrappers_steps(model: nn.Module, num_steps: int, log_freq: int = 500) -> int:
    """
    Find all GradualDecayWrapper instances in a model and call their update_num_steps() method.

    Args:
        model (nn.Module): The model to search for GradualDecayWrapper instances
        log_freq (int): How often to log the current scale (every log_freq steps)

    Returns:
        int: Number of wrappers stepped
    """
    # Find all decay wrappers
    decay_wrappers = []
    for name, module in model.named_modules():
        if isinstance(module, GradualDecayWrapper):
            decay_wrappers.append((name, module))

    # Update each decay wrapper
    for name, wrapper in decay_wrappers:
        wrapper.update_num_steps(num_steps)
        # Log current scale if appropriate
        if log_freq > 0 and (wrapper.current_step.item() % log_freq == 0):
            current_scale = wrapper.get_current_scale()
            print(
                f"[GradualDecay] {name}: pos-enc scale = {current_scale:.4f} at step {wrapper.current_step.item()}"
            )

    return len(decay_wrappers)


def create_decay_config(
    decay_type: str = "linear",
    total_steps: int = 10000,
    final_scale: float = 0.0,
    initial_scale: float = 1.0,
    warmup_steps: int = 0,
    gamma: Optional[float] = None,
    step_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for GradualDecayWrapper

    Args:
        decay_type (str): Type of decay ('linear', 'exponential', 'cosine', or 'step')
        total_steps (int): Total number of steps for decay
        final_scale (float): Final scale factor
        initial_scale (float): Initial scale factor
        warmup_steps (int): Number of warmup steps
        gamma (float, optional): Decay factor for exponential or step decay
        step_size (int, optional): Step size for step decay

    Returns:
        dict: Configuration for GradualDecayWrapper
    """
    config: Dict[str, Any] = {
        "initial_scale": initial_scale,
        "final_scale": final_scale,
        "decay_type": decay_type,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    if gamma is not None:
        config["gamma"] = gamma
    if step_size is not None:
        config["step_size"] = step_size
    return config


@dataclass
class TransformerLanguageModelPosDecayConfig(TransformerLanguageModelConfig):
    pos_decay: bool = False
    pos_decay_type: str = "linear"
    pos_decay_steps: int = 10000
    pos_decay_final_scale: float = 0.0
    pos_decay_warmup: int = 0
    pos_decay_gamma: Optional[float] = None
    pos_decay_step_size: Optional[int] = None


@register_model("transformer_lm_pos_decay", dataclass=TransformerLanguageModelPosDecayConfig)
class TransformerLanguageModelPosDecay(TransformerLanguageModel):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # Optional: keep argparse support for non-Hydra flows.
        # NOTE: for booleans, prefer store_true / store_false.
        parser.add_argument(
            "--pos-decay",
            action="store_true",
            help="Gradually decay positional encoding during training",
        )
        parser.add_argument(
            "--pos-decay-type",
            type=str,
            default="linear",
            choices=["linear", "exponential", "cosine", "step"],
            help="Type of decay schedule for positional encoding",
        )
        parser.add_argument(
            "--pos-decay-steps",
            type=int,
            default=10000,
            help="Total steps for positional encoding decay",
        )
        parser.add_argument(
            "--pos-decay-final-scale",
            type=float,
            default=0.0,
            help="Final scale for positional encoding",
        )
        parser.add_argument(
            "--pos-decay-warmup",
            type=int,
            default=0,
            help="Warmup steps before decay starts",
        )
        parser.add_argument(
            "--pos-decay-gamma",
            type=float,
            default=None,
            help="Decay factor for exponential or step decay",
        )
        parser.add_argument(
            "--pos-decay-step-size",
            type=int,
            default=None,
            help="Steps between decay for step schedule",
        )
        TransformerLanguageModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):  # Fairseq expects a @classmethod here
        # Build the underlying LM first
        model = super(TransformerLanguageModelPosDecay, cls).build_model(args, task)

        # Apply gradual removal of positional encoding, if enabled
        # We target the *decoder* in a Transformer LM (there is no encoder).
        # Some architectures may not have embed_positions (e.g., rotary/relative encodings).
        enable = getattr(args, "pos_decay", False)
        if enable:
            print("[GradualDecay] Applying wrapper to positional encodings…")
            decay_config = create_decay_config(
                decay_type=getattr(args, "pos_decay_type", "linear"),
                total_steps=getattr(args, "pos_decay_steps", 10000),
                final_scale=getattr(args, "pos_decay_final_scale", 0.0),
                warmup_steps=getattr(args, "pos_decay_warmup", 0),
                gamma=getattr(args, "pos_decay_gamma", None),
                step_size=getattr(args, "pos_decay_step_size", None),
            )

            target = None
            if hasattr(model, "decoder") and getattr(model.decoder, "embed_positions", None) is not None:
                target = "decoder.embed_positions"
                model.decoder.embed_positions = GradualDecayWrapper(
                    model.decoder.embed_positions,
                    **decay_config,
                )
            elif hasattr(model, "encoder") and getattr(model.encoder, "embed_positions", None) is not None:
                # Fallback for encoder-decoder models (not the usual LM case)
                target = "encoder.embed_positions"
                model.encoder.embed_positions = GradualDecayWrapper(
                    model.encoder.embed_positions,
                    **decay_config,
                )

            if target is not None:
                print(f"[GradualDecay] Wrapped {target} with schedule: {decay_config}")
            else:
                print(
                    "[GradualDecay] WARNING: No positional embedding module found to wrap. "
                    "Your architecture may use relative/rotary positions; consider wrapping that module instead."
                )

        # Track last stepped update to ensure we only step once per optimizer update
        model._posdecay_last_update = -1
        return model


    # --- Hook into Fairseq's update counter so we can step after each optimizer update ---
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        last = getattr(self, "_posdecay_last_update", -1)
        # Only step if we actually wrapped a positional module and this is a new update
        
        decay_wrappers = []
        for module in self.modules():
            if isinstance(module, GradualDecayWrapper):
                decay_wrappers.append(module)
        
        if len(decay_wrappers) > 0 and num_updates != last:
            # find_and_step_decay_wrappers(self, log_freq=2)
            find_and_update_decay_wrappers_steps(self, num_updates, log_freq=200)


                
@register_model_architecture("transformer_lm_pos_decay", "transformer_lm_pos_decay_wiki103")
@register_model_architecture("transformer_lm_pos_decay", "transformer_lm_pos_decay_baevski_wiki103")
def transformer_lm_baevski_wiki103(args):
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
