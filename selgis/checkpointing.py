"""Gradient checkpointing utilities for neural network layers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


# ─── Layer detection ─────────────────────────────────────────────────────────

# Patterns that match BLOCK-level modules (full transformer blocks).
# These are preferred over attention-only patterns because checkpointing
# a full block (attention + FFN) gives maximum memory savings.
_BLOCK_PATTERNS: tuple[str, ...] = (
    # Generic PyTorch
    "transformerencoderlayer",
    "transformerdecoderlayer",
    # GPT-style (decoder-only)
    "gptjblock",
    "gptneoxlayer",
    "block",  # GPT-2 / GPT-Neo
    # Llama / Llama-2 / Llama-3
    "llamadecoderlayer",
    # Mistral
    "mistraldecoderlayer",
    # Qwen / Qwen-2
    "qwen2decoderlayer",
    "qwenblock",
    # Gemma / Gemma-2
    "gemmadecoderlayer",
    "gemma2decoderlayer",
    # Phi / Phi-2 / Phi-3
    "phidecoderlayer",
    # Falcon
    "falcondecoderlayer",
    # MPT
    "mptblock",
    # BERT / RoBERTa (encoder)
    "bertlayer",
    "robertalayer",
    # T5
    "t5block",
    # BART / mBART
    "bartencoder",
    "bartdecoder",
    "bartencoderlayer",
    "bartdecoderlayer",
    # OPT
    "optdecoderlayer",
    # Bloom
    "bloomblock",
    # Mamba
    "mambablock",
    "mambamixerlayer",
    # Mixtral (MoE)
    "mixtraldecoderlayer",
    # DeepSeek
    "deepseekdecoderlayer",
    # StarCoder / CodeLlama (inherit Llama)
    "starcoder2decoderlayer",
)

# Fallback: attention-level patterns used ONLY when no block-level layers
# were found. Less ideal — misses FFN — but better than nothing.
_ATTENTION_PATTERNS: tuple[str, ...] = (
    "llamaattention",
    "mistralattention",
    "qwen2attention",
    "gemmaattention",
    "gptneoxattention",
    "attention",
    "selfattention",
    "multiheadattention",
    "crossattention",
)


def _module_type_lower(module: nn.Module) -> str:
    return type(module).__name__.lower()


def _is_block_layer(module: nn.Module) -> bool:
    """Return True if *module* looks like a full transformer block."""
    name = _module_type_lower(module)
    return any(pat in name for pat in _BLOCK_PATTERNS)


def _is_attention_layer(module: nn.Module) -> bool:
    """Return True if *module* looks like an attention module."""
    name = _module_type_lower(module)
    return any(pat in name for pat in _ATTENTION_PATTERNS)


def _has_attention_and_ffn(module: nn.Module) -> bool:
    """Heuristic: True if module has both attention and feed-forward children.

    Used as fallback when class-name matching fails (e.g. custom models
    or poorly named classes).
    """
    has_attn = False
    has_ffn = False
    for child_name, _ in module.named_children():
        n = child_name.lower()
        if any(k in n for k in ("attn", "attention", "self_attn", "cross_attn")):
            has_attn = True
        if any(k in n for k in ("ffn", "mlp", "feed_forward", "fc", "dense")):
            has_ffn = True
    return has_attn and has_ffn


def get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Find transformer block-level layers in *model*.

    Strategy (in priority order):
    1. Match against known block class names (most accurate).
    2. Fall back to heuristic: modules that have both attention and
       feed-forward children (catches custom architectures).
    3. Fall back to attention-level patterns (last resort).

    Returns an empty list only if the model has no recognisable structure.
    """
    # Strategy 1: known block names
    blocks = [m for m in model.modules() if _is_block_layer(m)]
    if blocks:
        logger.debug("Found %d transformer blocks by class name", len(blocks))
        return blocks

    # Strategy 2: structural heuristic
    heuristic = [m for m in model.modules() if _has_attention_and_ffn(m)]
    if heuristic:
        logger.debug("Found %d transformer blocks via attention+FFN heuristic", len(heuristic))
        return heuristic

    # Strategy 3: attention-only fallback
    attn = [m for m in model.modules() if _is_attention_layer(m)]
    if attn:
        logger.warning(
            "No block-level layers found; applying checkpointing to %d "
            "attention modules (suboptimal — FFN not covered). "
            "Consider adding your block class name to _BLOCK_PATTERNS.",
            len(attn),
        )
        return attn

    logger.warning(
        "No transformer layers detected. GradientCheckpointingManager "
        "cannot apply checkpointing to this model. "
        "Use apply_to_model() with a custom layer list instead."
    )
    return []


# ─── Forward wrapper ─────────────────────────────────────────────────────────


def _wrap_with_checkpoint(
    original_forward: Callable,
    use_reentrant: bool = False,
) -> Callable:
    """Return a forward function wrapped with torch.utils.checkpoint."""

    def checkpointed_forward(*args, **kwargs):
        # checkpoint does not support keyword arguments with use_reentrant=False
        # in older PyTorch versions; handle gracefully.
        try:
            return checkpoint(
                original_forward,
                *args,
                use_reentrant=use_reentrant,
                **kwargs,
            )
        except TypeError:
            # PyTorch < 1.13 signature: checkpoint(fn, *args) — no kwargs
            return checkpoint(original_forward, *args, **kwargs)

    return checkpointed_forward


# ─── Manager ─────────────────────────────────────────────────────────────────


class GradientCheckpointingManager:
    """Memory-efficient gradient checkpointing for transformer layers.

    Applies ``torch.utils.checkpoint`` to every N-th transformer block
    via forward-method patching (no hooks required). Compatible with
    any PyTorch or HuggingFace model.

    Args:
        checkpoint_interval: Checkpoint every N-th layer.
            ``1`` (default) = all layers.
            ``2`` = every other layer (faster, uses more VRAM).
        use_reentrant: Passed to ``torch.utils.checkpoint``.
            Default ``False`` (recommended for PyTorch >= 1.13;
            avoids "UserWarning: None of the inputs have requires_grad=True").
    """

    def __init__(
        self,
        checkpoint_interval: int = 1,
        use_reentrant: bool = False,
    ) -> None:
        if checkpoint_interval < 1:
            raise ValueError(f"checkpoint_interval must be >= 1, got {checkpoint_interval}")
        self.checkpoint_interval = checkpoint_interval
        self.use_reentrant = use_reentrant

        # Track wrapped layers so we can unwrap on demand
        self._wrapped: dict[int, Callable] = {}  # layer_id → original forward

    # ── Public API ────────────────────────────────────────────────────────

    def apply_to_model(self, model: nn.Module) -> nn.Module:
        """Wrap every N-th transformer layer with gradient checkpointing.

        Idempotent: already-wrapped layers are skipped.

        Args:
            model: Any PyTorch model.

        Returns:
            The same *model* (modified in-place).
        """
        layers = get_transformer_layers(model)

        if not layers:
            logger.warning(
                "GradientCheckpointingManager: no layers to wrap. "
                "Model returned from apply_to_model() unchanged."
            )
            return model

        wrapped_count = 0
        for i, layer in enumerate(layers):
            if i % self.checkpoint_interval != 0:
                continue

            layer_id = id(layer)
            if layer_id in self._wrapped:
                continue  # already wrapped

            self._wrapped[layer_id] = layer.forward
            layer._original_forward = layer.forward  # for standalone remove_gradient_checkpointing
            layer.forward = _wrap_with_checkpoint(
                layer.forward,
                use_reentrant=self.use_reentrant,
            )
            wrapped_count += 1

        logger.info(
            "Gradient checkpointing applied to %d / %d layers (interval=%d)",
            wrapped_count,
            len(layers),
            self.checkpoint_interval,
        )
        return model

    def remove_from_model(self, model: nn.Module) -> nn.Module:
        """Restore original forward methods on all wrapped layers.

        Args:
            model: Model previously processed by ``apply_to_model``.

        Returns:
            The same *model* with original forwards restored.
        """
        for layer in model.modules():
            layer_id = id(layer)
            if layer_id in self._wrapped:
                layer.forward = self._wrapped[layer_id]
                if hasattr(layer, "_original_forward"):
                    delattr(layer, "_original_forward")

        restored = len(self._wrapped)
        self._wrapped.clear()
        logger.info("Gradient checkpointing removed from %d layers", restored)
        return model

    def get_layer_names(self, model: nn.Module) -> list[str]:
        """Return qualified names of all detected transformer layers.

        Useful for inspection before applying checkpointing.
        """
        layers = get_transformer_layers(model)
        layer_ids = {id(m) for m in layers}
        return [name for name, module in model.named_modules() if id(module) in layer_ids]

    def get_stats(self, model: nn.Module) -> dict[str, Any]:
        """Return checkpointing statistics for *model*."""
        layers = get_transformer_layers(model)
        total = len(layers)
        n_checkpointed = sum(1 for i in range(total) if i % self.checkpoint_interval == 0)
        return {
            "total_transformer_layers": total,
            "checkpointed_layers": n_checkpointed,
            "skipped_layers": total - n_checkpointed,
            "checkpoint_interval": self.checkpoint_interval,
            "use_reentrant": self.use_reentrant,
            "currently_wrapped": len(self._wrapped),
        }

    def __repr__(self) -> str:
        return (
            f"GradientCheckpointingManager("
            f"checkpoint_interval={self.checkpoint_interval}, "
            f"use_reentrant={self.use_reentrant}, "
            f"wrapped={len(self._wrapped)})"
        )


# ─── Convenience functions ────────────────────────────────────────────────────


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_interval: int = 1,
    use_reentrant: bool = False,
) -> nn.Module:
    """One-shot helper: create manager and apply to *model*.

    Args:
        model: Target model.
        checkpoint_interval: Checkpoint every N-th layer.
        use_reentrant: Passed to ``torch.utils.checkpoint``.

    Returns:
        *model* modified in-place.
    """
    manager = GradientCheckpointingManager(
        checkpoint_interval=checkpoint_interval,
        use_reentrant=use_reentrant,
    )
    return manager.apply_to_model(model)


def remove_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """Best-effort removal of checkpointing applied via this module.

    Note: Only works for wrapping done through ``GradientCheckpointingManager``
    or ``apply_gradient_checkpointing``. HuggingFace's
    ``model.gradient_checkpointing_enable()`` manages its own state.
    """
    # Fallback path: look for _original_forward attr (legacy compat)
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            delattr(module, "_original_forward")
    return model
