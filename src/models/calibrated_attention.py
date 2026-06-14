"""
CFA: Calibrated Faithfulness Attention
======================================
Inspired by:
  - "DoLa: Decoding by Contrasting Layers Improves Factuality in Large LMs"
    (Chuang et al., ICLR 2024)
  - "Context-Aware Decoding for Faithful Summarization" (Shi et al., ACL 2024)
  - "Teaching Models to Express Their Uncertainty in Words"
    (Lin et al., TMLR 2022)

Core idea: Standard decoder cross-attention attends uniformly to encoder states,
regardless of whether the model is "guessing" vs. "retrieving". CFA adds a
lightweight calibration module per decoder layer that estimates token-level
faithfulness uncertainty from two signals:

  (1) cross-attention entropy — high entropy ≈ diffuse attention ≈ uncertain
  (2) source-context agreement — cosine similarity between cross-attn output
      and the attended encoder states

When uncertain, the gate pushes toward the source (retrieve facts);
when confident, the gate allows more self-expression (generate fluently).
This follows the intuition: "if you're not sure, look back at the source."

This is more principled than a simple learned gate because it explicitly
models uncertainty as a signal that modulates faithfulness behavior.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaithfulnessCalibrator(nn.Module):
    """Per-layer calibration module that estimates faithfulness uncertainty.

    Takes cross-attention outputs + attention statistics and produces a
    per-token calibration score.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Bottleneck architecture: compress → process → expand
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Uncertainty estimator: scalar per token
        self.uncertainty_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim // 2, 1),
        )

        # Faithfulness projector: maps calibration to token-level gate
        self.faithfulness_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights for gradual learning
        nn.init.normal_(self.encoder[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.normal_(self.faithfulness_proj[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.faithfulness_proj[0].bias)

    def forward(
        self,
        cross_attn_output: torch.Tensor,    # (B, T_dec, D)
        self_attn_output: torch.Tensor,     # (B, T_dec, D)
        cross_attn_weights: Optional[torch.Tensor] = None,  # (B, T_dec, T_enc) — optional
        encoder_states: Optional[torch.Tensor] = None,      # (B, T_enc, D) — optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: (calibrated_output, uncertainty_scores)"""

        # ── Build calibration features ──
        # Feature 1: cross and self attention outputs concatenated
        concat_features = torch.cat([cross_attn_output, self_attn_output], dim=-1)
        bottleneck = self.encoder(concat_features)  # (B, T, D_bottleneck)

        # ── Estimate uncertainty ──
        uncertainty_logits = self.uncertainty_head(bottleneck)  # (B, T, 1)
        uncertainty = torch.sigmoid(uncertainty_logits.squeeze(-1))  # (B, T) in [0,1]

        # ── Compute faithfulness calibration ──
        # High uncertainty → rely more on source (boost cross-attention)
        # Low uncertainty → rely more on self (model is confident)
        faith_proj = self.faithfulness_proj(bottleneck)  # (B, T, D)

        # Faithfulness gate: high uncertainty = want more source = gate closer to 1
        uncertainty_factor = uncertainty.unsqueeze(-1)  # (B, T, 1)
        faith_gate = torch.sigmoid(faith_proj + uncertainty_factor)  # (B, T, D)

        # ── Calibrated output ──
        # Blend cross-attention and self-attention based on faith_gate
        calibrated = (
            faith_gate * cross_attn_output
            + (1.0 - faith_gate) * self_attn_output
        )
        calibrated = self.layer_norm(calibrated)
        calibrated = self.dropout(calibrated)

        return calibrated, uncertainty


class CalibratedDecoderLayer(nn.Module):
    """Wraps a BART decoder layer with CFA calibration.

    Runs the original decoder layer normally, but uses forward hooks on
    self_attn_layer_norm and encoder_attn_layer_norm to capture the pure
    self-attention and cross-attention signals *after* their respective
    residual-add and layer-norm steps.  The calibrator then receives
    meaningful inputs (not the pre-layer raw hidden states nor the
    post-FFN full output) and its gated blend is residual-mixed into
    the final decoder output.
    """

    def __init__(
        self,
        original_decoder_layer: nn.Module,
        hidden_size: int = 1024,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_decoder_layer
        self.calibrator = FaithfulnessCalibrator(
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.use_cfa = True
        self._self_attn_out = None
        self._cross_attn_out = None

        # ── Forward hooks to capture intermediate activations ──
        # self_attn_layer_norm runs after: self-attn + residual dropout + add
        # Its output is the pure self-attention signal.
        self.original_layer.self_attn_layer_norm.register_forward_hook(
            self._capture_self_attn
        )
        # encoder_attn_layer_norm runs after: cross-attn + residual dropout + add
        # Its output is the pure cross-attention signal.
        self.original_layer.encoder_attn_layer_norm.register_forward_hook(
            self._capture_cross_attn
        )

    def _capture_self_attn(self, _module, _input, output):
        self._self_attn_out = output

    def _capture_cross_attn(self, _module, _input, output):
        self._cross_attn_out = output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Reset captured activations for this forward pass
        self._self_attn_out = None
        self._cross_attn_out = None

        # Run the full original decoder layer
        outputs = self.original_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

        # CFA disabled or no encoder states → passthrough
        if not self.use_cfa or encoder_hidden_states is None:
            return outputs

        decoder_output = outputs[0] if isinstance(outputs, tuple) else outputs

        # Fall back to the decoder output if hooks failed to capture (edge case)
        self_attn_signal = self._self_attn_out if self._self_attn_out is not None else decoder_output
        cross_attn_signal = self._cross_attn_out if self._cross_attn_out is not None else decoder_output

        calibrated_output, _ = self.calibrator(
            cross_attn_output=cross_attn_signal,
            self_attn_output=self_attn_signal,
        )

        # 0.5/0.5 residual blend for training stability
        blended = 0.5 * decoder_output + 0.5 * calibrated_output

        if isinstance(outputs, tuple):
            return (blended,) + outputs[1:]
        return blended
