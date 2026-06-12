import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaithfulnessGate(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        gate_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_hidden_dim = gate_hidden_dim

        self.gate_proj = nn.Linear(hidden_size * 2, gate_hidden_dim)
        self.gate_out = nn.Linear(gate_hidden_dim, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.xavier_uniform_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def forward(
        self,
        cross_attn_output: torch.Tensor,
        self_attn_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([cross_attn_output, self_attn_output], dim=-1)
        gate_hidden = F.relu(self.gate_proj(concat))
        gate_hidden = self.dropout(gate_hidden)
        gate_values = torch.sigmoid(self.gate_out(gate_hidden))

        gated_output = gate_values * cross_attn_output + (1 - gate_values) * self_attn_output
        gated_output = self.layer_norm(gated_output)

        return gated_output, gate_values


class FaithfulnessGatedDecoderLayer(nn.Module):
    def __init__(
        self,
        original_decoder_layer: nn.Module,
        hidden_size: int = 768,
        gate_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_decoder_layer
        self.faithfulness_gate = FaithfulnessGate(
            hidden_size=hidden_size,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )
        self.use_fgca = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self_attn_output_before_cross = hidden_states

        layer_outputs = self.original_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

        if isinstance(layer_outputs, tuple):
            decoder_output = layer_outputs[0]
        else:
            decoder_output = layer_outputs

        if not self.use_fgca or encoder_hidden_states is None:
            return layer_outputs

        cross_attn_output = decoder_output
        self_attn_residual = self_attn_output_before_cross

        gated_output, gate_values = self.faithfulness_gate(
            cross_attn_output, self_attn_residual
        )

        hybrid_output = 0.5 * decoder_output + 0.5 * gated_output

        if isinstance(layer_outputs, tuple):
            return (hybrid_output,) + layer_outputs[1:]
        else:
            return hybrid_output