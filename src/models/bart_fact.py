import copy
import json
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    AutoTokenizer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from models.hierarchical_structure import (
    HierarchicalStructureEncoder,
    batch_detect_boundaries,
)
from models.calibrated_attention import CalibratedDecoderLayer
from models.preference_loss import (
    ContrastivePreferenceLoss,
    generate_context_free_summary,
)


@dataclass
class BARTFaCTConfig:
    """Configuration for BART-FaCT with HSE + CFA + CPO modules."""

    use_hse: bool = True
    use_cfa: bool = True
    use_cpo: bool = True

    hse_num_heads: int = 4
    hse_ffn_dim: int = 256
    hse_dropout: float = 0.1

    cfa_bottleneck_dim: int = 128
    cfa_dropout: float = 0.1

    cpo_projection_dim: int = 128
    cpo_temperature: float = 0.1
    cpo_beta: float = 0.5
    cpo_alpha: float = 0.15

    dropout: float = 0.1
    base_model_name: str = "facebook/bart-large-cnn"
    max_input_length: int = 1024
    max_target_length: int = 256
    is_encoder_decoder: bool = True  # Required by Trainer internals

    @property
    def config_name(self):
        parts = []
        if self.use_hse:
            parts.append("hse")
        if self.use_cfa:
            parts.append("cfa")
        if self.use_cpo:
            parts.append("cpo")
        if not parts:
            return "bart_baseline"
        return "bart_fact_" + "_".join(parts)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "BARTFaCTConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in config_dict.items() if k in valid_keys})


# ── Ablation presets ───────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "bart_baseline": BARTFaCTConfig(
        use_hse=False, use_cfa=False, use_cpo=False,
    ),
    "bart_fact_no_hse": BARTFaCTConfig(
        use_hse=False, use_cfa=True, use_cpo=True,
    ),
    "bart_fact_no_cfa": BARTFaCTConfig(
        use_hse=True, use_cfa=False, use_cpo=True,
    ),
    "bart_fact_no_cpo": BARTFaCTConfig(
        use_hse=True, use_cfa=True, use_cpo=False,
    ),
    "bart_fact_full": BARTFaCTConfig(
        use_hse=True, use_cfa=True, use_cpo=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# BART-FaCT Model
# ═══════════════════════════════════════════════════════════════════════

class BARTFaCTForConditionalGeneration(nn.Module):
    """BART-based faithfulness-enhanced summarization with three modules:

    HSE  — Hierarchical Structure Encoding
           Multi-granularity (token→sentence→section) structure injection
    CFA  — Calibrated Faithfulness Attention
           Uncertainty-calibrated cross-attention gating per decoder layer
    CPO  — Contrastive Preference Optimization
           DPO-style preference loss using reference vs. context-free summaries
    """

    def __init__(self, config: BARTFaCTConfig = None):
        super().__init__()
        if config is None:
            config = BARTFaCTConfig()
        self.config = config
        self.bart_config = BartConfig.from_pretrained(config.base_model_name)

        self.bart = BartForConditionalGeneration.from_pretrained(config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.tokenizer.model_max_length = config.max_input_length

        hidden_size = self.bart_config.d_model  # 1024 for bart-large

        # ── Module switches ──
        self.use_hse = config.use_hse
        self.use_cfa = config.use_cfa
        self.use_cpo = config.use_cpo

        # ── HSE: Hierarchical Structure Encoding ──
        if self.use_hse:
            self.hse_encoder = HierarchicalStructureEncoder(
                hidden_size=hidden_size,
                num_heads=config.hse_num_heads,
                ffn_dim=config.hse_ffn_dim,
                dropout=config.hse_dropout,
            )

        # ── CFA: Calibrated Faithfulness Attention ──
        if self.use_cfa:
            self._inject_cfa_layers(
                hidden_size,
                config.cfa_bottleneck_dim,
                config.cfa_dropout,
            )

        # ── CPO: Contrastive Preference Optimization ──
        if self.use_cpo:
            self.cpo_loss = ContrastivePreferenceLoss(
                hidden_size=hidden_size,
                projection_dim=config.cpo_projection_dim,
                temperature=config.cpo_temperature,
                beta=config.cpo_beta,
                alpha=config.cpo_alpha,
            )

        self.cpo_alpha = config.cpo_alpha
        self.max_input_length = config.max_input_length

        self._setup_bart_generation()

    # ── Property passthroughs ──────────────────────────────────────────

    @property
    def generation_config(self):
        return self.bart.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.bart.generation_config = value

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_parameters(self):
        """Passthrough for Trainer internals that access model.num_parameters."""
        return sum(p.numel() for p in self.parameters())

    def prepare_decoder_input_ids_from_labels(self, labels):
        """Passthrough for Trainer predict_with_generate path."""
        return self.bart.prepare_decoder_input_ids_from_labels(labels)

    def can_generate(self) -> bool:
        return True

    def gradient_checkpointing_enable(self, gck_kwargs=None):
        self.bart.gradient_checkpointing_enable(gck_kwargs)

    def gradient_checkpointing_disable(self):
        self.bart.gradient_checkpointing_disable()

    def is_gradient_checkpointing(self):
        return getattr(self.bart, 'gradient_checkpointing', False)

    # ── Internal setup ─────────────────────────────────────────────────

    def _setup_bart_generation(self):
        self.bart.generation_config.max_length = self.config.max_target_length
        self.bart.config.eos_token_id = self.tokenizer.eos_token_id
        self.bart.config.decoder_start_token_id = (
            self.tokenizer.bos_token_id or self.tokenizer.cls_token_id or 0
        )

    def _inject_cfa_layers(self, hidden_size, bottleneck_dim, dropout):
        """Wrap each BART decoder layer with CFA calibration."""
        for i, decoder_layer in enumerate(self.bart.model.decoder.layers):
            calibrated_layer = CalibratedDecoderLayer(
                original_decoder_layer=decoder_layer,
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
            )
            self.bart.model.decoder.layers[i] = calibrated_layer

    def _prepare_boundary_mask(
        self, input_texts: List[str]
    ) -> torch.Tensor:
        return batch_detect_boundaries(
            texts=input_texts,
            tokenizer=self.tokenizer,
            max_length=self.max_input_length,
        ).to(self.bart.device)

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values: Optional[Tuple] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boundary_mask: Optional[torch.LongTensor] = None,
        input_texts: Optional[List[str]] = None,
    ):
        if self.use_hse:
            # ── HSE path: inject hierarchical structure into embeddings ──
            embed_grad_orig = self.bart.model.encoder.embed_tokens.weight.requires_grad
            self.bart.model.encoder.embed_tokens.requires_grad_(True)

            if input_ids is None:
                raise ValueError("input_ids is required when HSE is enabled")

            # Word embeddings (BART applies embed_scale)
            input_embeds = (
                self.bart.model.encoder.embed_tokens(input_ids)
                * self.bart.model.encoder.embed_scale
            )

            # Prepare boundary mask for sentence detection
            if boundary_mask is None:
                if input_texts is not None:
                    boundary_mask = self._prepare_boundary_mask(input_texts)
                else:
                    boundary_mask = torch.zeros(
                        input_ids.shape[0], input_ids.shape[1],
                        dtype=torch.long, device=input_ids.device,
                    )

            # Align boundary mask length
            if boundary_mask.shape[1] != input_ids.shape[1]:
                seq_len = input_ids.shape[1]
                if boundary_mask.shape[1] > seq_len:
                    boundary_mask = boundary_mask[:, :seq_len]
                else:
                    pad_len = seq_len - boundary_mask.shape[1]
                    boundary_mask = torch.nn.functional.pad(
                        boundary_mask, (0, pad_len), value=0,
                    )

            # Apply HSE: enrich embeddings with hierarchical structure
            input_embeds = self.hse_encoder(
                token_embeddings=input_embeds,
                boundary_mask=boundary_mask,
                attention_mask=attention_mask,
            )

            # Encode with structure-enriched embeddings
            encoder = self.bart.model.encoder
            encoder_out = encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            enc_hidden = (
                encoder_out.last_hidden_state
                if hasattr(encoder_out, 'last_hidden_state')
                else encoder_out[0]
            )

            if not embed_grad_orig:
                self.bart.model.encoder.embed_tokens.requires_grad_(False)

            encoder_outputs_obj = BaseModelOutput(last_hidden_state=enc_hidden)
            outputs = self.bart(
                input_ids=None,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs_obj,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # Standard BART forward
            outputs = self.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return outputs

    # ── Generate ───────────────────────────────────────────────────────

    def generate(
        self,
        input_ids,
        attention_mask=None,
        boundary_mask=None,
        input_texts=None,
        **generate_kwargs,
    ):
        was_training = self.training
        self.eval()
        if self.use_hse:
            generate_kwargs.setdefault("use_cache", False)

        if self.use_hse:
            embed_grad_orig = self.bart.model.encoder.embed_tokens.weight.requires_grad
            self.bart.model.encoder.embed_tokens.requires_grad_(True)

            input_embeds = (
                self.bart.model.encoder.embed_tokens(input_ids)
                * self.bart.model.encoder.embed_scale
            )

            if boundary_mask is None:
                if input_texts is not None:
                    boundary_mask = self._prepare_boundary_mask(input_texts)
                else:
                    boundary_mask = torch.zeros(
                        input_ids.shape[0], input_ids.shape[1],
                        dtype=torch.long, device=input_ids.device,
                    )

            if boundary_mask.shape[1] != input_ids.shape[1]:
                seq_len = input_ids.shape[1]
                if boundary_mask.shape[1] > seq_len:
                    boundary_mask = boundary_mask[:, :seq_len]
                else:
                    pad_len = seq_len - boundary_mask.shape[1]
                    boundary_mask = torch.nn.functional.pad(
                        boundary_mask, (0, pad_len), value=0,
                    )

            input_embeds = self.hse_encoder(
                token_embeddings=input_embeds,
                boundary_mask=boundary_mask,
                attention_mask=attention_mask,
            )

            encoder = self.bart.model.encoder
            encoder_out = encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
                return_dict=True,
            )
            enc_hidden = (
                encoder_out.last_hidden_state
                if hasattr(encoder_out, 'last_hidden_state')
                else encoder_out[0]
            )

            if not embed_grad_orig:
                self.bart.model.encoder.embed_tokens.requires_grad_(False)

            encoder_outputs_obj = BaseModelOutput(last_hidden_state=enc_hidden)
            outputs = self.bart.generate(
                input_ids=None,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs_obj,
                **generate_kwargs,
            )
        else:
            outputs = self.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        if was_training:
            self.train()

        return outputs

    # ── Save / Load ────────────────────────────────────────────────────

    def save_pretrained(self, save_directory: str, **kwargs):
        import os
        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, "bart_fact_config.json"), "w") as f:
            f.write(self.config.to_json_string())

        if self.use_hse:
            torch.save(
                self.hse_encoder.state_dict(),
                os.path.join(save_directory, "hse_encoder.pt"),
            )

        if self.use_cfa:
            cfa_state = {}
            original_cfa_layers = {}
            for i, layer in enumerate(self.bart.model.decoder.layers):
                if isinstance(layer, CalibratedDecoderLayer):
                    cfa_state[f"layer_{i}"] = layer.calibrator.state_dict()
                    original_cfa_layers[i] = layer
                    self.bart.model.decoder.layers[i] = layer.original_layer
            # Write CFA state to disk BEFORE saving BART (which needs unwrapped layers)
            torch.save(
                cfa_state,
                os.path.join(save_directory, "cfa_calibrators.pt"),
            )

        try:
            bart_path = os.path.join(save_directory, "bart_base")
            # safetensors avoids the "unexpected pos" pickle error that
            # occurs when decoder layers are swapped in-place during save.
            self.bart.save_pretrained(bart_path, safe_serialization=True)
            self.tokenizer.save_pretrained(bart_path)
        finally:
            if self.use_cfa:
                for i, calibrated_layer in original_cfa_layers.items():
                    self.bart.model.decoder.layers[i] = calibrated_layer

        if self.use_cpo:
            torch.save(
                self.cpo_loss.state_dict(),
                os.path.join(save_directory, "cpo_loss.pt"),
            )

    @classmethod
    def from_pretrained(
        cls, load_directory: str, config_override: BARTFaCTConfig = None
    ):
        import os

        config_path = os.path.join(load_directory, "bart_fact_config.json")
        with open(config_path, "r") as f:
            config = BARTFaCTConfig.from_dict(json.load(f))
        if config_override is not None:
            config.use_hse = config_override.use_hse
            config.use_cfa = config_override.use_cfa
            config.use_cpo = config_override.use_cpo

        model = cls(config)

        model_path = os.path.join(load_directory, "bart_base")
        if os.path.exists(model_path):
            loaded_bart = BartForConditionalGeneration.from_pretrained(model_path)

            if model.use_cfa:
                cfa_wrapped_layers = {}
                for i, layer in enumerate(model.bart.model.decoder.layers):
                    if isinstance(layer, CalibratedDecoderLayer):
                        cfa_wrapped_layers[i] = layer
                        model.bart.model.decoder.layers[i] = layer.original_layer

            model.bart.load_state_dict(loaded_bart.state_dict(), strict=False)
            del loaded_bart
            model.tokenizer = AutoTokenizer.from_pretrained(model_path)

            if model.use_cfa:
                for i, calib_layer in cfa_wrapped_layers.items():
                    model.bart.model.decoder.layers[i] = calib_layer

        if model.use_hse:
            hse_path = os.path.join(load_directory, "hse_encoder.pt")
            if os.path.exists(hse_path):
                model.hse_encoder.load_state_dict(
                    torch.load(hse_path, map_location="cpu", weights_only=True)
                )

        if model.use_cfa:
            cfa_path = os.path.join(load_directory, "cfa_calibrators.pt")
            if os.path.exists(cfa_path):
                cfa_state = torch.load(
                    cfa_path, map_location="cpu", weights_only=True
                )
                for i, layer in enumerate(model.bart.model.decoder.layers):
                    if isinstance(layer, CalibratedDecoderLayer):
                        key = f"layer_{i}"
                        if key in cfa_state:
                            layer.calibrator.load_state_dict(cfa_state[key])

        if model.use_cpo:
            cpo_path = os.path.join(load_directory, "cpo_loss.pt")
            if os.path.exists(cpo_path):
                model.cpo_loss.load_state_dict(
                    torch.load(cpo_path, map_location="cpu", weights_only=True)
                )

        return model

    def get_trainable_params_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        module_params = {"total": total, "trainable": trainable}
        if self.use_hse:
            module_params["hse"] = sum(
                p.numel() for p in self.hse_encoder.parameters()
            )
        if self.use_cfa:
            cfa_p = 0
            for layer in self.bart.model.decoder.layers:
                if isinstance(layer, CalibratedDecoderLayer):
                    cfa_p += sum(
                        p.numel() for p in layer.calibrator.parameters()
                    )
            module_params["cfa"] = cfa_p
        if self.use_cpo:
            module_params["cpo"] = sum(
                p.numel() for p in self.cpo_loss.parameters()
            )

        # Avoid double-counting: CFA calibrator weights are in self.bart.parameters()
        # because CalibratedDecoderLayer is injected into bart's decoder layers.
        bart_total = sum(p.numel() for p in self.bart.parameters())
        if self.use_cfa:
            module_params["bart_base"] = bart_total - module_params.get("cfa", 0)
        else:
            module_params["bart_base"] = bart_total
        return module_params
