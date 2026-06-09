import copy
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import (
    LEDForConditionalGeneration,
    LEDConfig,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from models.section_embedding import SectionDetector, SectionAwareEmbedding, NUM_SECTION_TYPES
from models.faithfulness_gate import FaithfulnessGatedDecoderLayer, FaithfulnessGate
from models.contrastive_loss import ContrastiveFactualityLoss, SummaryPerturbator


@dataclass
class LEDFaCTConfig:
    use_sae: bool = True
    use_fgca: bool = True
    use_cfl: bool = True
    section_embed_dim: int = 64
    fgca_hidden_dim: int = 256
    cfl_projection_dim: int = 128
    cfl_temperature: float = 0.07
    cfl_alpha: float = 0.1
    dropout: float = 0.1
    base_model_name: str = "allenai/led-base-16384"
    max_input_length: int = 16384
    max_target_length: int = 256

    @property
    def config_name(self):
        parts = []
        if self.use_sae:
            parts.append("sae")
        if self.use_fgca:
            parts.append("fgca")
        if self.use_cfl:
            parts.append("cfl")
        if not parts:
            return "led_baseline"
        return "led_fact_" + "_".join(parts)


ABLATION_CONFIGS = {
    "led_baseline": LEDFaCTConfig(use_sae=False, use_fgca=False, use_cfl=False),
    "led_fact_no_sae": LEDFaCTConfig(use_sae=False, use_fgca=True, use_cfl=True),
    "led_fact_no_fgca": LEDFaCTConfig(use_sae=True, use_fgca=False, use_cfl=True),
    "led_fact_no_cfl": LEDFaCTConfig(use_sae=True, use_fgca=True, use_cfl=False),
    "led_fact_full": LEDFaCTConfig(use_sae=True, use_fgca=True, use_cfl=True),
}


class LEDFaCTForConditionalGeneration(nn.Module):
    def __init__(self, config: LEDFaCTConfig = None):
        super().__init__()
        if config is None:
            config = LEDFaCTConfig()
        self.config = config
        self.led_config = LEDConfig.from_pretrained(config.base_model_name)

        self.led = LEDForConditionalGeneration.from_pretrained(config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.tokenizer.model_max_length = config.max_input_length

        hidden_size = self.led_config.d_model

        self.section_detector = SectionDetector()

        self.use_sae = config.use_sae
        self.use_fgca = config.use_fgca
        self.use_cfl = config.use_cfl

        if self.use_sae:
            self.section_embedding = SectionAwareEmbedding(
                hidden_size=hidden_size,
                num_section_types=NUM_SECTION_TYPES,
                section_embed_dim=config.section_embed_dim,
                dropout=config.dropout,
            )

        if self.use_fgca:
            self._inject_fgca_layers(hidden_size, config.fgca_hidden_dim, config.dropout)

        if self.use_cfl:
            self.cfl_loss = ContrastiveFactualityLoss(
                hidden_size=hidden_size,
                projection_dim=config.cfl_projection_dim,
                temperature=config.cfl_temperature,
                alpha=config.cfl_alpha,
            )

        self.cfl_alpha = config.cfl_alpha
        self.max_input_length = config.max_input_length

        self._setup_led_for_long_context()

    def _setup_led_for_long_context(self):
        num_layers = self.led.config.num_hidden_layers
        self.led.config.attention_window = [1024] * num_layers
        self.led.config.attention_mode = "sliding_chunks"
        self.led.config.max_length = self.config.max_target_length
        self.led.config.eos_token_id = self.tokenizer.eos_token_id
        self.led.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id

    def _inject_fgca_layers(self, hidden_size: int, gate_hidden_dim: int, dropout: float):
        for i, decoder_layer in enumerate(self.led.base_model.decoder.layers):
            gated_layer = FaithfulnessGatedDecoderLayer(
                original_decoder_layer=decoder_layer,
                hidden_size=hidden_size,
                gate_hidden_dim=gate_hidden_dim,
                dropout=dropout,
            )
            self.led.base_model.decoder.layers[i] = gated_layer

    def _prepare_section_ids(self, input_texts: List[str]) -> torch.Tensor:
        section_ids = self.section_detector.batch_text_to_section_ids(
            texts=input_texts,
            tokenizer=self.tokenizer,
            max_length=self.max_input_length,
        )
        return section_ids.to(self.led.device)

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
        section_ids: Optional[torch.LongTensor] = None,
        perturbed_labels: Optional[torch.LongTensor] = None,
        input_texts: Optional[List[str]] = None,
    ):
        if self.use_sae:
            embed_grad_orig = self.led.base_model.encoder.embed_tokens.weight.requires_grad
            self.led.base_model.encoder.embed_tokens.requires_grad_(True)

            input_embeds = self.led.base_model.encoder.embed_tokens(input_ids)

            if section_ids is None:
                if input_texts is not None:
                    section_ids = self._prepare_section_ids(input_texts)
                else:
                    section_ids = torch.zeros(
                        input_ids.shape[0], input_ids.shape[1],
                        dtype=torch.long, device=input_ids.device
                    )

            input_embeds = self.section_embedding(input_embeds, section_ids)

            encoder = self.led.base_model.encoder
            from transformers.models.led.modeling_led import LEDEncoder
            encoder_out = encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            encoder_hidden_states = encoder_out.last_hidden_state if hasattr(encoder_out, 'last_hidden_state') else encoder_out[0]

            if not embed_grad_orig:
                self.led.base_model.encoder.embed_tokens.requires_grad_(False)
        else:
            encoder_hidden_states = None
            encoder_out = None

        if self.use_cfl and labels is not None and self.training:
            outputs = self._forward_with_cfl(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                labels=labels,
                perturbed_labels=perturbed_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            if encoder_hidden_states is not None:
                outputs = self.led(
                    input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_outputs=(encoder_hidden_states,),
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                outputs = self.led(
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

    def _forward_with_cfl(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_hidden_states,
        labels,
        perturbed_labels,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        if encoder_hidden_states is not None:
            outputs = self.led(
                input_ids=None,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=(encoder_hidden_states,),
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = self.led(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )

        ce_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)

        if outputs.decoder_hidden_states is not None:
            decoder_hidden = outputs.decoder_hidden_states[-1]
            cfl_loss, cfl_metrics = self.cfl_loss(
                decoder_hidden_states=decoder_hidden,
                labels=labels,
                perturbed_labels=perturbed_labels,
            )

            total_loss = ce_loss + self.cfl_alpha * cfl_loss

            outputs.loss = total_loss
            if hasattr(outputs, '__dict__'):
                outputs.__dict__['cfl_loss'] = cfl_loss.item()
                outputs.__dict__['ce_loss'] = ce_loss.item()
                outputs.__dict__['total_loss'] = total_loss.item()
                for k, v in cfl_metrics.items():
                    outputs.__dict__[f'cfl_{k}'] = v

        return outputs

    def generate(self, input_ids, attention_mask=None, section_ids=None, input_texts=None, **generate_kwargs):
        was_training = self.training
        self.eval()

        if self.use_sae:
            embed_grad_orig = self.led.base_model.encoder.embed_tokens.weight.requires_grad
            self.led.base_model.encoder.embed_tokens.requires_grad_(True)

            input_embeds = self.led.base_model.encoder.embed_tokens(input_ids)

            if section_ids is None:
                if input_texts is not None:
                    section_ids = self._prepare_section_ids(input_texts)
                else:
                    section_ids = torch.zeros(
                        input_ids.shape[0], input_ids.shape[1],
                        dtype=torch.long, device=input_ids.device
                    )

            input_embeds = self.section_embedding(input_embeds, section_ids)

            encoder = self.led.base_model.encoder
            encoder_out = encoder(
                attention_mask=attention_mask,
                inputs_embeds=input_embeds,
                return_dict=True,
            )
            encoder_hidden_states = encoder_out.last_hidden_state if hasattr(encoder_out, 'last_hidden_state') else encoder_out[0]

            if not embed_grad_orig:
                self.led.base_model.encoder.embed_tokens.requires_grad_(False)

            outputs = self.led.generate(
                input_ids=None,
                attention_mask=attention_mask,
                encoder_outputs=(encoder_hidden_states,),
                **generate_kwargs,
            )
        else:
            outputs = self.led.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        if was_training:
            self.train()

        return outputs

    def save_pretrained(self, save_directory: str):
        import os
        os.makedirs(save_directory, exist_ok=True)

        import json
        config_dict = {
            "use_sae": self.config.use_sae,
            "use_fgca": self.config.use_fgca,
            "use_cfl": self.config.use_cfl,
            "section_embed_dim": self.config.section_embed_dim,
            "fgca_hidden_dim": self.config.fgca_hidden_dim,
            "cfl_projection_dim": self.config.cfl_projection_dim,
            "cfl_temperature": self.config.cfl_temperature,
            "cfl_alpha": self.config.cfl_alpha,
            "base_model_name": self.config.base_model_name,
            "max_input_length": self.config.max_input_length,
            "max_target_length": self.config.max_target_length,
        }
        with open(os.path.join(save_directory, "led_fact_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        led_path = os.path.join(save_directory, "led_base")
        self.led.save_pretrained(led_path)
        self.tokenizer.save_pretrained(led_path)

        if self.use_sae:
            torch.save(self.section_embedding.state_dict(),
                       os.path.join(save_directory, "section_embedding.pt"))
        if self.use_fgca:
            fgca_state = {}
            for i, layer in enumerate(self.led.base_model.decoder.layers):
                if isinstance(layer, FaithfulnessGatedDecoderLayer):
                    fgca_state[f"layer_{i}"] = layer.faithfulness_gate.state_dict()
            torch.save(fgca_state, os.path.join(save_directory, "fgca_gates.pt"))
        if self.use_cfl:
            torch.save(self.cfl_loss.state_dict(),
                       os.path.join(save_directory, "cfl_loss.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, config_override: LEDFaCTConfig = None):
        import os, json

        config_path = os.path.join(load_directory, "led_fact_config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = LEDFaCTConfig(**config_dict)
        if config_override is not None:
            config.use_sae = config_override.use_sae
            config.use_fgca = config_override.use_fgca
            config.use_cfl = config_override.use_cfl

        model = cls(config)

        model_path = os.path.join(load_directory, "led_base")
        if os.path.exists(model_path):
            model.led = LEDForConditionalGeneration.from_pretrained(model_path)
            model.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if model.use_sae:
            sae_path = os.path.join(load_directory, "section_embedding.pt")
            if os.path.exists(sae_path):
                model.section_embedding.load_state_dict(torch.load(sae_path, map_location="cpu"))

        if model.use_fgca:
            fgca_path = os.path.join(load_directory, "fgca_gates.pt")
            if os.path.exists(fgca_path):
                fgca_state = torch.load(fgca_path, map_location="cpu")
                model._inject_fgca_layers(
                    model.led.config.d_model,
                    config.fgca_hidden_dim,
                    config.dropout,
                )
                for i, layer in enumerate(model.led.base_model.decoder.layers):
                    if isinstance(layer, FaithfulnessGatedDecoderLayer):
                        if f"layer_{i}" in fgca_state:
                            layer.faithfulness_gate.load_state_dict(fgca_state[f"layer_{i}"])

        if model.use_cfl:
            cfl_path = os.path.join(load_directory, "cfl_loss.pt")
            if os.path.exists(cfl_path):
                model.cfl_loss.load_state_dict(torch.load(cfl_path, map_location="cpu"))

        return model

    def get_trainable_params_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        module_params = {"total": total, "trainable": trainable}
        if self.use_sae:
            module_params["sae"] = sum(p.numel() for p in self.section_embedding.parameters())
        if self.use_fgca:
            fgca_params = 0
            for layer in self.led.base_model.decoder.layers:
                if isinstance(layer, FaithfulnessGatedDecoderLayer):
                    fgca_params += sum(p.numel() for p in layer.faithfulness_gate.parameters())
            module_params["fgca"] = fgca_params
        if self.use_cfl:
            module_params["cfl"] = sum(p.numel() for p in self.cfl_loss.parameters())

        module_params["led_base"] = sum(p.numel() for p in self.led.parameters())
        return module_params