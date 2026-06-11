import os
import json
import gc
import logging
from datetime import datetime
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    LEDForConditionalGeneration,
)

from config import (
    ModelConfig, TrainingConfig, MODEL_CONFIGS, get_model_config, get_device,
    get_led_fact_config,
)
from data_utils import (
    load_arxiv_dataset, load_pubmed_dataset,
    prepare_dataset_for_model, prepare_dataset_for_led_fact, set_seed,
)
from models.led_fact import LEDFaCTForConditionalGeneration, LEDFaCTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LEDFaCTTrainer(Seq2SeqTrainer):
    def __init__(self, *args, led_fact_model=None, use_cfl=False, cfl_alpha=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.led_fact_model = led_fact_model
        self.use_cfl = use_cfl
        self.cfl_alpha = cfl_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        is_led_fact = self.led_fact_model is not None

        if is_led_fact:
            section_ids = inputs.pop("section_ids", None)
            input_texts = inputs.pop("input_texts", None)
            labels = inputs.get("labels")

            outputs = self.led_fact_model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                decoder_input_ids=inputs.get("decoder_input_ids"),
                labels=labels,
                section_ids=section_ids,
                input_texts=input_texts,
                output_hidden_states=True,
                return_dict=True,
            )

            ce_loss = outputs.loss

            if self.use_cfl and labels is not None:
                decoder_hidden = outputs.decoder_hidden_states[-1] if outputs.decoder_hidden_states else None
                if decoder_hidden is not None:
                    perturbator = self.led_fact_model.cfl_loss.perturbator
                    tokenizer = self.led_fact_model.tokenizer
                    valid_mask = labels != -100
                    if valid_mask.any():
                        valid_labels = labels.clone()
                        valid_labels[valid_labels == -100] = tokenizer.pad_token_id

                        decoded_texts = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
                        perturbed_texts = perturbator.perturb_batch(decoded_texts, strategy="mixed")
                        perturbed_labels = tokenizer(
                            perturbed_texts,
                            max_length=valid_labels.shape[1],
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        )["input_ids"].to(labels.device)

                        perturbed_decoder_input_ids = self.led_fact_model.led.prepare_decoder_input_ids_from_labels(perturbed_labels)

                        with torch.no_grad():
                            neg_outputs = self.led_fact_model(
                                input_ids=inputs.get("input_ids"),
                                attention_mask=inputs.get("attention_mask"),
                                decoder_input_ids=perturbed_decoder_input_ids,
                                section_ids=section_ids,
                                input_texts=input_texts,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                        neg_decoder_hidden = neg_outputs.decoder_hidden_states[-1].detach() if neg_outputs.decoder_hidden_states else decoder_hidden.detach()
                        del neg_outputs, perturbed_decoder_input_ids, perturbed_labels
                        gc.collect()
                        torch.cuda.empty_cache()

                        cfl_loss, cfl_metrics = self.led_fact_model.cfl_loss(
                            decoder_hidden_states=decoder_hidden,
                            labels=labels,
                            neg_decoder_hidden_states=neg_decoder_hidden,
                        )

                        total_loss = ce_loss + self.cfl_alpha * cfl_loss

                        del neg_decoder_hidden
                        gc.collect()
                        torch.cuda.empty_cache()

                        self.log({
                            "ce_loss": ce_loss.item(),
                            "cfl_loss": cfl_loss.item(),
                            "total_loss": total_loss.item(),
                        })

                        return (total_loss, outputs) if return_outputs else total_loss

            return (ce_loss, outputs) if return_outputs else ce_loss

        inputs.pop("section_ids", None)
        inputs.pop("input_texts", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch, **kwargs)


def load_model_and_tokenizer(model_config: ModelConfig, device=None):
    if device is None:
        device = get_device()

    logger.info(f"Loading model: {model_config.name} from {model_config.hf_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_path)

    if model_config.is_led:
        model = LEDForConditionalGeneration.from_pretrained(
            model_config.hf_path,
        )
        tokenizer.model_max_length = model_config.max_input_length

        if model_config.max_input_length > 1024:
            n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 12
            model.config.attention_window = [
                min(1024, model_config.max_input_length // 2)
            ] * n_layers

        model.config.max_length = model_config.max_target_length
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_config.hf_path,
        )

    model = model.to(device)
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


def load_led_fact_model(model_config: ModelConfig, device=None, led_fact_config_override=None):
    if device is None:
        device = get_device()

    if led_fact_config_override is not None:
        led_fact_config = led_fact_config_override
    else:
        led_fact_config = get_led_fact_config(model_config.name)
    led_fact_config.max_input_length = model_config.max_input_length
    led_fact_config.max_target_length = model_config.max_target_length

    logger.info(f"Loading LED-FaCT model: {model_config.name}")
    logger.info(f"  SAE: {led_fact_config.use_sae}, FGCA: {led_fact_config.use_fgca}, CFL: {led_fact_config.use_cfl}")

    model = LEDFaCTForConditionalGeneration(led_fact_config)
    model = model.to(device)

    param_info = model.get_trainable_params_summary()
    logger.info(f"Model parameters: {param_info}")

    return model, model.tokenizer


def train_model(
    model_name: str,
    dataset_name: str = "arxiv",
    max_samples: int = None,
    training_config: TrainingConfig = None,
    max_input_length: int = None,
    led_fact_config_override=None,
):
    set_seed(training_config.seed if training_config else 42)

    model_config = get_model_config(model_name)
    if max_input_length is not None:
        model_config.max_input_length = min(max_input_length, model_config.max_input_length)

    if training_config is None:
        training_config = TrainingConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
        )

    output_dir = os.path.join(
        training_config.output_dir,
        f"{model_name}_{dataset_name}_ctx{model_config.max_input_length}",
    )
    os.makedirs(output_dir, exist_ok=True)

    is_led_fact = model_config.is_led_fact

    if is_led_fact:
        model, tokenizer = load_led_fact_model(model_config, led_fact_config_override=led_fact_config_override)
    else:
        model, tokenizer = load_model_and_tokenizer(model_config)

    if model_config.is_led and not is_led_fact:
        model.config.attention_mode = "sliding_chunks"
        model.config.attention_window = [1024] * model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else [1024] * 12
        model.config.max_source_positions = model_config.max_input_length
        model.config.max_target_positions = 256
        tokenizer.model_max_length = model_config.max_input_length

    if is_led_fact:
        dataset = prepare_dataset_for_led_fact(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_input_length=model_config.max_input_length,
            max_target_length=model_config.max_target_length,
            max_samples=max_samples,
        )
    else:
        dataset = prepare_dataset_for_model(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_input_length=model_config.max_input_length,
            max_target_length=model_config.max_target_length,
            max_samples=max_samples,
            is_led=model_config.is_led,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.led if is_led_fact else model,
        padding=True,
    )

    led_fact_model = model if is_led_fact else None
    cfl_enabled = False
    cfl_alpha = 0.1
    if is_led_fact and led_fact_model is not None:
        cfl_enabled = led_fact_model.config.use_cfl
        cfl_alpha = led_fact_model.config.cfl_alpha

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_train_epochs=training_config.num_train_epochs,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        adam_epsilon=training_config.adam_epsilon,
        max_grad_norm=training_config.max_grad_norm,
        fp16=training_config.fp16 and get_device().type == "cuda",
        logging_steps=training_config.logging_steps,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=training_config.eval_steps if "validation" in dataset else None,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        predict_with_generate=True,
        generation_max_length=model_config.max_target_length,
        report_to=["tensorboard"],
        load_best_model_at_end=True if "validation" in dataset else False,
        metric_for_best_model="eval_loss" if "validation" in dataset else None,
        seed=training_config.seed,
        dataloader_num_workers=0 if get_device().type == "cpu" else 4,
        dataloader_pin_memory=get_device().type == "cuda",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    if is_led_fact:
        trainer_kwargs.update(
            led_fact_model=led_fact_model,
            use_cfl=cfl_enabled,
            cfl_alpha=cfl_alpha,
        )
    trainer = LEDFaCTTrainer(**trainer_kwargs) if is_led_fact else Seq2SeqTrainer(**trainer_kwargs)

    logger.info(f"Starting training: {model_name} on {dataset_name} with ctx={model_config.max_input_length}")
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)

    if is_led_fact and led_fact_model is not None:
        led_fact_model.save_pretrained(output_dir)
        logger.info(f"LED-FaCT model saved to {output_dir}")
    else:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    logger.info(f"Training complete. Metrics: {metrics}")
    return trainer, model, tokenizer


def train_multiple_context_lengths(
    model_name: str,
    context_lengths: list = None,
    dataset_name: str = "arxiv",
    max_samples: int = None,
    base_config: TrainingConfig = None,
):
    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096, 8192]

    model_config = get_model_config(model_name)
    valid_lengths = [cl for cl in context_lengths if cl <= model_config.max_input_length]

    results = {}
    for ctx_len in valid_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} with context length {ctx_len}")
        logger.info(f"{'='*60}\n")

        config = base_config or TrainingConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
        )

        try:
            trainer, model, tokenizer = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                training_config=config,
                max_input_length=ctx_len,
            )
            results[ctx_len] = {"status": "success"}
        except Exception as e:
            logger.error(f"Failed at context length {ctx_len}: {e}")
            results[ctx_len] = {"status": "failed", "error": str(e)}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train summarization models")
    parser.add_argument("--model", type=str, default="bart-large-cnn", help="Model name")
    parser.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_input_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--context_lengths", type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        dataset_name=args.dataset,
        model_name=args.model,
        max_samples=args.max_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if args.context_lengths:
        ctx_lengths = [int(x) for x in args.context_lengths.split(",")]
        results = train_multiple_context_lengths(
            model_name=args.model,
            context_lengths=ctx_lengths,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            base_config=config,
        )
        print(json.dumps({k: v.get("status", "unknown") for k, v in results.items()}, indent=2))
    else:
        trainer, model, tokenizer = train_model(
            model_name=args.model,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            training_config=config,
            max_input_length=args.max_input_length,
        )