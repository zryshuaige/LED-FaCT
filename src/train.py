import os
import json
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    BartForConditionalGeneration,
)

from config import (
    ModelConfig,
    TrainingConfig,
    MODEL_CONFIGS,
    get_model_config,
    get_device,
    get_bart_fact_config,
)
from data_utils import (
    load_arxiv_dataset,
    load_pubmed_dataset,
    prepare_dataset_for_model,
    prepare_dataset_for_bart_fact,
    set_seed,
)
from models.bart_fact import BARTFaCTForConditionalGeneration, BARTFaCTConfig
from models.preference_loss import generate_context_free_summary

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data collator for BART-FaCT
# ═══════════════════════════════════════════════════════════════════════


class BARTFaCTDataCollator(DataCollatorForSeq2Seq):
    """Extends Seq2Seq collator to handle boundary_mask and input_texts."""

    def __call__(self, features, return_tensors=None):
        input_texts_list = None
        boundary_mask_list = None

        if features and "input_texts" in features[0]:
            input_texts_list = [f.pop("input_texts") for f in features]

        if features and "boundary_mask" in features[0]:
            boundary_mask_list = [f.pop("boundary_mask") for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        if boundary_mask_list is not None:
            max_len = max(len(m) for m in boundary_mask_list)
            pad_val = 0
            padded = []
            for m in boundary_mask_list:
                if len(m) < max_len:
                    padded.append(m + [pad_val] * (max_len - len(m)))
                else:
                    padded.append(m[:max_len])
            batch["boundary_mask"] = torch.tensor(padded, dtype=torch.long)

        if input_texts_list is not None:
            batch["input_texts"] = input_texts_list

        return batch


# ═══════════════════════════════════════════════════════════════════════
# Custom trainer with CPO support
# ═══════════════════════════════════════════════════════════════════════


class BARTFaCTTrainer(Seq2SeqTrainer):
    """Trainer that supports the Contrastive Preference Optimization (CPO) loss."""

    def __init__(
        self,
        *args,
        bart_fact_model=None,
        use_cpo=False,
        cpo_alpha=0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bart_fact_model = bart_fact_model
        self.use_cpo = use_cpo
        self.cpo_alpha = cpo_alpha

    def save_model(self, output_dir=None, _internal_call=False):
        if self.bart_fact_model is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            self.bart_fact_model.save_pretrained(output_dir)
            self.bart_fact_model.tokenizer.save_pretrained(output_dir)
        else:
            super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def compute_loss(
        self, model, inputs, return_outputs=False, **kwargs
    ):
        is_bart_fact = self.bart_fact_model is not None

        if is_bart_fact:
            boundary_mask = inputs.pop("boundary_mask", None)
            input_texts = inputs.pop("input_texts", None)
            labels = inputs.get("labels")

            # Forward pass with HSE
            outputs = self.bart_fact_model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                decoder_input_ids=inputs.get("decoder_input_ids"),
                labels=labels,
                boundary_mask=boundary_mask,
                input_texts=input_texts,
                output_hidden_states=True,
                return_dict=True,
            )

            ce_loss = outputs.loss

            # ── CPO: Contrastive Preference Optimization ──
            if self.use_cpo and labels is not None:
                decoder_hidden = (
                    outputs.decoder_hidden_states[-1]
                    if outputs.decoder_hidden_states
                    else None
                )
                if decoder_hidden is not None:
                    tokenizer = self.bart_fact_model.tokenizer

                    # Generate context-free summaries as "dispreferred" responses
                    with torch.no_grad():
                        neg_ids = generate_context_free_summary(
                            model=self.bart_fact_model,
                            tokenizer=tokenizer,
                            num_tokens=min(128, labels.shape[1]),
                        )
                        # Expand single negative to match batch size
                        B = labels.shape[0]
                        if neg_ids.shape[0] == 1 and B > 1:
                            neg_ids = neg_ids.repeat(B, 1)

                        # Pad to match label length
                        if neg_ids.shape[1] < labels.shape[1]:
                            pad_len = labels.shape[1] - neg_ids.shape[1]
                            neg_ids = torch.nn.functional.pad(
                                neg_ids, (0, pad_len),
                                value=tokenizer.pad_token_id or 0,
                            )
                        else:
                            neg_ids = neg_ids[:, :labels.shape[1]]

                        neg_ids = neg_ids.to(labels.device)

                        # Get decoder hidden states for the negative (context-free) summary
                        neg_decoder_input_ids = (
                            self.bart_fact_model.bart.prepare_decoder_input_ids_from_labels(
                                neg_ids
                            )
                        )

                        neg_outputs = self.bart_fact_model(
                            input_ids=inputs.get("input_ids"),
                            attention_mask=inputs.get("attention_mask"),
                            decoder_input_ids=neg_decoder_input_ids,
                            boundary_mask=boundary_mask,
                            input_texts=input_texts,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    neg_hidden = (
                        neg_outputs.decoder_hidden_states[-1].detach()
                        if neg_outputs.decoder_hidden_states
                        else decoder_hidden.detach()
                    )
                    del neg_outputs, neg_decoder_input_ids, neg_ids
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Compute CPO loss
                    cpo_loss_val, cpo_metrics = self.bart_fact_model.cpo_loss(
                        pos_hidden_states=decoder_hidden,
                        neg_hidden_states=neg_hidden,
                    )

                    total_loss = ce_loss + self.cpo_alpha * cpo_loss_val

                    del neg_hidden
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    self.log(
                        {
                            "ce_loss": ce_loss.item(),
                            "cpo_loss": cpo_loss_val.item(),
                            "total_loss": total_loss.item(),
                            "cpo_margin": cpo_metrics.get("preference_margin", 0),
                            "cpo_accuracy": cpo_metrics.get("preference_accuracy", 0),
                        }
                    )

                    return (total_loss, outputs) if return_outputs else total_loss

            return (ce_loss, outputs) if return_outputs else ce_loss

        # Standard model path
        inputs.pop("boundary_mask", None)
        inputs.pop("input_texts", None)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════


def load_model_and_tokenizer(model_config: ModelConfig, device=None):
    """Load a standard HuggingFace seq2seq model."""
    if device is None:
        device = get_device()

    logger.info(f"Loading model: {model_config.name} from {model_config.hf_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_config.hf_path,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    logger.info(
        f"Model loaded. Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    return model, tokenizer


def load_bart_fact_model(
    model_config: ModelConfig, device=None, bart_fact_config_override=None
):
    """Load a BART-FaCT model (BART + HSE + CFA + CPO)."""
    if device is None:
        device = get_device()

    if bart_fact_config_override is not None:
        bart_fact_config = bart_fact_config_override
    else:
        bart_fact_config = get_bart_fact_config(model_config.name)
    bart_fact_config.max_input_length = model_config.max_input_length
    bart_fact_config.max_target_length = model_config.max_target_length

    logger.info(f"Loading BART-FaCT model: {model_config.name}")
    logger.info(
        f"  HSE: {bart_fact_config.use_hse}, "
        f"CFA: {bart_fact_config.use_cfa}, "
        f"CPO: {bart_fact_config.use_cpo}"
    )

    model = BARTFaCTForConditionalGeneration(bart_fact_config)
    model = model.to(device)

    param_info = model.get_trainable_params_summary()
    logger.info(f"Model parameters: {param_info}")

    return model, model.tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Main training entry point
# ═══════════════════════════════════════════════════════════════════════


def train_model(
    model_name: str,
    dataset_name: str = "arxiv",
    max_samples: int = None,
    training_config: TrainingConfig = None,
    max_input_length: int = None,
    bart_fact_config_override=None,
):
    """Train a single model (standard or BART-FaCT)."""
    set_seed(training_config.seed if training_config else 42)

    model_config = get_model_config(model_name)
    if max_input_length is not None:
        model_config.max_input_length = min(
            max_input_length, model_config.max_input_length
        )

    if training_config is None:
        training_config = TrainingConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
        )

    output_dir = os.path.normpath(
        os.path.join(
            training_config.output_dir,
            f"{model_name}_{dataset_name}_ctx{model_config.max_input_length}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    is_bart_fact = model_config.is_bart_fact

    # Load model
    if is_bart_fact:
        model, tokenizer = load_bart_fact_model(
            model_config, bart_fact_config_override=bart_fact_config_override
        )
    else:
        model, tokenizer = load_model_and_tokenizer(model_config)

    # Prepare dataset
    if is_bart_fact:
        dataset = prepare_dataset_for_bart_fact(
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
        )

    # Data collator
    if is_bart_fact:
        data_collator = BARTFaCTDataCollator(
            tokenizer=tokenizer,
            model=model.bart,
            padding=True,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )

    # CPO config
    bart_fact_model = model if is_bart_fact else None
    cpo_enabled = False
    cpo_alpha = 0.15
    if is_bart_fact and bart_fact_model is not None:
        cpo_enabled = bart_fact_model.config.use_cpo
        cpo_alpha = bart_fact_model.config.cpo_alpha

    # Build kwargs dict first so we can auto-detect eval_strategy vs evaluation_strategy
    eval_strategy_val = "steps" if "validation" in dataset else "no"
    has_validation = "validation" in dataset
    _eval_steps = max(training_config.eval_steps, 1) if has_validation else None
    _save_steps = training_config.save_steps
    _load_best = has_validation

    # Transformers (>=4.35-ish) requires save_steps to be a round multiple of
    # eval_steps when load_best_model_at_end=True.  Tweak to satisfy the check.
    if _load_best and _eval_steps is not None:
        if _save_steps < _eval_steps:
            _save_steps = _eval_steps
        elif _save_steps % _eval_steps != 0:
            _save_steps = ((_save_steps // _eval_steps) + 1) * _eval_steps

    # TensorBoard on Windows fails when the output path contains non-ASCII
    # characters (e.g. Chinese).  Fall back gracefully.
    _report_to = ["tensorboard"]
    try:
        output_dir.encode("ascii")
    except (UnicodeEncodeError, UnicodeDecodeError):
        _report_to = []
        logger.info("Output path contains non-ASCII chars; TensorBoard disabled")

    training_kwargs = dict(
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
        gradient_checkpointing=training_config.gradient_checkpointing,
        logging_steps=training_config.logging_steps,
        eval_steps=_eval_steps,
        save_steps=_save_steps,
        save_total_limit=training_config.save_total_limit,
        predict_with_generate=True,
        generation_max_length=model_config.max_target_length,
        report_to=_report_to,
        load_best_model_at_end=_load_best,
        metric_for_best_model="eval_loss" if has_validation else None,
        seed=training_config.seed,
        remove_unused_columns=False if is_bart_fact else True,
        dataloader_num_workers=0 if get_device().type == "cpu" else 2,
        dataloader_pin_memory=get_device().type == "cuda",
    )
    # Auto-detect: transformers>=4.45 uses 'eval_strategy', older uses 'evaluation_strategy'
    try:
        training_args = Seq2SeqTrainingArguments(
            **training_kwargs, eval_strategy=eval_strategy_val
        )
    except TypeError:
        training_args = Seq2SeqTrainingArguments(
            **training_kwargs, evaluation_strategy=eval_strategy_val
        )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if is_bart_fact:
        trainer_kwargs.update(
            bart_fact_model=bart_fact_model,
            use_cpo=cpo_enabled,
            cpo_alpha=cpo_alpha,
        )

    trainer = (
        BARTFaCTTrainer(**trainer_kwargs)
        if is_bart_fact
        else Seq2SeqTrainer(**trainer_kwargs)
    )

    logger.info(
        f"Starting training: {model_name} on {dataset_name} "
        f"with ctx={model_config.max_input_length}"
    )
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)

    trainer.save_model(output_dir)
    if not is_bart_fact:
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
    """Train a model at multiple context lengths."""
    if context_lengths is None:
        context_lengths = [256, 512, 768, 1024]

    model_config = get_model_config(model_name)
    valid_lengths = [
        cl for cl in context_lengths if cl <= model_config.max_input_length
    ]

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


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train summarization models")
    parser.add_argument(
        "--model", type=str, default="bart-large-cnn", help="Model name"
    )
    parser.add_argument(
        "--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"]
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
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
        print(
            json.dumps(
                {k: v.get("status", "unknown") for k, v in results.items()}, indent=2
            )
        )
    else:
        trainer, model, tokenizer = train_model(
            model_name=args.model,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            training_config=config,
            max_input_length=args.max_input_length,
        )
