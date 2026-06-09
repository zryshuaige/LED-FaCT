import os
import json
import logging
from typing import Dict, List

import torch
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from config import (
    ModelConfig, TrainingConfig, MODEL_CONFIGS, get_model_config, get_device,
    get_led_fact_config,
)
from data_utils import (
    load_arxiv_dataset, load_pubmed_dataset,
    prepare_dataset_for_model, prepare_dataset_for_led_fact, set_seed,
)
from models.led_fact import LEDFaCTForConditionalGeneration, LEDFaCTConfig, ABLATION_CONFIGS
from train import LEDFaCTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ABLATION_MODELS = {
    "led_baseline": {
        "config": ABLATION_CONFIGS["led_baseline"],
        "model_name": "led-baseline",
        "description": "LED baseline (no novel modules)",
    },
    "led_fact_no_sae": {
        "config": ABLATION_CONFIGS["led_fact_no_sae"],
        "model_name": "led-fact-no-sae",
        "description": "LED-FaCT w/o SAE (no section-aware embedding)",
    },
    "led_fact_no_fgca": {
        "config": ABLATION_CONFIGS["led_fact_no_fgca"],
        "model_name": "led-fact-no-fgca",
        "description": "LED-FaCT w/o FGCA (no faithfulness-gated cross-attention)",
    },
    "led_fact_no_cfl": {
        "config": ABLATION_CONFIGS["led_fact_no_cfl"],
        "model_name": "led-fact-no-cfl",
        "description": "LED-FaCT w/o CFL (no contrastive factuality loss)",
    },
    "led_fact_full": {
        "config": ABLATION_CONFIGS["led_fact_full"],
        "model_name": "led-fact-full",
        "description": "LED-FaCT (Full): LED + SAE + FGCA + CFL",
    },
}


def run_single_ablation(
    ablation_name: str,
    dataset_name: str = "arxiv",
    max_samples: int = 5000,
    num_test: int = 500,
    output_dir: str = "./results/ablation",
    epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 2,
):
    if ablation_name not in ABLATION_MODELS:
        raise ValueError(f"Unknown ablation: {ablation_name}. Available: {list(ABLATION_MODELS.keys())}")

    ablation_info = ABLATION_MODELS[ablation_name]
    led_fact_config = ablation_info["config"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Ablation Study: {ablation_name}")
    logger.info(f"  Description: {ablation_info['description']}")
    logger.info(f"  SAE: {led_fact_config.use_sae}, FGCA: {led_fact_config.use_fgca}, CFL: {led_fact_config.use_cfl}")
    logger.info(f"{'='*60}")

    set_seed(42)
    device = get_device()

    ablation_dir = os.path.join(output_dir, ablation_name)
    os.makedirs(ablation_dir, exist_ok=True)

    config_copy = LEDFaCTConfig(
        use_sae=led_fact_config.use_sae,
        use_fgca=led_fact_config.use_fgca,
        use_cfl=led_fact_config.use_cfl,
        section_embed_dim=led_fact_config.section_embed_dim,
        fgca_hidden_dim=led_fact_config.fgca_hidden_dim,
        cfl_projection_dim=led_fact_config.cfl_projection_dim,
        cfl_temperature=led_fact_config.cfl_temperature,
        cfl_alpha=led_fact_config.cfl_alpha,
        dropout=led_fact_config.dropout,
        base_model_name=led_fact_config.base_model_name,
        max_input_length=led_fact_config.max_input_length,
        max_target_length=led_fact_config.max_target_length,
    )

    model = LEDFaCTForConditionalGeneration(config_copy)
    model = model.to(device)

    param_info = model.get_trainable_params_summary()
    logger.info(f"Model parameters: {param_info}")

    with open(os.path.join(ablation_dir, "model_params.json"), "w") as f:
        json.dump(param_info, f, indent=2)

    tokenizer = model.tokenizer

    dataset = prepare_dataset_for_led_fact(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_input_length=config_copy.max_input_length,
        max_target_length=config_copy.max_target_length,
        max_samples=max_samples,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.led,
        padding=True,
    )

    use_cfl = config_copy.use_cfl
    cfl_alpha = config_copy.cfl_alpha

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(ablation_dir, "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=device.type == "cuda",
        logging_steps=100,
        eval_strategy="no",
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=config_copy.max_target_length,
        report_to=[],
        seed=42,
        dataloader_num_workers=0 if device.type == "cpu" else 4,
        dataloader_pin_memory=device.type == "cuda",
    )

    trainer = LEDFaCTTrainer(
        model=model.led,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        processing_class=tokenizer,
        data_collator=data_collator,
        led_fact_model=model,
        use_cfl=use_cfl,
        cfl_alpha=cfl_alpha,
    )

    logger.info(f"Starting training for ablation: {ablation_name}")
    train_result = trainer.train()

    model.save_pretrained(os.path.join(ablation_dir, "model"))

    logger.info(f"Training complete for {ablation_name}. Evaluating...")

    from evaluate import evaluate_model
    eval_results, summaries, references = evaluate_model(
        model_name=ablation_info["model_name"],
        dataset_name=dataset_name,
        num_test_samples=num_test,
        output_dir=ablation_dir,
    )

    from hallucination import evaluate_hallucination_for_model
    if dataset_name == "arxiv":
        ds = load_arxiv_dataset(max_samples=None)
    else:
        ds = load_pubmed_dataset(max_samples=None)
    test_key = "test" if "test" in ds else "validation"
    test_data = ds[test_key]
    if num_test and len(test_data) > num_test:
        test_data = test_data.select(range(num_test))
    source_texts = [sample["article"] for sample in test_data]

    hallucination_results = evaluate_hallucination_for_model(
        model_name=ablation_name,
        source_texts=source_texts[:len(summaries)],
        generated_summaries=summaries,
        references=references[:len(summaries)],
        use_nli=True,
        output_dir=os.path.join(ablation_dir, "hallucination"),
    )

    ablation_result = {
        "ablation_name": ablation_name,
        "description": ablation_info["description"],
        "use_sae": config_copy.use_sae,
        "use_fgca": config_copy.use_fgca,
        "use_cfl": config_copy.use_cfl,
        "model_params": param_info,
        "eval_results": eval_results,
        "hallucination_results": hallucination_results,
    }

    result_path = os.path.join(ablation_dir, "ablation_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(ablation_result, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Ablation {ablation_name} complete. Results saved to {result_path}")
    return ablation_result


def run_all_ablations(
    dataset_name: str = "arxiv",
    max_samples: int = 5000,
    num_test: int = 500,
    output_dir: str = "./results/ablation",
    epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 2,
    ablation_list: List[str] = None,
):
    if ablation_list is None:
        ablation_list = list(ABLATION_MODELS.keys())

    logger.info(f"\n{'='*60}")
    logger.info("Running Full Ablation Study (Module Ablation)")
    logger.info(f"Ablations to run: {ablation_list}")
    logger.info(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for ablation_name in ablation_list:
        logger.info(f"\n--- Running ablation: {ablation_name} ---")
        try:
            result = run_single_ablation(
                ablation_name=ablation_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                num_test=num_test,
                output_dir=output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            all_results[ablation_name] = result
        except Exception as e:
            logger.error(f"Failed ablation {ablation_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ablation_name] = {"error": str(e)}

    compare_results = {}
    for name, result in all_results.items():
        if isinstance(result, dict) and "error" not in result:
            entry = {
                "use_sae": result.get("use_sae", False),
                "use_fgca": result.get("use_fgca", False),
                "use_cfl": result.get("use_cfl", False),
            }
            if "eval_results" in result and isinstance(result["eval_results"], dict):
                if "rouge" in result["eval_results"]:
                    rouge = result["eval_results"]["rouge"]
                    entry["rouge1"] = rouge.get("rouge1", {}).get("fmeasure", 0)
                    entry["rouge2"] = rouge.get("rouge2", {}).get("fmeasure", 0)
                    entry["rougeL"] = rouge.get("rougeL", {}).get("fmeasure", 0)
            if "hallucination_results" in result and isinstance(result["hallucination_results"], dict):
                nli = result["hallucination_results"].get("nli_metrics", {})
                entry["factuality_rate"] = nli.get("factuality_rate", 0)
                entry["hallucination_rate"] = nli.get("hallucination_rate", 0)
            compare_results[name] = entry

    summary_path = os.path.join(output_dir, "ablation_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compare_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nAblation comparison saved to {summary_path}")
    logger.info("\nAblation Results Summary:")
    logger.info(f"{'Model':<25} {'SAE':>5} {'FGCA':>5} {'CFL':>5} {'R1':>8} {'R2':>8} {'RL':>8} {'Fact':>8} {'Hall':>8}")
    logger.info("-" * 90)
    for name, entry in compare_results.items():
        logger.info(
            f"{name:<25} {str(entry.get('use_sae', '-')):>5} {str(entry.get('use_fgca', '-')):>5} "
            f"{str(entry.get('use_cfl', '-')):>5} {entry.get('rouge1', 0):>8.4f} "
            f"{entry.get('rouge2', 0):>8.4f} {entry.get('rougeL', 0):>8.4f} "
            f"{entry.get('factuality_rate', 0):>8.4f} {entry.get('hallucination_rate', 0):>8.4f}"
        )

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation experiments (module ablation)")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=list(ABLATION_MODELS.keys()) + ["all"],
                        help="Which ablation to run")
    parser.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./results/ablation")

    args = parser.parse_args()

    if args.ablation == "all":
        run_all_ablations(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            num_test=args.num_test,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
    else:
        run_single_ablation(
            ablation_name=args.ablation,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            num_test=args.num_test,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )