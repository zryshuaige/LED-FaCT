import os
import json
import logging
import argparse
from datetime import datetime
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from config import (
    ModelConfig, TrainingConfig, ContextLengthExperiment,
    MODEL_CONFIGS, get_model_config, get_available_models, get_device,
)
from data_utils import set_seed, load_arxiv_dataset, load_pubmed_dataset
from train import train_model, train_multiple_context_lengths
from evaluate import evaluate_model, evaluate_context_length_impact
from hallucination import evaluate_hallucination_for_model
from ablation import run_single_ablation, run_all_ablations, ABLATION_MODELS
from sensitivity import (
    sensitivity_beam_size, sensitivity_length_penalty,
    sensitivity_learning_rate, sensitivity_cfl_alpha,
    sensitivity_fgca_dim, sensitivity_epochs,
    sensitivity_truncation_strategy, run_all_sensitivity,
)
from analyze import (
    plot_rouge_comparison, plot_context_length_impact,
    plot_hallucination_comparison, plot_ablation_comparison,
    generate_latex_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_experiment_1_model_comparison(
    dataset_name="arxiv",
    max_samples=5000,
    num_test=500,
    models=None,
    output_dir="./results",
):
    logger.info("=" * 60)
    logger.info("Experiment 1: Multi-model comparison on summarization")
    logger.info("=" * 60)

    if models is None:
        models = ["bart-large-cnn", "pegasus-arxiv", "led-base-16384", "led-fact-full"]

    all_results = {}
    all_predictions = {}

    for model_name in models:
        logger.info(f"\n--- Training {model_name} ---")
        try:
            trainer, model, tokenizer = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                training_config=TrainingConfig(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    max_samples=max_samples,
                    num_train_epochs=3,
                    output_dir=output_dir,
                ),
            )
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            continue

        logger.info(f"\n--- Evaluating {model_name} ---")
        try:
            results, summaries, references = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test,
                output_dir=output_dir,
            )
            all_results[model_name] = results
            all_predictions[model_name] = list(zip(summaries, references))
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            continue

    return all_results, all_predictions


def run_experiment_2_ablation(
    dataset_name="arxiv",
    max_samples=5000,
    num_test=500,
    output_dir="./results/ablation",
    ablation_list=None,
):
    logger.info("=" * 60)
    logger.info("Experiment 2: Module Ablation Study")
    logger.info("=" * 60)

    return run_all_ablations(
        dataset_name=dataset_name,
        max_samples=max_samples,
        num_test=num_test,
        output_dir=output_dir,
        ablation_list=ablation_list,
    )


def run_experiment_3_hallucination(
    predictions_dict=None,
    source_texts=None,
    output_dir="./results",
):
    logger.info("=" * 60)
    logger.info("Experiment 3: Hallucination detection and analysis")
    logger.info("=" * 60)

    all_hallucination_results = {}

    if predictions_dict is not None:
        for model_name, predictions in predictions_dict.items():
            summaries = [p[0] for p in predictions]
            references = [p[1] for p in predictions]

            if source_texts is None:
                ds = load_arxiv_dataset()
                source_texts = [sample["article"] for sample in ds["test"].select(range(len(summaries)))]

            results = evaluate_hallucination_for_model(
                model_name=model_name,
                source_texts=source_texts[:len(summaries)],
                generated_summaries=summaries,
                references=references,
                use_nli=True,
                output_dir=os.path.join(output_dir, "hallucination"),
            )
            all_hallucination_results[model_name] = results

    return all_hallucination_results


def run_experiment_4_context_length(
    model_name="led-fact-full",
    dataset_name="arxiv",
    context_lengths=None,
    max_samples=5000,
    num_test=500,
    output_dir="./results",
):
    logger.info("=" * 60)
    logger.info(f"Experiment 4: Context length impact ({model_name})")
    logger.info("=" * 60)

    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096, 8192]

    all_results = train_multiple_context_lengths(
        model_name=model_name,
        context_lengths=context_lengths,
        dataset_name=dataset_name,
        max_samples=max_samples,
    )

    eval_results = evaluate_context_length_impact(
        model_name=model_name,
        context_lengths=context_lengths,
        dataset_name=dataset_name,
        num_test_samples=num_test,
        output_dir=output_dir,
    )

    return eval_results


def run_experiment_5_sensitivity(
    model_name="led-fact-full",
    dataset_name="arxiv",
    max_samples=5000,
    num_test=500,
    output_dir="./results/sensitivity",
):
    logger.info("=" * 60)
    logger.info("Experiment 5: Parameter sensitivity analysis")
    logger.info("=" * 60)

    return run_all_sensitivity(
        model_name=model_name,
        dataset_name=dataset_name,
        max_samples=max_samples,
        num_test=num_test,
        output_dir=output_dir,
    )


def run_experiment_6_truncation(
    model_name="led-fact-full",
    dataset_name="arxiv",
    num_test=500,
    output_dir="./results/sensitivity",
):
    logger.info("=" * 60)
    logger.info("Experiment 6: Truncation strategy comparison")
    logger.info("=" * 60)

    return sensitivity_truncation_strategy(
        model_name=model_name,
        dataset_name=dataset_name,
        num_test=num_test,
        output_dir=output_dir,
    )


def run_full_pipeline(
    dataset_name="arxiv",
    max_samples=5000,
    num_test=500,
    output_dir="./results",
    models=None,
    context_lengths=None,
):
    set_seed(42)

    if models is None:
        models = ["bart-large-cnn", "pegasus-arxiv", "led-base-16384", "led-fact-full"]
    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096, 8192]

    results_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Starting full experimental pipeline...")

    exp1_results, predictions = run_experiment_1_model_comparison(
        dataset_name=dataset_name, max_samples=max_samples, num_test=num_test,
        models=models, output_dir=results_dir,
    )

    exp2_results = run_experiment_2_ablation(
        dataset_name=dataset_name, max_samples=max_samples, num_test=num_test,
        output_dir=os.path.join(results_dir, "ablation"),
    )

    if predictions:
        ds = load_arxiv_dataset() if dataset_name == "arxiv" else load_pubmed_dataset()
        source_texts = [sample["article"] for sample in ds["test"].select(range(num_test))]
        exp3_results = run_experiment_3_hallucination(
            predictions_dict=predictions, source_texts=source_texts, output_dir=results_dir,
        )

    led_config = get_model_config("led-fact-full")
    valid_lengths = [cl for cl in context_lengths if cl <= led_config.max_input_length]
    exp4_results = run_experiment_4_context_length(
        model_name="led-fact-full", dataset_name=dataset_name,
        context_lengths=valid_lengths, max_samples=max_samples,
        num_test=num_test, output_dir=results_dir,
    )

    exp5_results = run_experiment_5_sensitivity(
        model_name="led-fact-full", dataset_name=dataset_name,
        max_samples=max_samples, num_test=num_test,
        output_dir=os.path.join(results_dir, "sensitivity"),
    )

    figures_dir = os.path.join(results_dir, "figures")
    if exp1_results:
        plot_rouge_comparison(exp1_results, figures_dir, dataset_name)
        generate_latex_table(exp1_results, figures_dir)

    if exp2_results:
        plot_ablation_comparison(exp2_results, figures_dir)

    if exp4_results:
        plot_context_length_impact(exp4_results, figures_dir, "led-fact-full")

    logger.info(f"\nFull pipeline complete! Results saved to: {results_dir}")
    return results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LED-FaCT: Run experiments")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6",
                                 "ablation", "sensitivity", "quick_test"],
                        help="Which experiment to run")
    parser.add_argument("--ablation_type", type=str, default="all",
                        choices=["all"] + list(ABLATION_MODELS.keys()),
                        help="Which ablation to run")
    parser.add_argument("--sensitivity_type", type=str, default="all",
                        choices=["all", "beam_size", "length_penalty", "learning_rate",
                                  "cfl_alpha", "fgca_dim", "epochs", "truncation"],
                        help="Which sensitivity analysis to run")
    parser.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--model", type=str, default="led-fact-full")
    parser.add_argument("--context_lengths", type=str, default=None)

    args = parser.parse_args()

    models = args.models.split(",") if args.models else None
    ctx_lengths = [int(x) for x in args.context_lengths.split(",")] if args.context_lengths else None

    if args.mode == "full":
        run_full_pipeline(
            dataset_name=args.dataset, max_samples=args.max_samples,
            num_test=args.num_test, output_dir=args.output_dir,
            models=models, context_lengths=ctx_lengths,
        )
    elif args.mode == "exp1":
        run_experiment_1_model_comparison(
            dataset_name=args.dataset, max_samples=args.max_samples,
            num_test=args.num_test, models=models, output_dir=args.output_dir,
        )
    elif args.mode == "exp2" or args.mode == "ablation":
        if args.ablation_type == "all":
            run_experiment_2_ablation(
                dataset_name=args.dataset, max_samples=args.max_samples,
                num_test=args.num_test, output_dir=os.path.join(args.output_dir, "ablation"),
            )
        else:
            run_single_ablation(
                ablation_name=args.ablation_type, dataset_name=args.dataset,
                max_samples=args.max_samples, num_test=args.num_test,
                output_dir=os.path.join(args.output_dir, "ablation"),
            )
    elif args.mode == "exp3":
        logger.info("Exp3 requires prediction files. Use --mode full instead.")
    elif args.mode == "exp4":
        run_experiment_4_context_length(
            model_name=args.model, context_lengths=ctx_lengths,
            max_samples=args.max_samples, num_test=args.num_test, output_dir=args.output_dir,
        )
    elif args.mode == "exp5" or args.mode == "sensitivity":
        run_experiment_5_sensitivity(
            model_name=args.model, dataset_name=args.dataset,
            max_samples=args.max_samples, num_test=args.num_test,
            output_dir=os.path.join(args.output_dir, "sensitivity"),
        )
    elif args.mode == "exp6":
        run_experiment_6_truncation(
            model_name=args.model, dataset_name=args.dataset,
            num_test=args.num_test, output_dir=args.output_dir,
        )
    elif args.mode == "quick_test":
        logger.info("Running quick test with minimal data...")
        results, preds = run_experiment_1_model_comparison(
            dataset_name=args.dataset, max_samples=100, num_test=10,
            models=["led-fact-full"], output_dir=os.path.join(args.output_dir, "quick_test"),
        )
        for k, v in results.items():
            print(f"\n{k}: ROUGE-1={v['rouge']['rouge1']['fmeasure']:.4f}, "
                  f"ROUGE-L={v['rouge']['rougeL']['fmeasure']:.4f}")