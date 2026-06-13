import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def _setup_chinese_font():
    chinese_fonts = [
        "SimHei", "Microsoft YaHei", "STHeiti", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Noto Sans SC", "PingFang SC", "Hiragino Sans GB",
        "Source Han Sans SC", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams.get("font.sans-serif", ["DejaVu Sans"])
            logger.info(f"Using Chinese font: {font_name}")
            break
    else:
        logger.warning("No Chinese font found. Chinese characters may not display correctly in plots.")

_setup_chinese_font()
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]

MODEL_DISPLAY_NAMES = {
    "bart-large": "BART-Large",
    "bart-large-cnn": "BART-Large-CNN",
    "pegasus-arxiv": "PEGASUS-arXiv",
    "pegasus-pubmed": "PEGASUS-PubMed",
    "led-base-16384": "LED-Base-16K",
    "led-baseline": "LED (Baseline)",
    "primera": "PRIMERA",
    "led-fact-full": "LED-FaCT (Full)",
    "led-fact-no-sae": "LED-FaCT w/o SAE",
    "led-fact-no-fgca": "LED-FaCT w/o FGCA",
    "led-fact-no-cfl": "LED-FaCT w/o CFL",
}

ABLATION_DISPLAY_NAMES = {
    "led_baseline": "LED (Baseline)",
    "led_fact_no_sae": "w/o SAE",
    "led_fact_no_fgca": "w/o FGCA",
    "led_fact_no_cfl": "w/o CFL",
    "led_fact_full": "LED-FaCT (Full)",
}

ABLATION_SHORT_NAMES = {
    "led_baseline": "Baseline",
    "led_fact_no_sae": "-SAE",
    "led_fact_no_fgca": "-FGCA",
    "led_fact_no_cfl": "-CFL",
    "led_fact_full": "Full",
}


def load_results(results_dir: str, model_name: str, dataset_name: str) -> Optional[Dict]:
    for ctx_len in [512, 1024, 2048, 4096, 8192]:
        path = os.path.join(results_dir, f"{model_name}_{dataset_name}_ctx{ctx_len}", "eval_results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def load_context_length_results(results_dir: str, pattern: str = "context_length_impact") -> Dict:
    all_results = {}
    for fname in os.listdir(results_dir):
        if pattern in fname and fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_results[fname] = data
    return all_results


def plot_rouge_comparison(
    results_dict: Dict[str, Dict],
    output_path: str = "./results/figures",
    dataset_name: str = "arxiv",
):
    os.makedirs(output_path, exist_ok=True)

    models = list(results_dict.keys())
    rouge1_f = [results_dict[m]["rouge"]["rouge1"]["fmeasure"] for m in models]
    rouge2_f = [results_dict[m]["rouge"]["rouge2"]["fmeasure"] for m in models]
    rougeL_f = [results_dict[m]["rouge"]["rougeL"]["fmeasure"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    for idx, (metric, values, title) in enumerate([
        ("ROUGE-1", rouge1_f, "ROUGE-1 F1"),
        ("ROUGE-2", rouge2_f, "ROUGE-2 F1"),
        ("ROUGE-L", rougeL_f, "ROUGE-L F1"),
    ]):
        bars = axes[idx].bar(range(len(models)), values, color=COLORS[:len(models)])
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
        axes[idx].set_ylabel("F1 Score", fontsize=11)
        axes[idx].set_title(title, fontsize=13, fontweight="bold")
        axes[idx].set_ylim(0, max(values) * 1.2)
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                          f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Model Performance Comparison ({dataset_name.upper()})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "rouge_comparison.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_path, "rouge_comparison.pdf"), bbox_inches="tight")
    plt.close()

    df = pd.DataFrame({
        "Model": display_names, "ROUGE-1": rouge1_f, "ROUGE-2": rouge2_f, "ROUGE-L": rougeL_f,
    })
    df.to_csv(os.path.join(output_path, "rouge_comparison.csv"), index=False)
    return df


def plot_ablation_comparison(
    ablation_results: Dict,
    output_path: str = "./results/figures",
):
    os.makedirs(output_path, exist_ok=True)

    ablation_order = ["led_baseline", "led_fact_no_sae", "led_fact_no_fgca", "led_fact_no_cfl", "led_fact_full"]
    available = [k for k in ablation_order if k in ablation_results and isinstance(ablation_results[k], dict) and "error" not in ablation_results[k]]

    if not available:
        logger.warning("No ablation results available for plotting")
        return

    names = [ABLATION_SHORT_NAMES.get(k, k) for k in available]
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    fact_scores = []
    hall_scores = []

    for k in available:
        r = ablation_results[k]
        if "eval_results" in r:
            eval_r = r["eval_results"]
            rouge1_scores.append(eval_r.get("rouge", {}).get("rouge1", {}).get("fmeasure", 0))
            rouge2_scores.append(eval_r.get("rouge", {}).get("rouge2", {}).get("fmeasure", 0))
            rougeL_scores.append(eval_r.get("rouge", {}).get("rougeL", {}).get("fmeasure", 0))
        else:
            rouge1_scores.append(0)
            rouge2_scores.append(0)
            rougeL_scores.append(0)

        if "hallucination_results" in r:
            hall_r = r["hallucination_results"]
            nli = hall_r.get("nli_metrics", {})
            fact_scores.append(nli.get("factuality_rate", 0))
            hall_scores.append(nli.get("hallucination_rate", 0))
        else:
            fact_scores.append(0)
            hall_scores.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(range(len(names)), rouge1_scores, color=COLORS[:len(names)])
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, fontsize=10)
    axes[0, 0].set_ylabel("F1 Score", fontsize=11)
    axes[0, 0].set_title("ROUGE-1", fontsize=13, fontweight="bold")
    for i, v in enumerate(rouge1_scores):
        axes[0, 0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    axes[0, 1].bar(range(len(names)), rouge2_scores, color=COLORS[:len(names)])
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, fontsize=10)
    axes[0, 1].set_ylabel("F1 Score", fontsize=11)
    axes[0, 1].set_title("ROUGE-2", fontsize=13, fontweight="bold")
    for i, v in enumerate(rouge2_scores):
        axes[0, 1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    axes[1, 0].bar(range(len(names)), rougeL_scores, color=COLORS[:len(names)])
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, fontsize=10)
    axes[1, 0].set_ylabel("F1 Score", fontsize=11)
    axes[1, 0].set_title("ROUGE-L", fontsize=13, fontweight="bold")
    for i, v in enumerate(rougeL_scores):
        axes[1, 0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    if any(s > 0 for s in fact_scores):
        width = 0.35
        x = np.arange(len(names))
        axes[1, 1].bar(x - width / 2, fact_scores, width, label="Factuality", color=COLORS[1])
        axes[1, 1].bar(x + width / 2, hall_scores, width, label="Hallucination", color=COLORS[3])
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(names, fontsize=10)
        axes[1, 1].set_ylabel("Rate", fontsize=11)
        axes[1, 1].set_title("Factuality vs Hallucination", fontsize=13, fontweight="bold")
        axes[1, 1].legend(fontsize=10)

    plt.suptitle("Module Ablation Study Results", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ablation_comparison.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_path, "ablation_comparison.pdf"), bbox_inches="tight")
    plt.close()

    df = pd.DataFrame({
        "Model": names,
        "SAE": [str(ablation_results.get(k, {}).get("use_sae", "-")) for k in available],
        "FGCA": [str(ablation_results.get(k, {}).get("use_fgca", "-")) for k in available],
        "CFL": [str(ablation_results.get(k, {}).get("use_cfl", "-")) for k in available],
        "ROUGE-1": rouge1_scores,
        "ROUGE-2": rouge2_scores,
        "ROUGE-L": rougeL_scores,
        "Factuality": fact_scores,
        "Hallucination": hall_scores,
    })
    df.to_csv(os.path.join(output_path, "ablation_comparison.csv"), index=False)
    return df


def plot_context_length_impact(
    context_results: Dict[int, Dict],
    output_path: str = "./results/figures",
    model_name: str = "",
):
    os.makedirs(output_path, exist_ok=True)

    ctx_lengths = sorted(context_results.keys())
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for cl in ctx_lengths:
        r = context_results[cl]
        if "rouge" in r:
            rouge1_scores.append(r["rouge"]["rouge1"]["fmeasure"])
            rouge2_scores.append(r["rouge"]["rouge2"]["fmeasure"])
            rougeL_scores.append(r["rouge"]["rougeL"]["fmeasure"])
        elif "error" in r:
            rouge1_scores.append(None)
            rouge2_scores.append(None)
            rougeL_scores.append(None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ctx_lengths, rouge1_scores, "o-", label="ROUGE-1", linewidth=2, markersize=8, color=COLORS[0])
    ax.plot(ctx_lengths, rouge2_scores, "s-", label="ROUGE-2", linewidth=2, markersize=8, color=COLORS[1])
    ax.plot(ctx_lengths, rougeL_scores, "^-", label="ROUGE-L", linewidth=2, markersize=8, color=COLORS[2])

    ax.set_xlabel("Input Context Length (tokens)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(f"Impact of Context Length on Summarization Quality ({display_name})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx_lengths)
    ax.set_xticklabels([str(x) for x in ctx_lengths])

    for x, y1, y2, y3 in zip(ctx_lengths, rouge1_scores, rouge2_scores, rougeL_scores):
        if y1 is not None:
            ax.annotate(f"{y1:.3f}", (x, y1), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"context_length_impact_{model_name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_path, f"context_length_impact_{model_name}.pdf"), bbox_inches="tight")
    plt.close()

    df = pd.DataFrame({
        "Context Length": ctx_lengths, "ROUGE-1": rouge1_scores,
        "ROUGE-2": rouge2_scores, "ROUGE-L": rougeL_scores,
    })
    df.to_csv(os.path.join(output_path, f"context_length_impact_{model_name}.csv"), index=False)
    return df


def plot_hallucination_comparison(
    hallucination_results: Dict[str, Dict],
    output_path: str = "./results/figures",
):
    os.makedirs(output_path, exist_ok=True)

    models = list(hallucination_results.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    factuality_rates = []
    hallucination_rates = []
    intrinsic_rates = []
    extrinsic_rates = []
    contradiction_rates = []

    for m in models:
        r = hallucination_results[m]
        if "nli_metrics" in r:
            factuality_rates.append(r["nli_metrics"].get("factuality_rate", 0))
            hallucination_rates.append(r["nli_metrics"].get("hallucination_rate", 0))
        if "hallucination_types" in r:
            ht = r["hallucination_types"]
            intrinsic_rates.append(ht.get("intrinsic_rate", 0))
            extrinsic_rates.append(ht.get("extrinsic_rate", 0))
            contradiction_rates.append(ht.get("contradictory_rate", 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if factuality_rates:
        x = range(len(models))
        width = 0.35
        axes[0].bar([i - width / 2 for i in x], factuality_rates, width, label="Factuality", color=COLORS[1])
        axes[0].bar([i + width / 2 for i in x], hallucination_rates, width, label="Hallucination", color=COLORS[3])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
        axes[0].set_ylabel("Rate", fontsize=11)
        axes[0].set_title("Factuality vs Hallucination Rate", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=10)

    if intrinsic_rates:
        x = range(len(models))
        width = 0.25
        axes[1].bar([i - width for i in x], intrinsic_rates, width, label="Intrinsic", color=COLORS[0])
        axes[1].bar([i for i in x], extrinsic_rates, width, label="Extrinsic", color=COLORS[2])
        axes[1].bar([i + width for i in x], contradiction_rates, width, label="Contradictory", color=COLORS[3])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
        axes[1].set_ylabel("Rate", fontsize=11)
        axes[1].set_title("Hallucination Type Distribution", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "hallucination_comparison.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_path, "hallucination_comparison.pdf"), bbox_inches="tight")
    plt.close()

    df = pd.DataFrame({
        "Model": display_names, "Factuality Rate": factuality_rates,
        "Hallucination Rate": hallucination_rates, "Intrinsic Rate": intrinsic_rates,
        "Extrinsic Rate": extrinsic_rates, "Contradictory Rate": contradiction_rates,
    })
    df.to_csv(os.path.join(output_path, "hallucination_comparison.csv"), index=False)
    return df


def generate_latex_table(
    results_dict: Dict[str, Dict],
    output_path: str = "./results/figures",
):
    os.makedirs(output_path, exist_ok=True)

    latex_rows = []
    for model_name, results in results_dict.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        r1 = results["rouge"]["rouge1"]["fmeasure"]
        r2 = results["rouge"]["rouge2"]["fmeasure"]
        rL = results["rouge"]["rougeL"]["fmeasure"]
        ctx = results.get("max_input_length", "1024")
        latex_rows.append(f"{display_name} & {ctx} & {r1:.4f} & {r2:.4f} & {rL:.4f} \\\\")

    latex_table = "\\begin{table}[htbp]\n\\centering\n\\caption{Summarization performance comparison}\n"
    latex_table += "\\label{tab:rouge_results}\n\\begin{tabular}{lcccc}\n\\hline\n"
    latex_table += "Model & Context Length & ROUGE-1 & ROUGE-2 & ROUGE-L \\\\\n\\hline\n"
    latex_table += "\n".join(latex_rows) + "\n\\hline\n\\end{tabular}\n\\end{table}"

    with open(os.path.join(output_path, "results_table.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table)

    return latex_table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate analysis figures and tables")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    args = parser.parse_args()

    all_results = {}
    for model_name in ["bart-large", "bart-large-cnn", "pegasus-arxiv", "led-base-16384", "led-fact-full"]:
        for ctx in [1024, 8192]:
            path = os.path.join(args.results_dir, f"{model_name}_arxiv_ctx{ctx}", "eval_results.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    all_results[model_name] = json.load(f)
                break

    if all_results:
        plot_rouge_comparison(all_results, args.output_dir)
        generate_latex_table(all_results, args.output_dir)

    ablation_dir = os.path.join(args.results_dir, "ablation")
    if os.path.exists(ablation_dir):
        comparison_path = os.path.join(ablation_dir, "ablation_comparison.json")
        if os.path.exists(comparison_path):
            with open(comparison_path, "r", encoding="utf-8") as f:
                ablation_data = json.load(f)
            plot_ablation_comparison(ablation_data, args.output_dir)