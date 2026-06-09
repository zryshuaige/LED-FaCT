import os
import html
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from IPython.display import HTML, display

matplotlib.rcParams.update({
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.edgecolor": "#555555",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

from models.section_embedding import SectionDetector, SECTION_LABELS, LABEL_TO_ID, NUM_SECTION_TYPES


SECTION_COLORS = {
    "PAD": "#D9D9D9",
    "ABSTRACT": "#4ECDC4",
    "INTRODUCTION": "#45B7D1",
    "METHOD": "#F7DC6F",
    "EXPERIMENT": "#F39C12",
    "RESULT": "#E74C3C",
    "CONCLUSION": "#9B59B6",
    "OTHER": "#95A5A6",
}

SECTION_COLORS_IDX = {i: SECTION_COLORS[v] for i, v in SECTION_LABELS.items()}


def visualize_section_detection(
    text: str,
    max_chars: int = 3000,
    title: str = "SAE Section Detection",
):
    detector = SectionDetector()
    spans = detector.detect_sections(text)

    display_text = text[:max_chars]
    current_pos = 0
    html_parts = [f"<h3>{title}</h3><div style='font-family:monospace; font-size:13px; line-height:1.8; white-space:pre-wrap;'>"]

    for span in spans:
        start = span.start_char
        end = min(span.end_char, max_chars)
        if start >= max_chars:
            break
        if start > current_pos:
            html_parts.append(html.escape(display_text[current_pos:start]))
        color = SECTION_COLORS.get(span.label, "#95A5A6")
        segment = html.escape(display_text[start:end])
        html_parts.append(f'<span style="background-color:{color}; border-radius:3px; padding:1px 2px;" title="{span.label}">{segment}</span>')
        current_pos = end

    if current_pos < len(display_text):
        html_parts.append(html.escape(display_text[current_pos:]))

    legend = "<div style='margin-top:10px; display:flex; flex-wrap:wrap; gap:8px;'>"
    for label, color in SECTION_COLORS.items():
        if label != "PAD":
            legend += f"<span style='background-color:{color}; padding:2px 8px; border-radius:3px; font-size:12px;'>{label}</span>"
    legend += "</div>"

    html_parts.append(legend)
    html_parts.append("</div>")
    display(HTML("".join(html_parts)))

    return spans


def plot_section_distribution(
    section_ids_list: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Section ID Distribution",
    output_path: Optional[str] = None,
):
    if labels is None:
        labels = [f"Sample {i+1}" for i in range(len(section_ids_list))]

    fig, axes = plt.subplots(1, len(section_ids_list), figsize=(5 * len(section_ids_list), 4))
    if len(section_ids_list) == 1:
        axes = [axes]

    section_names = [SECTION_LABELS[i] for i in range(NUM_SECTION_TYPES)]
    for ax, sids, label in zip(axes, section_ids_list, labels):
        valid = sids[sids > 0]
        counts = np.bincount(valid, minlength=NUM_SECTION_TYPES)
        colors = [SECTION_COLORS_IDX[i] for i in range(NUM_SECTION_TYPES)]
        bars = ax.bar(range(NUM_SECTION_TYPES), counts, color=colors)
        ax.set_xticks(range(NUM_SECTION_TYPES))
        ax.set_xticklabels(section_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Token Count")
        ax.set_title(label, fontsize=11)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_section_embedding_heatmap(
    section_embedding_weight: np.ndarray,
    title: str = "Section Embedding Heatmap",
    output_path: Optional[str] = None,
):
    section_names = [SECTION_LABELS[i] for i in range(NUM_SECTION_TYPES)]
    embed_dim = section_embedding_weight.shape[1]

    fig, ax = plt.subplots(figsize=(max(8, embed_dim * 0.05), 4))
    sns.heatmap(
        section_embedding_weight,
        xticklabels=False,
        yticklabels=section_names,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Section Type")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_section_embedding_tsne(
    section_embedding_weight: np.ndarray,
    title: str = "Section Embedding t-SNE",
    output_path: Optional[str] = None,
):
    from sklearn.manifold import TSNE

    section_names = [SECTION_LABELS[i] for i in range(NUM_SECTION_TYPES)]
    colors = [SECTION_COLORS_IDX[i] for i in range(NUM_SECTION_TYPES)]

    if section_embedding_weight.shape[1] <= 2:
        coords = section_embedding_weight
    else:
        perplexity = min(3, NUM_SECTION_TYPES - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(section_embedding_weight)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(section_names, colors)):
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=200, label=name, zorder=5)
        ax.annotate(name, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_fgca_gate_heatmap(
    gate_values_all_layers: List[np.ndarray],
    token_labels: Optional[List[str]] = None,
    max_tokens: int = 50,
    title: str = "FGCA Gate Values Heatmap",
    output_path: Optional[str] = None,
):
    num_layers = len(gate_values_all_layers)
    min_len = min(g.shape[0] for g in gate_values_all_layers)
    min_len = min(min_len, max_tokens)

    gate_matrix = np.array([g[:min_len].mean(axis=-1) if g.ndim > 1 else g[:min_len]
                            for g in gate_values_all_layers])

    if token_labels is not None:
        token_labels = token_labels[:min_len]
    else:
        token_labels = [f"t{i}" for i in range(min_len)]

    fig, ax = plt.subplots(figsize=(max(12, min_len * 0.3), max(4, num_layers * 0.5)))

    cmap = LinearSegmentedColormap.from_list("gate_cmap", ["#E74C3C", "#FDEBD0", "#27AE60"])
    im = ax.imshow(gate_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(num_layers)], fontsize=9)
    ax.set_xlabel("Decoder Token Position")
    ax.set_ylabel("Decoder Layer")
    ax.set_title(title, fontsize=14, fontweight="bold")

    step = max(1, min_len // 20)
    ax.set_xticks(range(0, min_len, step))
    ax.set_xticklabels(token_labels[::step], rotation=45, ha="right", fontsize=7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Gate Value (0=Self, 1=Faithful)", fontsize=10)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_fgca_gate_histogram(
    gate_values: np.ndarray,
    title: str = "FGCA Gate Value Distribution",
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    flat = gate_values.flatten()
    ax.hist(flat, bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.5, color="#E74C3C", linestyle="--", linewidth=2, label="gate=0.5 (threshold)")
    ax.axvline(x=float(np.mean(flat)), color="#27AE60", linestyle="-", linewidth=2,
               label=f"mean={np.mean(flat):.3f}")

    ax.set_xlabel("Gate Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    mean_val = np.mean(flat)
    idx_below = np.sum(flat < 0.5) / len(flat)
    idx_above = np.sum(flat >= 0.5) / len(flat)
    ax.text(0.02, 0.95, f"Mean: {mean_val:.3f}\n<0.5 (self): {idx_below:.1%}\n>=0.5 (faithful): {idx_above:.1%}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def highlight_diff(original: str, perturbed: str) -> str:
    orig_words = original.split()
    pert_words = perturbed.split()
    result_parts = []

    max_len = max(len(orig_words), len(pert_words))
    for i in range(max_len):
        if i < len(pert_words):
            if i < len(orig_words) and pert_words[i] == orig_words[i]:
                result_parts.append(html.escape(pert_words[i]))
            else:
                result_parts.append(
                    f'<span style="background-color:#FFEB3B; padding:1px 2px; border-radius:2px;">'
                    f'{html.escape(pert_words[i])}</span>'
                )

    return " ".join(result_parts)


def visualize_cfl_perturbation(
    original: str,
    title: str = "CFL Contrastive Perturbation",
):
    from models.contrastive_loss import SummaryPerturbator

    perturbator = SummaryPerturbator(seed=42)

    perturbed_num = perturbator.perturb(original, strategy="numbers")
    perturbed_ent = perturbator.perturb(original, strategy="entities")
    perturbed_ant = perturbator.perturb(original, strategy="antonyms")

    max_display = 500
    orig_trunc = original[:max_display] + ("..." if len(original) > max_display else "")

    html_parts = [
        f"<h3>{title}</h3>",
        "<div style='font-family:monospace; font-size:13px; line-height:1.8;'>",

        "<div style='margin-bottom:12px;'>",
        "<strong style='color:#2C3E50;'>Original Summary:</strong><br>",
        f"<div style='background-color:#E8F5E9; padding:8px; border-radius:5px; margin-top:4px;'>"
        f"{html.escape(orig_trunc)}</div></div>",

        "<div style='margin-bottom:12px;'>",
        "<strong style='color:#E74C3C;'>Perturbed (Numbers):</strong><br>",
        f"<div style='background-color:#FFF3E0; padding:8px; border-radius:5px; margin-top:4px;'>"
        f"{highlight_diff(orig_trunc, perturbed_num[:max_display])}</div></div>",

        "<div style='margin-bottom:12px;'>",
        "<strong style='color:#9B59B6;'>Perturbed (Entities):</strong><br>",
        f"<div style='background-color:#E8EAF6; padding:8px; border-radius:5px; margin-top:4px;'>"
        f"{highlight_diff(orig_trunc, perturbed_ent[:max_display])}</div></div>",

        "<div style='margin-bottom:12px;'>",
        "<strong style='color:#F39C12;'>Perturbed (Antonyms):</strong><br>",
        f"<div style='background-color:#FCE4EC; padding:8px; border-radius:5px; margin-top:4px;'>"
        f"{highlight_diff(orig_trunc, perturbed_ant[:max_display])}</div></div>",

        "</div>",
    ]

    display(HTML("".join(html_parts)))
    return {
        "original": original,
        "perturbed_numbers": perturbed_num,
        "perturbed_entities": perturbed_ent,
        "perturbed_antonyms": perturbed_ant,
    }


def plot_contrastive_similarity(
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
    title: str = "CFL Positive vs Negative Similarity",
    output_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(pos_sims, bins=30, alpha=0.7, color="#27AE60", label=f"Positive (mean={np.mean(pos_sims):.3f})", density=True)
    ax.hist(neg_sims, bins=30, alpha=0.7, color="#E74C3C", label=f"Negative (mean={np.mean(neg_sims):.3f})", density=True)

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    margin = np.mean(pos_sims) - np.mean(neg_sims)
    ax.text(0.02, 0.95, f"Margin: {margin:.3f}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_summary_comparison(
    summaries_dict: Dict[str, str],
    reference: str,
    source_title: str = "",
    max_len: int = 400,
    title: str = "Generated Summary Comparison",
):
    model_colors = {
        "Reference": "#27AE60",
        "LED (Baseline)": "#3498DB",
        "LED-FaCT (Full)": "#E74C3C",
        "LED-FaCT w/o SAE": "#F39C12",
        "LED-FaCT w/o FGCA": "#9B59B6",
        "LED-FaCT w/o CFL": "#1ABC9C",
        "BART-Large-CNN": "#95A5A6",
        "PEGASUS-arXiv": "#34495E",
    }

    html_parts = [
        f"<h3>{title}</h3>",
    ]
    if source_title:
        html_parts.append(f"<p><strong>Source:</strong> {html.escape(source_title)}</p>")

    html_parts.append("<div style='display:flex; flex-direction:column; gap:8px; font-family:monospace; font-size:13px;'>")

    ref_trunc = reference[:max_len] + ("..." if len(reference) > max_len else "")
    html_parts.append(
        f"<div style='background-color:#E8F5E9; padding:8px; border-radius:5px; "
        f"border-left:4px solid {model_colors['Reference']};'>"
        f"<strong style='color:{model_colors['Reference']};'>Reference:</strong><br>"
        f"{html.escape(ref_trunc)}</div>"
    )

    for model_name, summary in summaries_dict.items():
        color = model_colors.get(model_name, "#607D8B")
        sum_trunc = summary[:max_len] + ("..." if len(summary) > max_len else "")
        html_parts.append(
            f"<div style='background-color:#FAFAFA; padding:8px; border-radius:5px; "
            f"border-left:4px solid {color};'>"
            f"<strong style='color:{color};'>{html.escape(model_name)}:</strong><br>"
            f"{html.escape(sum_trunc)}</div>"
        )

    html_parts.append("</div>")
    display(HTML("".join(html_parts)))


def plot_ablation_radar(
    ablation_results: Dict[str, Dict],
    metrics: Optional[List[str]] = None,
    title: str = "Module Ablation Radar Chart",
    output_path: Optional[str] = None,
):
    if metrics is None:
        metrics = ["rouge1", "rouge2", "rougeL", "factuality", "bertscore_f1"]

    metric_labels = {
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "factuality": "Factuality",
        "bertscore_f1": "BERTScore",
    }

    ablation_order = ["led_baseline", "led_fact_no_sae", "led_fact_no_fgca", "led_fact_no_cfl", "led_fact_full"]
    ablation_labels = {
        "led_baseline": "LED (Baseline)",
        "led_fact_no_sae": "w/o SAE",
        "led_fact_no_fgca": "w/o FGCA",
        "led_fact_no_cfl": "w/o CFL",
        "led_fact_full": "LED-FaCT (Full)",
    }
    ablation_colors = {
        "led_baseline": "#3498DB",
        "led_fact_no_sae": "#F39C12",
        "led_fact_no_fgca": "#9B59B6",
        "led_fact_no_cfl": "#1ABC9C",
        "led_fact_full": "#E74C3C",
    }

    available = [k for k in ablation_order if k in ablation_results and isinstance(ablation_results[k], dict) and "error" not in ablation_results[k]]
    if not available:
        print("No ablation results available for radar chart")
        return None

    def extract_metric(data, metric):
        r = data
        if metric.startswith("rouge"):
            if "eval_results" in r and isinstance(r["eval_results"], dict):
                rouge = r["eval_results"].get("rouge", {})
                key = metric + "" if not metric.endswith("L") else metric
                return rouge.get(metric, {}).get("fmeasure", 0)
        if metric == "factuality":
            if "hallucination_results" in r:
                nli = r["hallucination_results"].get("nli_metrics", {})
                return nli.get("factuality_rate", 0)
        if metric == "bertscore_f1":
            if "eval_results" in r and isinstance(r["eval_results"], dict):
                bench = r["eval_results"].get("benchmark", {})
                bs = bench.get("bertscore", {})
                return bs.get("bertscore_f1", 0)
        return 0

    data_matrix = []
    for key in available:
        row = []
        for m in metrics:
            val = extract_metric(ablation_results[key], m)
            row.append(val)
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)

    mins = data_matrix.min(axis=0)
    maxs = data_matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (data_matrix - mins) / ranges

    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, key in enumerate(available):
        values = normalized[i].tolist()
        values += values[:1]
        label = ablation_labels.get(key, key)
        color = ablation_colors.get(key, "#607D8B")
        ax.plot(angles, values, "o-", linewidth=2, markersize=6, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    angle_labels = [metric_labels.get(m, m) for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(angle_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def print_comparison_table(
    results_df: pd.DataFrame,
    title: str = "Experiment Results Comparison",
):
    styled = results_df.style.set_caption(title)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        styled = styled.background_gradient(subset=numeric_cols, cmap="RdYlGn", axis=0)
    styled = styled.set_properties(**{"text-align": "center", "font-size": "12px"})
    styled = styled.set_table_styles([
        {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold"), ("font-size", "12px")]},
        {"selector": "caption", "props": [("font-size", "14px"), ("font-weight", "bold")]},
    ])
    display(styled)


def plot_training_curves(
    log_history: List[Dict],
    title: str = "Training Loss Curves",
    output_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_losses = [e["loss"] for e in log_history if "loss" in e]

    if train_steps:
        axes[0].plot(train_steps, train_losses, linewidth=1.5, color="#3498DB")
        axes[0].set_xlabel("Training Steps")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Training Loss", fontsize=13, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

    ce_losses = [e.get("ce_loss", e.get("loss", 0)) for e in log_history if "loss" in e]
    cfl_losses = [e.get("cfl_loss", 0) for e in log_history if "loss" in e]

    if any(c > 0 for c in cfl_losses):
        axes[1].plot(train_steps, ce_losses, linewidth=1.5, color="#3498DB", label="CE Loss")
        axes[1].plot(train_steps, cfl_losses, linewidth=1.5, color="#E74C3C", label="CFL Loss")
        axes[1].set_xlabel("Training Steps")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("CE Loss vs CFL Loss", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No CFL loss data\n(LED baseline or CFL disabled)",
                     ha="center", va="center", fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title("CFL Loss (N/A)", fontsize=13, fontweight="bold")

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def create_rouge_bar_comparison(
    results_dict: Dict[str, Dict],
    title: str = "ROUGE Score Comparison",
    output_path: Optional[str] = None,
):
    models = list(results_dict.keys())
    rouge1 = []
    rouge2 = []
    rougeL = []

    for m in models:
        r = results_dict[m]
        if "rouge" in r:
            rouge1.append(r["rouge"].get("rouge1", {}).get("fmeasure", 0))
            rouge2.append(r["rouge"].get("rouge2", {}).get("fmeasure", 0))
            rougeL.append(r["rouge"].get("rougeL", {}).get("fmeasure", 0))
        else:
            rouge1.append(0)
            rouge2.append(0)
            rougeL.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, rouge1, width, label="ROUGE-1", color="#3498DB")
    bars2 = ax.bar(x, rouge2, width, label="ROUGE-2", color="#27AE60")
    bars3 = ax.bar(x + width, rougeL, width, label="ROUGE-L", color="#E74C3C")

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig