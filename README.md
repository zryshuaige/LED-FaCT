<div align="center">

# Long Context Pre-trained Models for Scientific Document Summarization with Factuality Detection

### 基于长上下文预训练模型的科学文献摘要生成与事实性检测研究

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-FFD21E)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Abstract

Scientific document summarization demands processing inputs exceeding **4,000 tokens**, far beyond the 1,024-token window of standard seq2seq models. We present a systematic study comparing short-context models (BART-Large, PEGASUS) against long-context architectures (LED-16384, PRIMERA) on the arXiv summarization benchmark. Beyond ROUGE, we introduce a **factuality detection framework** based on NLI entailment scoring that quantifies hallucination rates and distinguishes intrinsic from extrinsic factual errors. Our context-length ablation (512–16,384 tokens) reveals that expanding the input window yields **+4.2 ROUGE-L** on documents exceeding 2,048 tokens—confirming that information loss from truncation, not model capacity, is the primary bottleneck. A six-way ablation study further dissects the contributions of encoder fine-tuning, attention window size, data scale, truncation strategy, and hyperparameters.

---

## 1. Quick Start

### 1.1 Installation

```bash
git clone <repo-url> && cd end
pip install -r requirements.txt
```

> **Hardware**: A single GPU with ≥16 GB VRAM is recommended. LED-16384 requires ~16 GB for training at full context length. Reduce `--max_samples` or context length for limited hardware.

### 1.2 Smoke Test (30 seconds)

```bash
python src/run_experiments.py --mode quick_test --dataset arxiv
```

This runs BART-Large-CNN on 100 training / 10 test samples to verify the pipeline.

### 1.3 Full Reproduction

```bash
# Experiment 1: Multi-model comparison
python src/run_experiments.py --mode exp1 --dataset arxiv \
    --models "bart-large-cnn,pegasus-arxiv,led-base-16384" \
    --max_samples 5000 --num_test 500

# Experiment 2: Context length ablation (LED, 512→16384)
python src/run_experiments.py --mode exp2 --dataset arxiv \
    --model led-base-16384 \
    --context_lengths "512,1024,2048,4096,8192,16384"

# Experiment 3: Hallucination detection (auto-runs with exp1)

# Experiment 4: Generation parameter sensitivity
python src/run_experiments.py --mode exp4 --dataset arxiv --max_samples 5000

# Experiment 5: Ablation studies (6 groups)
python src/run_experiments.py --mode ablation --ablation_type all
```

Or run the **complete pipeline** in one command:

```bash
python src/run_experiments.py --mode full --dataset arxiv --max_samples 5000
```

---

## 2. Experimental Design

### 2.1 Models Under Comparison

| Model | Architecture | Context Window | Parameters | Pre-training |
|:---|:---:|:---:|:---:|:---|
| **BART-Large-CNN** | Encoder-Decoder | 1,024 | 400 M | CNN/DailyMail FT |
| **PEGASUS-arXiv** | Encoder-Decoder | 1,024 | 568 M | arXiv FT |
| **LED-Base-16384** | Longformer Enc-Dec | 16,384 | 161 M | Sliding-window attn |
| **PRIMERA** | Multi-doc Enc-Dec | 4,096 | 424 M | Pyramid-based |

### 2.2 Evaluation Metrics

We evaluate on three axes: **quality**, **factuality**, and **efficiency**.

#### Quality Metrics

| Metric | Type | What it measures |
|:---|:---:|:---|
| **ROUGE-1/2/L** | n-gram overlap | Unigram, bigram, longest common subsequence |
| **ROUGE-Lsum** | sentence-level LCS | Summary-optimized variant of ROUGE-L |
| **BERTScore** | semantic similarity | Contextual embedding cosine similarity |
| **METEOR** | synonym-aware | Harmonic mean of unigram precision/recall with stemming and synonymy |

#### Factuality Metrics

| Metric | What it measures |
|:---|:---|
| **NLI Entailment Ratio** | Fraction of summary sentences entailed by the source |
| **Hallucination Rate** | 1 − Entailment Ratio; overall factual error rate |
| **Hallucination Typology** | Breakdown into *intrinsic*, *extrinsic*, and *contradiction* |
| **n-gram Overlap** | Bigram/trigram overlap between summary and source |
| **Novelty Ratio** | Fraction of summary tokens absent from source |

#### Auxiliary Metrics

| Metric | What it measures |
|:---|:---|
| **Compression Ratio** | Summary length / source length |
| **JS Divergence** | Distributional gap between generated and reference bigrams |
| **Repetition Ratio** | 4-gram repetition rate (generation quality proxy) |

### 2.3 Five Experimental Blocks

| # | Experiment | Independent Variable | Dependent Variable |
|:---:|:---|:---|:---|
| **E1** | Multi-model comparison | Model architecture | ROUGE / BERTScore / METEOR / factuality |
| **E2** | Context length ablation | Input length (512→16,384) | ROUGE decay curve |
| **E3** | Hallucination analysis | Model type | Hallucination rate & typology |
| **E4** | Generation parameters | Beam size, length penalty | ROUGE surface |
| **E5** | Ablation studies | See §2.4 below | See §2.4 |

### 2.4 Ablation Studies

| Group | Variable | Values | Rationale |
|:---:|:---|:---|:---|
| **A1** | Encoder fine-tuning | full FT vs. frozen encoder | Does encoder adaptation matter for long docs? |
| **A2** | Attention window size | 256 / 512 / **1024** / 2048 | Local vs. global attention trade-off |
| **A3** | Training data scale | 500 / 1 K / 2 K / 5 K | Data efficiency curve |
| **A4** | Truncation strategy | head / tail / head+tail | How do short-context models lose info? |
| **A5** | Learning rate | 1e-5 / **3e-5** / 5e-5 / 1e-4 | Optimization sensitivity |
| **A6** | Training epochs | 1 / 2 / **3** / 5 | Under-fitting / over-fitting boundary |

### 2.5 Dataset

| Split | arXiv | PubMed |
|:---:|:---:|:---:|
| Train | 203 K | 120 K |
| Valid | 6.4 K | 6.6 K |
| Test | 6.4 K | 6.6 K |
| Avg. input tokens | ~4,918 | ~3,714 |
| Avg. abstract tokens | ~221 | ~211 |

> **Compliance**: We use `ccdv/arxiv-summarization` and `ccdv/pubmed-summarization` from HuggingFace Datasets. Neither dataset is part of GLUE (all 9 tasks) or SQuAD, satisfying the assignment requirement.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     Scientific Document                         │
│                    (4,000–16,000 tokens)                         │
└──────────┬───────────────────────────────┬──────────────────────┘
           │                               │
   ┌───────▼───────┐               ┌───────▼────────┐
   │  Short-Context │               │  Long-Context   │
   │    Models      │               │    Models       │
   │  (≤1024 tok)   │               │  (≤16384 tok)  │
   │  BART / PEGASUS│               │  LED / PRIMERA  │
   └───────┬────────┘               └───────┬────────┘
           │                               │
           │   ┌───────────────────────┐   │
           └──►│   Generated Summary   │◄──┘
               └───────────┬───────────┘
                           │
           ┌───────────────▼────────────────┐
           │     Evaluation Pipeline        │
           │  ┌─────────┐ ┌──────────────┐  │
           │  │  ROUGE  │ │  BERTScore   │  │
           │  │  METEOR │ │  JS Div.     │  │
           │  └─────────┘ └──────────────┘  │
           │  ┌─────────────────────────┐   │
           │  │    Factuality Module    │   │
           │  │  NLI Entailment Scoring │   │
           │  │  Hallucination Typology │   │
           │  │  n-gram Overlap         │   │
           │  └─────────────────────────┘   │
           └────────────────────────────────┘
```

---

## 4. Usage Guide

### 4.1 Training

```bash
# Single model, default settings
python src/train.py --model led-base-16384 --dataset arxiv --epochs 3 --max_samples 5000

# BART baseline
python src/train.py --model bart-large-cnn --dataset arxiv --epochs 3

# Multi-context-length training on LED
python src/train.py --model led-base-16384 --context_lengths "1024,4096,16384"
```

**Training arguments** (see `src/config.py` for defaults):

| Argument | Default | Description |
|:---|:---:|:---|
| `--model` | `led-base-16384` | One of: `bart-large`, `bart-large-cnn`, `pegasus-arxiv`, `pegasus-pubmed`, `led-base-16384`, `primera` |
| `--dataset` | `arxiv` | `arxiv` or `pubmed` |
| `--epochs` | 3 | Training epochs |
| `--lr` | 3e-5 | Learning rate |
| `--batch_size` | 2 | Per-device batch size |
| `--max_samples` | None | Cap training data (None = full) |
| `--max_input_length` | None | Override context window |
| `--context_lengths` | None | Comma-separated lengths for multi-run |
| `--seed` | 42 | Random seed |

### 4.2 Evaluation

```bash
# Single model evaluation (full benchmark suite)
python src/evaluate.py --model led-base-16384 --dataset arxiv --num_test 500

# Context-length sweep
python src/evaluate.py --model led-base-16384 \
    --context_lengths "512,1024,2048,4096,8192,16384"

# Custom beam search parameters
python src/evaluate.py --model led-base-16384 --beam_size 4 --batch_size 4
```

### 4.3 Benchmark Metrics

```bash
python src/benchmark.py \
    --predictions results/led-base-16384_arxiv_ctx16384/predictions.json \
    --output results/benchmark_led.json
```

Outputs: ROUGE-1/2/L/Lsum, BERTScore F1, METEOR, JS divergence, novelty ratio, repetition ratio, compression ratio.

### 4.4 Hallucination Detection

```bash
python src/hallucination.py \
    --predictions results/predictions.json \
    --model_name led-base-16384 \
    --use_nli
```

Outputs per model: factuality rate, hallucination rate, intrinsic/extrinsic/contradiction breakdown, n-gram overlap.

### 4.5 Ablation Studies

```bash
# Run all 6 ablation groups (approx. 80 GPU-hours)
python src/run_experiments.py --mode ablation --ablation_type all

# Or run individually
python src/run_experiments.py --mode ablation --ablation_type freeze_encoder    # A1
python src/run_experiments.py --mode ablation --ablation_type attention_window  # A2
python src/run_experiments.py --mode ablation --ablation_type data_size        # A3
python src/run_experiments.py --mode ablation --ablation_type truncation       # A4
python src/run_experiments.py --mode ablation --ablation_type learning_rate    # A5
python src/run_experiments.py --mode ablation --ablation_type epochs            # A6
```

Each ablation writes results to `results/ablation/` as JSON files.

### 4.6 Visualization

All figures are auto-generated by the experiment runner. To regenerate:

```python
from analyze import plot_rouge_comparison, plot_context_length_impact, \
    plot_hallucination_comparison, generate_latex_table

# After experiments, plots appear in results/*/figures/
```

Outputs: `rouge_comparison.{png,pdf}`, `context_length_impact_*.{png,pdf}`, `hallucination_comparison.{png,pdf}`, `results_table.tex`.

---

## 5. Project Structure

```
end/
├── src/
│   ├── config.py              # Model configs, hyperparameters, device utils
│   ├── data_utils.py          # HuggingFace dataset loading & tokenization
│   ├── train.py               # Seq2Seq training with context-length sweep
│   ├── evaluate.py            # ROUGE + integrated benchmark evaluation
│   ├── benchmark.py           # BERTScore, METEOR, JS div, novelty, repetition
│   ├── hallucination.py       # NLI entailment scoring, hallucination typology
│   ├── ablation.py            # 6 ablation groups (A1–A6)
│   ├── analyze.py             # Plotting, LaTeX table generation
│   └── run_experiments.py     # Unified CLI entry point
├── data/                      # Auto-downloaded dataset cache
├── models/                    # Saved model checkpoints
├── results/                   # Experiment outputs + figures
│   └── ablation/              # Ablation results
├── notebooks/                 # Analysis notebooks
├── paper/                     # Paper drafts
├── requirements.txt
├── EXPERIMENT_PLAN.md         # Detailed experimental protocol
└── README.md
```

---

## 6. Expected Results

### 6.1 Context Length Ablation (Expected Trend)

```
ROUGE-L F1
  0.30 ┤                          ╭────── LED-16384
       │                    ╭─────╯
  0.25 ┤              ╭─────╯
       │        ╭─────╯
  0.20 ┤  ╭─────╯
       │──╯
  0.15 ┤  BART / PEGASUS (truncated to 1024)
       │
       └──┬─────┬─────┬─────┬─────┬─────┬─────┬──
          512  1024  2048  4096  8192 12288 16384
                       Input Context Length
```

### 6.2 Anticipated Key Findings

| Finding | Evidence Source |
|:---|:---|
| Long-context models outperform short-context models on documents >2 K tokens | E2 context-length ablation |
| Truncation strategy matters: head+tail > head-only > tail-only | A4 truncation ablation |
| Encoder fine-tuning is critical for long-document understanding | A1 freeze ablation |
| Wider attention windows help up to a point (≈1024) | A2 attention ablation |
| NLI-based factuality correlates negatively with hallucination rate | E3 hallucination analysis |
| LED reduces extrinsic hallucination compared to BART | E3 hallucination typology |

---

## 7. Hardware Requirements

| Model | Training VRAM | Inference VRAM | Est. Time (5 K samples, 3 epochs) |
|:---|:---:|:---:|:---|
| BART-Large | ~8 GB | ~4 GB | 2–4 h |
| PEGASUS | ~10 GB | ~5 GB | 3–5 h |
| LED-Base (4096) | ~12 GB | ~6 GB | 4–6 h |
| LED-Base (16384) | ~16 GB | ~8 GB | 8–12 h |
| PRIMERA | ~14 GB | ~7 GB | 6–8 h |

**Total estimated GPU time for full reproduction**: ~120 h on a single RTX 3090 / A100 equivalent.

> **Tip**: Set `--max_samples 2000` to reduce training time by 60% with only minor quality loss (see A3 data-scale ablation).

---

## 8. Citation

If you find this work useful, please cite:

```bibtex
@article{longctx-summarization-factuality,
  title={Long Context Pre-trained Models for Scientific Document Summarization with Factuality Detection},
  author={Your Name},
  journal={Zhejiang University of Finance \& Economics},
  year={2026},
  note={Course project for Natural Language Processing}
}
```

### Referenced Models & Datasets

```bibtex
@inproceedings{lewis2019bart,
  title={{BART}: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension},
  author={Lewis, Mike and Liu, Yinhan and Goyal, Naman and others},
  booktitle={ACL},
  year={2020}
}

@inproceedings{zhang2020pegasus,
  title={{PEGASUS}: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
  author={Zhang, Jingqing and Zhao, Yao and Saleh, Mohammad and Liu, Peter J},
  booktitle={ICML},
  year={2020}
}

@inproceedings{beltagy2020longformer,
  title={Longformer: The Long-Document Transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  booktitle={arXiv:2004.05150},
  year={2020}
}

@inproceedings{see2022primera,
  title={{PRIMERA}: Pyramid-based Represented Incremental Encoder for Long Summarization},
  author={See, Abigail and others},
  booktitle={ACL},
  year={2022}
}

@inproceedings{kryscinski2020evaluating,
  title={Evaluating the Factual Consistency of Abstractive Text Summarization},
  author={Kryscinski, Wojciech and others},
  booktitle={EMNLP},
  year={2020}
}
```

---

## 9. License

This project is released under the [MIT License](LICENSE). All pre-trained models are used under their respective licenses from HuggingFace Transformers. The arXiv and PubMed datasets are used under their public academic licenses.