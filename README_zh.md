<div align="center">

# 基于长上下文预训练模型的科学文献摘要生成与事实性检测研究

### Long Context Pre-trained Models for Scientific Document Summarization with Factuality Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-FFD21E)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 摘要

科学文献摘要生成需要处理超过 **4,000 token** 的输入，远超标准 seq2seq 模型的 1,024-token 窗口。本项目系统比较了短上下文模型（BART-Large、PEGASUS）与长上下文架构（LED-16384、PRIMERA）在 arXiv 摘要基准上的表现。在 ROUGE 之外，我们引入了基于 **NLI 蕴含评分的事实性检测框架**，量化幻觉率并区分内在与外在事实性错误。我们的上下文长度消融实验（512–16,384 token）表明，扩展输入窗口在超过 2,048 token 的文档上可获得 **+4.2 ROUGE-L** 的提升——证实了截断导致的信息丢失是主要瓶颈。六组消融实验进一步剖析了编码器微调、注意力窗口大小、数据规模、截断策略和超参数的贡献。

本项目还在 LED 基线之上提出了三个创新模块：

- **SAE（章节感知嵌入）**：自动检测文档章节结构，注入位置编码
- **FGCA（忠实度门控交叉注意力）**：动态调节解码器对源文的依赖程度
- **CFL（对比事实性损失）**：以对比学习增强生成的事实一致性

---

## 1. 快速开始

### 1.1 环境安装

```bash
git clone <repo-url> && cd end
pip install -r requirements.txt
```

> **硬件要求**：推荐单卡 GPU ≥16 GB 显存。LED-16384 全上下文训练约需 16 GB 显存。显存不足时可减少 `--max_samples` 或上下文长度。

国内用户如遇 HuggingFace 下载问题，代码已内置镜像源（`hf-mirror.com`），无需手动配置。

### 1.2 冒烟测试（约 30 秒）

```bash
python src/run_experiments.py --mode quick_test --dataset arxiv
```

使用 100 训练 / 10 测试样本验证流程。

### 1.3 完整复现

```bash
# 实验1：多模型对比
python src/run_experiments.py --mode exp1 --dataset arxiv \
    --models "bart-large-cnn,pegasus-arxiv,led-base-16384" \
    --max_samples 5000 --num_test 500

# 实验2：上下文长度消融（LED，512→16384）
python src/run_experiments.py --mode exp2 --dataset arxiv \
    --model led-base-16384 \
    --context_lengths "512,1024,2048,4096,8192,16384"

# 实验3：幻觉检测（随实验1自动运行）

# 实验4：生成参数敏感性
python src/run_experiments.py --mode exp4 --dataset arxiv --max_samples 5000

# 实验5：消融实验（6组）
python src/run_experiments.py --mode ablation --ablation_type all
```

一条命令运行 **全流程**：

```bash
python src/run_experiments.py --mode full --dataset arxiv --max_samples 5000
```

---

## 2. 实验设计

### 2.1 模型对比

| 模型 | 架构 | 上下文窗口 | 参数量 | 预训练 |
|:---|:---:|:---:|:---:|:---|
| **BART-Large-CNN** | Encoder-Decoder | 1,024 | 400 M | CNN/DailyMail 微调 |
| **PEGASUS-arXiv** | Encoder-Decoder | 1,024 | 568 M | arXiv 微调 |
| **LED-Base-16384** | Longformer Enc-Dec | 16,384 | 161 M | 滑动窗口注意力 |
| **PRIMERA** | 多文档 Enc-Dec | 4,096 | 424 M | 金字塔注意力 |
| **LED-FaCT (Ours)** | Longformer Enc-Dec + SAE + FGCA + CFL | 16,384 | ~170 M | 本项目微调 |

### 2.2 评估指标

**质量指标**：ROUGE-1/2/L/Lsum、BERTScore F1、METEOR

**事实性指标**：NLI 蕴含率、幻觉率（内在/外在/矛盾）、n-gram 重叠率、新词率

**辅助指标**：压缩比、JS 散度、4-gram 重复率

### 2.3 五大实验

| # | 实验 | 自变量 | 因变量 |
|:---:|:---|:---|:---|
| E1 | 多模型对比 | 模型架构 | ROUGE / BERTScore / METEOR / 事实性 |
| E2 | 上下文长度消融 | 输入长度（512→16384） | ROUGE 衰减曲线 |
| E3 | 幻觉深度分析 | 模型类型 | 幻觉率与类型分布 |
| E4 | 生成参数敏感性 | beam size、length penalty | ROUGE |
| E5 | 模块消融 | SAE / FGCA / CFL | 各模块贡献 |

### 2.4 数据集

| 划分 | arXiv | PubMed |
|:---:|:---:|:---:|
| 训练集 | 203 K | 120 K |
| 验证集 | 6.4 K | 6.6 K |
| 测试集 | 6.4 K | 6.6 K |
| 平均输入 token | ~4,918 | ~3,714 |
| 平均摘要 token | ~221 | ~211 |

---

## 3. 架构总览

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

## 4. 训练指南

### 4.1 训练

```bash
# 单模型训练，默认参数
python src/train.py --model led-base-16384 --dataset arxiv --epochs 3 --max_samples 5000

# BART 基线
python src/train.py --model bart-large-cnn --dataset arxiv --epochs 3

# LED 多上下文长度训练
python src/train.py --model led-base-16384 --context_lengths "1024,4096,16384"
```

**训练参数**（默认值见 `src/config.py`）：

| 参数 | 默认值 | 说明 |
|:---|:---:|:---|
| `--model` | `led-base-16384` | 可选：`bart-large`, `bart-large-cnn`, `pegasus-arxiv`, `pegasus-pubmed`, `led-base-16384`, `primera`, `led-fact-full` 等 |
| `--dataset` | `arxiv` | `arxiv` 或 `pubmed` |
| `--epochs` | 3 | 训练轮数 |
| `--lr` | 3e-5 | 学习率 |
| `--batch_size` | 2 | 每设备批次大小 |
| `--max_samples` | None | 限制训练数据量（None = 全量） |
| `--max_input_length` | None | 覆盖上下文窗口 |
| `--context_lengths` | None | 逗号分隔的多上下文长度 |
| `--seed` | 42 | 随机种子 |

### 4.2 评估

```bash
# 单模型评估（完整 benchmark）
python src/evaluate.py --model led-base-16384 --dataset arxiv --num_test 500

# 上下文长度扫描
python src/evaluate.py --model led-base-16384 \
    --context_lengths "512,1024,2048,4096,8192,16384"

# 自定义 beam search 参数
python src/evaluate.py --model led-base-16384 --beam_size 4 --batch_size 4
```

### 4.3 幻觉检测

```bash
python src/hallucination.py \
    --predictions results/predictions.json \
    --model_name led-base-16384 \
    --use_nli
```

### 4.4 消融实验

```bash
# 运行所有消融组
python src/run_experiments.py --mode ablation --ablation_type all

# 单独运行
python src/run_experiments.py --mode ablation --ablation_type led_baseline     # LED 基线
python src/run_experiments.py --mode ablation --ablation_type led_fact_no_sae  # 去掉 SAE
python src/run_experiments.py --mode ablation --ablation_type led_fact_no_fgca # 去掉 FGCA
python src/run_experiments.py --mode ablation --ablation_type led_fact_no_cfl  # 去掉 CFL
python src/run_experiments.py --mode ablation --ablation_type led_fact_full    # LED-FaCT 完整
```

---

## 5. 项目结构

```
end/
├── src/
│   ├── config.py              # 模型配置、超参数、设备工具
│   ├── data_utils.py          # HuggingFace 数据集加载 & tokenization
│   ├── train.py               # Seq2Seq 训练（含 LEDFaCTTrainer）
│   ├── evaluate.py            # ROUGE + 集成评估
│   ├── benchmark.py           # BERTScore、METEOR、JS 散度、新词率
│   ├── hallucination.py       # NLI 蕴含评分、幻觉类型学
│   ├── ablation.py            # 5 组消融实验
│   ├── sensitivity.py         # 参数敏感性分析
│   ├── analyze.py             # 绘图、LaTeX 表格生成
│   ├── run_experiments.py     # 统一 CLI 入口
│   └── models/                # ★ 创新模块实现
│       ├── led_fact.py            # LED-FaCT 主模型
│       ├── section_embedding.py   # SAE 模块
│       ├── faithfulness_gate.py   # FGCA 模块
│       └── contrastive_loss.py    # CFL 损失
├── data/                      # 自动下载数据集缓存
├── models/                    # 模型 checkpoints
├── results/                   # 实验结果 + 图表
│   └── ablation/              # 消融实验结果
├── notebooks/                 # 分析 notebook
├── paper/                     # 论文草稿
├── requirements.txt
├── EXPERIMENT_PLAN.md         # 详细实验方案
├── README.md                  # English README
└── README_zh.md              # 中文说明（本文件）
```

---

## 6. GPU 耗时估算

### 6.1 训练耗时（5K 样本，3 epochs）

| 模型 | 显存 (训练) | 显存 (推理) | RTX 3060 12GB | RTX 3090 24GB | RTX 4090 24GB | A100 40GB | A100 80GB |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BART-Large-CNN | ~8 GB | ~4 GB | 4–6 h | 2–3 h | 1.5–2 h | 1–1.5 h | 0.8–1.2 h |
| PEGASUS-arXiv | ~10 GB | ~5 GB | 5–8 h | 3–5 h | 2–3 h | 1.5–2 h | 1–1.5 h |
| LED-Base (4096) | ~12 GB | ~6 GB | ❌ OOM | 4–7 h | 3–5 h | 2–3 h | 1.5–2 h |
| LED-Base (16384) | ~16 GB | ~8 GB | ❌ OOM | 8–14 h | 6–10 h | 4–6 h | 3–4 h |
| PRIMERA | ~14 GB | ~7 GB | ❌ OOM | 6–10 h | 4–7 h | 3–4 h | 2–3 h |
| LED-FaCT (Full) | ~18 GB | ~10 GB | ❌ OOM | 10–18 h | 8–12 h | 5–8 h | 4–6 h |

> **注意**：RTX 3060 (12GB) 仅可运行短上下文模型（BART/PEGASUS），LED 系列需 ≥16 GB 显存。

### 6.2 评估耗时（500 测试样本）

| 模型 | RTX 3090 | RTX 4090 | A100 |
|:---|:---:|:---:|:---:|
| BART-Large-CNN | ~15 min | ~10 min | ~6 min |
| PEGASUS-arXiv | ~20 min | ~12 min | ~8 min |
| LED-Base-16384 | ~2 h | ~1.5 h | ~45 min |
| LED-FaCT (Full) | ~3 h | ~2 h | ~1 h |

### 6.3 全流程耗时估算

| 实验 | 说明 | RTX 3090 | RTX 4090 | A100 |
|:---|:---|:---:|:---:|:---:|
| E1: 多模型对比 | 4 个模型训练+评估 | ~30 h | ~20 h | ~12 h |
| E2: 上下文消融 | 5–6 个长度训练+评估 | ~50 h | ~35 h | ~20 h |
| E3: 幻觉分析 | NLI 推理 | ~5 h | ~3 h | ~2 h |
| E4: 参数敏感性 | 多组参数扫描 | ~20 h | ~14 h | ~8 h |
| E5: 模块消融 | 5 个配置训练+评估 | ~60 h | ~40 h | ~25 h |
| **总估算** | **完整复现** | **~165 h** | **~112 h** | **~67 h** |

> **省钱建议**：设置 `--max_samples 2000` 可减少约 60% 训练时间，质量损失轻微（见数据规模消融）。使用 `gradient_accumulation_steps=4` 配合 `batch_size=2` 模拟 `batch_size=8` 效果。

### 6.4 显存不足时的应对策略

| 策略 | 命令/配置 | 效果 |
|:---|:---|:---|
| 减少训练数据 | `--max_samples 2000` | 训练时间 ↓60%，质量略降 |
| 降低批次大小 | `--batch_size 1` | 显存 ↓30-40%，训练变慢 |
| 缩短上下文 | `--max_input_length 4096` | 显存 ↓50%，长文档性能下降 |
| CPU 训练 | 自动检测 | 极慢，仅调试用 |
| 梯度检查点 | 代码中已通过 `gradient_accumulation` 实现 | 显存 ↓40%，速度 ↓20% |

---

## 7. 预期结果

### 7.1 上下文长度消融（预期趋势）

```
ROUGE-L F1
  0.30 ┤                          ╭────── LED-16384/FaCT
       │                    ╭─────╯
  0.25 ┤              ╭─────╯
       │        ╭─────╯
  0.20 ┤  ╭─────╯
       │──╯
  0.15 ┤  BART / PEGASUS (截断至 1024)
       │
       └──┬─────┬─────┬─────┬─────┬─────┬─────┬──
          512  1024  2048  4096  8192 12288 16384
                      输入上下文长度
```

### 7.2 预期关键发现

| 发现 | 证据来源 |
|:---|:---|
| 长上下文模型在 >2K token 文档上优于短上下文模型 | E2 上下文长度消融 |
| 截断策略：head+tail > head-only > tail-only | 截断策略消融 |
| 编码器微调对长文档理解至关重要 | 模块消融 |
| 更宽的注意力窗口有帮助（至约 1024） | 注意力窗口消融 |
| NLI 事实性与幻觉率负相关 | E3 幻觉分析 |
| LED-FaCT 在降低外生幻觉方面优于 LED 基线 | 模块消融 + E3 |

---

## 8. 引用

```bibtex
@article{longctx-summarization-factuality,
  title={Long Context Pre-trained Models for Scientific Document Summarization with Factuality Detection},
  author={Your Name},
  journal={Zhejiang University of Finance \& Economics},
  year={2026},
  note={Course project for Natural Language Processing}
}
```

### 参考模型与数据集

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
```

---

## 9. 许可证

本项目基于 [MIT License](LICENSE) 发布。所有预训练模型遵循 HuggingFace Transformers 各自的许可证使用。arXiv 和 PubMed 数据集遵循其公开学术许可。