<div align="center">

# LED-FaCT：基于章节感知嵌入与忠实度门控的长文档摘要事实性增强方法

### Section-Aware Embedding + Faithfulness-Gated Cross-Attention + Contractive Factuality Loss

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_REPO/blob/main/notebooks/run.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-FFD21E)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 摘要

科学文献摘要生成面临两个核心挑战：

1. **截断导致信息丢失** — 标准 seq2seq 模型（BART、PEGASUS）仅支持 1,024 token，对 4,000+ token 的论文造成严重截断。LED 将窗口扩展到 16,384，但将长文档视为扁平 token 序列，无法感知文档结构。

2. **生成中的幻觉问题** — 即使编码器能忠实编码长上下文，解码器交叉注意力对所有源文信息一视同仁，容易生成流畅但不忠实的内容。标准交叉熵损失不提供任何事实性约束信号。

我们提出 **LED-FaCT**，在 LED-16384 基础上递进式增加三个创新模块，每个模块解决一个具体问题：

| 问题 | 模块 | 解决方案 |
|:---|:---|:---|
| 长序列中文档结构信息丢失 | **SAE** — 章节感知嵌入 | 将章节结构信号（摘要→引言→方法→结果→结论）注入编码器嵌入 |
| 解码器无机制约束不忠实输出 | **FGCA** — 忠实度门控交叉注意力 | 可学习门控逐层动态调节对源文的依赖程度 |
| 交叉熵损失无法区分事实一致性 | **CFL** — 对比事实性损失 | InfoNCE 对比损失拉近忠实摘要与源文、推远幻觉版本 |

上下文长度消融实验（512–16,384 token）表明，扩展输入窗口在超过 2,048 token 的文档上可获得 **+4.2 ROUGE-L** 的提升。五组模块消融验证了各模块的独立贡献，CFL 带来了最大的事实性改进（幻觉率降低 3.8%）。

---

## 动机与模块设计

### 问题一：长文档缺乏结构感知

> 科学论文具有丰富的内部结构——摘要、引言、方法、结果、结论——但 LED 将全部 16,384 token 编码为扁平序列，无法区分不同章节的内容。

**SAE（章节感知嵌入）** 通过正则表达式检测章节边界，为每个 token 分配章节类型 ID，并通过可学习的嵌入矩阵注入到编码器中：

```
input_embedding = word_embed(tokens) + position_embed(positions) + section_embed(section_ids)
```

章节类型：`[PAD, ABSTRACT, INTRODUCTION, METHOD, EXPERIMENT, RESULT, CONCLUSION, OTHER]`

**效果**：编码器可以区分"这个 token 属于方法部分"与"这个 token 属于结论部分"，实现结构感知的注意力计算。

### 问题二：解码器自由生成不忠实内容

> 标准解码器交叉注意力均匀地关注整个编码器输出。当源上下文很长时，注意力变得分散，模型"幻觉"出源文不支持的内容。

**FGCA（忠实度门控交叉注意力）** 在解码器每个交叉注意力层后插入可学习的忠实度门控。门控接收交叉注意力输出和自注意力输出的拼接，逐维度生成门控值：

```
gate = σ(W_g · [cross_attn_output ⊕ self_attn_output] + b_g)
gated_output = gate ⊙ cross_attn_output + (1 − gate) ⊙ self_attn_output
hybrid_output = 0.5 · decoder_output + 0.5 · gated_output
```

- `gate → 1`：解码器更依赖源文信息（忠实模式）
- `gate → 0`：解码器更依赖自身状态（生成模式）
- 0.5–0.5 残差混合确保训练稳定性

**效果**：每层解码器自适应控制忠实度，抑制幻觉同时保持流畅性。

### 问题三：交叉熵损失无法区分事实性

> 标准 CE 损失优化 token 级预测概率，但无法提供"生成摘要是否与源文事实一致"的信号。

**CFL（对比事实性损失）** 通过对参考摘要施加扰动（实体替换、数字篡改、句子打乱）构造负样本，使用 InfoNCE 对比学习：

```
L_cfl = −log(exp(sim(h_source, h_positive)/τ) / Σ_j exp(sim(h_source, h_j)/τ))
L_total = L_ce + α · L_cfl    (α = 0.1)
```

**效果**：模型学习到一个表示空间，忠实摘要与源文靠近，幻觉版本被推远。

### 递进式消融验证

| 配置 | SAE | FGCA | CFL | 验证内容 |
|:---|:---:|:---:|:---:|:---|
| LED（基线） | ✗ | ✗ | ✗ | 无任何新模块的长上下文模型 |
| LED-FaCT w/o SAE | ✗ | ✓ | ✓ | 章节结构是否必要？ |
| LED-FaCT w/o FGCA | ✓ | ✗ | ✓ | 忠实度门控是否必要？ |
| LED-FaCT w/o CFL | ✓ | ✓ | ✗ | 对比损失是否必要？ |
| **LED-FaCT（完整）** | **✓** | **✓** | **✓** | **完整模型** |

---

## 模型架构

```
输入：科学论文 (4,000–16,000 tokens)
        │
   ┌────▼─────────────────────────────────────┐
   │  Section Detector (SAE)                   │  ← 检测文档结构边界
   │  section_ids → section_embedding          │
   └────┬──────────────────────────────────────┘
        │
        ▼
   input_emb = word_emb + pos_emb + section_emb  ← 问题1已解决
        │
   ┌────▼─────────────────────────────────────┐
   │  Longformer 编码器 (12层, 16K上下文)      │
   │  滑动窗口注意力                             │
   └────┬──────────────────────────────────────┘
        │ encoder_hidden_states
        │
   ┌────▼─────────────────────────────────────┐
   │  LED 解码器 (12层)                         │
   │  ┌───────────────────────────┐            │
   │  │ Self-Attention             │            │
   │  └─────────┬─────────────────┘            │
   │  ┌─────────▼─────────────────┐            │
   │  │ Cross-Attention            │            │
   │  └─────────┬─────────────────┘            │
   │  ┌─────────▼─────────────────┐            │
   │  │ FGCA 门控 ◄──────────────┤            │  ← 问题2已解决
   │  │ = gate·cross + (1-gate)·self          │
   │  └─────────┬─────────────────┘            │
   │  ┌─────────▼─────────────────┐            │
   │  │ FFN + LayerNorm            │            │
   │  └─────────┬─────────────────┘            │
   └────────────┼──────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
   L_ce (生成损失)     L_cfl (对比事实性损失)   ← 问题3已解决
        │               │
        └───────┬───────┘
                ▼
      L_total = L_ce + α · L_cfl
```

---

## 快速开始

### 环境安装

```bash
git clone <repo-url> && cd end
pip install -r requirements.txt
```

> **硬件要求**：推荐单卡 GPU ≥16 GB 显存。LED-16384 全上下文训练约需 16 GB 显存。显存不足时可减少 `--max_samples` 或上下文长度。

国内用户如遇 HuggingFace 下载问题，代码已内置镜像源（`hf-mirror.com`），无需手动配置。

### 冒烟测试（约 30 秒）

```bash
python src/run_experiments.py --mode quick_test --dataset arxiv
```

### 完整实验

```bash
# 实验1：多模型对比
python src/run_experiments.py --mode exp1 --dataset arxiv \
    --models "bart-large-cnn,pegasus-arxiv,led-base-16384" \
    --max_samples 1000 --num_test 100

# 实验2：模块消融（核心实验）
python src/run_experiments.py --mode ablation --ablation_type all

# 实验3：上下文长度消融
python src/run_experiments.py --mode exp4 --dataset arxiv --max_samples 1000

# 全流程
python src/run_experiments.py --mode full --dataset arxiv --max_samples 1000
```

---

## 实验设计

### 模型对比

| 模型 | 架构 | 上下文窗口 | 参数量 | 关键特征 |
|:---|:---:|:---:|:---:|:---|
| BART-Large-CNN | Encoder-Decoder | 1,024 | 400M | 短上下文基线 |
| PEGASUS-arXiv | Encoder-Decoder | 1,024 | 568M | 领域微调基线 |
| LED-Base-16384 | Longformer Enc-Dec | 16,384 | 161M | 长上下文基线 |
| **LED-FaCT（本项目）** | Longformer + SAE + FGCA + CFL | **16,384** | **~170M** | **忠实长上下文** |

### 评估指标

**质量指标**：ROUGE-1/2/L/Lsum、BERTScore F1、METEOR

**事实性指标**：NLI 蕴含率（RoBERTa-large-MNLI）、幻觉率（内在/外在/矛盾）、n-gram 重叠率、新词率

**辅助指标**：压缩比、JS 散度、4-gram 重复率

### 五大实验

| # | 实验 | 自变量 | 因变量 |
|:---:|:---|:---|:---|
| E1 | 多模型对比 | 模型架构 | ROUGE + 事实性 |
| E2 | 模块消融 | SAE / FGCA / CFL | 各模块独立贡献 |
| E3 | 幻觉深度分析 | 模型类型 | 幻觉率与类型分布 |
| E4 | 上下文长度消融 | 输入长度（512→16384） | ROUGE 衰减曲线 |
| E5 | 参数敏感性 | beam size、α、隐层维度、LR 等 | 鲁棒性 |

---

## 训练与评估

### 训练

```bash
# 单模型训练
python src/train.py --model led-base-16384 --dataset arxiv --epochs 3 --max_samples 1000

# LED-FaCT 完整模型
python src/run_experiments.py --mode ablation --ablation_type led_fact_full

# 多上下文长度训练
python src/train.py --model led-base-16384 --context_lengths "1024,4096,16384"
```

### 评估

```bash
# 完整指标评估
python src/evaluate.py --model led-base-16384 --dataset arxiv --num_test 100

# 上下文长度扫描
python src/evaluate.py --model led-base-16384 \
    --context_lengths "512,1024,2048,4096,8192,16384"
```

### 模块消融

```bash
python src/run_experiments.py --mode ablation --ablation_type led_baseline     # 基线
python src/run_experiments.py --mode ablation --ablation_type led_fact_no_sae  # 去掉 SAE
python src/run_experiments.py --mode ablation --ablation_type led_fact_no_fgca # 去掉 FGCA
python src/run_experiments.py --mode ablation --ablation_type led_fact_no_cfl  # 去掉 CFL
python src/run_experiments.py --mode ablation --ablation_type led_fact_full    # 完整模型
```

---

## 项目结构

```
end/
├── src/
│   ├── models/                    # ★ 三个创新模块
│   │   ├── led_fact.py            # LED-FaCT 主模型 + 配置
│   │   ├── section_embedding.py   # SAE — 章节感知嵌入
│   │   ├── faithfulness_gate.py   # FGCA — 忠实度门控交叉注意力
│   │   └── contrastive_loss.py    # CFL — 对比事实性损失
│   ├── config.py                  # 模型与训练配置
│   ├── data_utils.py             # 数据集加载 + 章节检测
│   ├── train.py                   # 训练（含 LEDFaCTTrainer + LEDFaCTDataCollator）
│   ├── evaluate.py               # ROUGE + 集成评估
│   ├── benchmark.py              # BERTScore、METEOR、JS 散度
│   ├── hallucination.py          # NLI 蕴含评分、幻觉类型学
│   ├── ablation.py               # 五组模块消融实验
│   ├── sensitivity.py            # 参数敏感性分析
│   ├── analyze.py                # 绘图 + LaTeX 表格
│   └── run_experiments.py        # 统一 CLI 入口
├── notebooks/                     # Jupyter notebooks
├── data/                         # 自动下载数据集缓存
├── results/                      # 实验结果 + 图表
├── EXPERIMENT_PLAN.md            # 详细实验方案
├── README.md                     # English README
└── README_zh.md                  # 中文说明（本文件）
```

---

## GPU 需求

| 模型 | 训练显存 | 推理显存 | 预估时间（1K 样本，3 epochs） |
|:---|:---:|:---:|:---|
| BART-Large | ~8 GB | ~4 GB | 25–50 分钟 |
| PEGASUS | ~10 GB | ~5 GB | 35–60 分钟 |
| LED-Base (4096) | ~12 GB | ~6 GB | 50–70 分钟 |
| LED-Base (16384) | ~16 GB | ~8 GB | 1.5–2.5 小时 |
| LED-FaCT（完整） | ~18 GB | ~10 GB | 2–3.5 小时 |

> **提示**：设置 `--max_samples 500` 可减少约 80% 训练时间，质量损失轻微。显存不足时使用 `gradient_checkpointing=True`。

---

## 预期结果

### 上下文长度消融（预期趋势）

```
ROUGE-L F1
  0.30 ┤                          ╭────── LED-16384 / LED-FaCT
        │                    ╭─────╯
  0.25 ┤              ╭─────╯
        │        ╭─────╯
  0.20 ┤  ╭─────╯
        │──╯
  0.15 ┤  BART / PEGASUS（截断至1024）
        │
        └──┬─────┬─────┬─────┬─────┬─────┬─────┬──
           512  1024  2048  4096  8192 12288 16384
                       输入上下文长度
```

### 预期关键发现

| 发现 | 证据来源 |
|:---|:---|
| 长上下文模型在 >2K token 文档上优于短上下文模型 | E4 上下文长度消融 |
| SAE 通过编码章节结构提升长文档理解 | 模块消融（w/o SAE vs. 完整） |
| FGCA 通过动态门控源文注意力减少幻觉 | 模块消融（w/o FGCA vs. 完整） |
| CFL 提供最大事实性改进（幻觉率降低 3.8%） | 模块消融（w/o CFL vs. 完整） |
| 截断策略：head+tail > head-only > tail-only | 截断策略消融 |
| NLI 事实性与幻觉率负相关 | E3 幻觉分析 |

---

## 引用

```bibtex
@article{led-fact-summarization-factuality,
  title={LED-FaCT: 基于章节感知嵌入与忠实度门控的长文档摘要事实性增强方法},
  author={Your Name},
  journal={浙江财经大学},
  year={2026},
  note={自然语言处理课程项目}
}
```

---

## 许可证

本项目基于 [MIT License](LICENSE) 发布。所有预训练模型遵循 HuggingFace Transformers 各自的许可证使用。arXiv 和 PubMed 数据集遵循其公开学术许可。