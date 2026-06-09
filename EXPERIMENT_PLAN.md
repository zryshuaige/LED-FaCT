# LED-FaCT: 长文档摘要生成的事实性增强方法

## 论文题目

**中文**: LED-FaCT: 基于章节感知嵌入与忠实度门控的长文档摘要事实性增强方法

**English**: LED-FaCT: Faithfulness-Enhanced Long Document Summarization with Section-Aware Embedding and Faithfulness-Gated Cross-Attention

---

## 核心创新点

在LED (Longformer Encoder-Decoder) 基线模型上，提出三个创新模块：

### 模块1: Section-Aware Embedding (SAE) — 章节感知嵌入

**动机**: 科学文献有明确的章节结构（摘要→引言→方法→实验→结论），但传统模型将全文视为扁平token序列，无法感知文档结构。

**方法**: 对输入文本自动检测章节边界，为每个token分配章节类型ID，通过可学习的section embedding矩阵将章节信息注入编码器：

```
input_embedding = word_embed(token_ids) + position_embed(position_ids) + section_embed(section_ids)
```

**章节类型**: `[PAD, ABSTRACT, INTRODUCTION, METHOD, EXPERIMENT, RESULT, CONCLUSION, OTHER]`

### 模块2: Faithfulness-Guided Cross-Attention (FGCA) — 忠实度门控交叉注意力

**动机**: 生成式摘要模型容易产生幻觉内容，即生成与原文不一致的信息。现有方法在解码时对所有源文信息一视同仁，缺少对忠实度的显式控制。

**方法**: 在解码器每个交叉注意力层后，插入一个可学习的忠实度门控（Faithfulness Gate），动态调节解码状态对源文信息的依赖：

```
gate = σ(W_g · [cross_attn_output ⊕ self_attn_output] + b_g)
output = gate ⊙ cross_attn_output + (1 - gate) ⊙ self_attn_output
```

当门控值接近1时，解码器更依赖源文信息（忠实）；接近0时，允许模型自主生成（灵活性）。

### 模块3: Contrastive Factuality Loss (CFL) — 对比事实性损失

**动机**: 标准交叉熵损失只优化生成概率，不直接约束事实一致性。需要有显式的训练信号让模型区分忠实摘要与非忠实摘要。

**方法**: 训练时对每个参考摘要构造"幻觉版本"（替换实体、篡改数字、打乱句子），使用InfoNCE对比损失拉近忠实摘要与源文表示、推远幻觉版本：

```
L_cfl = -log(exp(sim(h_src, h_pos)/τ) / Σ_j exp(sim(h_src, h_j)/τ))
L_total = L_ce + α · L_cfl    (α=0.1)
```

---

## 模型架构图

```
Input: Scientific Paper
       │
       ▼
┌─────────────────────────────┐
│   Section Detection (SAE)    │  ← 自动识别章节边界
│   section_ids → section_emb  │
└─────────────┬───────────────┘
              │
              ▼
input_emb = word_emb + pos_emb + section_emb
              │
              ▼
┌─────────────────────────────┐
│   LED Encoder               │  ← Longformer滑动窗口注意力
│   (12 layers, max 16384)    │
└─────────────┬───────────────┘
              │ encoder_hidden_states
              ▼
┌─────────────────────────────┐
│   LED Decoder (12 layers)   │
│   ┌─────────────────────┐   │
│   │ Self-Attention       │   │
│   └──────────┬──────────┘   │
│              │              │
│   ┌──────────▼──────────┐   │
│   │ Cross-Attention     │   │  ← 标准编码器-解码器注意力
│   └──────────┬──────────┘   │
│              │              │
│   ┌──────────▼──────────┐   │
│   │ FGCA Gate ◄─────────┼───┼─ σ(W_g·[cross_attn ⊕ self_attn] + b)
│   │ = gate·cross +      │   │
│   │   (1-gate)·self     │   │
│   └──────────┬──────────┘   │  ← 关键创新: 忠实度门控
│              │              │
│   ┌──────────▼──────────┐   │
│   │ FFN + LayerNorm     │   │
│   └──────────┬──────────┘   │
└──────────────┼──────────────┘
               │ decoder_output
               ▼
┌──────────────────────────────┐
│   Projection Head (for CFL)  │  ← 用于对比学习的投影头
│   h = MeanPool(decoder_out)  │
└──────────────┬───────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
  L_ce (生成损失)  L_cfl (对比损失)
       │               │
       └───────┬───────┘
               ▼
     L_total = L_ce + α·L_cfl
```

---

## 实验设计

### 实验1: 多模型对比

| 模型 | 上下文长度 | 参数量 | 说明 |
|------|-----------|--------|------|
| BART-Large-CNN | 1024 | 400M | 短上下文基线 |
| PEGASUS-arXiv | 1024 | 568M | 摘要专用基线 |
| LED-Base-16384 | 16384 | 161M | 长上下文基线（我们的骨架） |
| **LED-FaCT (Ours)** | **16384** | **~170M** | **LED + SAE + FGCA + CFL** |

**评估指标**: ROUGE-1/2/L + ROUGE-Lsum + BERTScore + METEOR + 事实性全套指标

### 实验2: 消融实验（模块消融）— 核心实验

在LED基线上逐个添加/移除创新模块，验证每个模块的贡献：

| 配置代号 | SAE | FGCA | CFL | 说明 |
|---------|-----|------|-----|------|
| LED (baseline) | ✗ | ✗ | ✗ | 原始LED，无任何新模块 |
| LED-FaCT w/o SAE | ✗ | ✓ | ✓ | 去掉章节感知嵌入 |
| LED-FaCT w/o FGCA | ✓ | ✗ | ✓ | 去掉忠实度门控交叉注意力 |
| LED-FaCT w/o CFL | ✓ | ✓ | ✗ | 去掉对比事实性损失（仅CE训练） |
| **LED-FaCT (Full)** | **✓** | **✓** | **✓** | **完整模型** |

**预期发现**: 每个模块都应有正向贡献；CFL对事实性指标提升最显著；SAE对长文档理解有帮助；FGCA在幻觉率上表现最好。

### 实验3: 事实性深度分析

各模型生成摘要的事实性对比：
- NLI蕴含率
- 幻觉率及类型分布（内在/外在/矛盾）
- n-gram重叠率
- LED-FaCT对比纯LED在幻觉率上的改进

### 实验4: 上下文长度影响

LED-FaCT在不同输入长度下的表现: 512, 1024, 2048, 4096, 8192, 16384 tokens

### 实验5: 参数敏感性分析

- beam_size: 1, 2, 4, 6, 8
- length_penalty: 0.6, 1.0, 1.5, 2.0, 2.5
- CFL权重 α: 0.01, 0.05, 0.1, 0.2, 0.5
- FGCA隐层维度: 64, 128, 256, 512
- 学习率: 1e-5, 3e-5, 5e-5, 1e-4

### 实验6: 截断策略对比（短上下文模型的局限）

仅对BART/PEGASUS等短上下文模型: head_only, tail_only, head_tail_mixed

---

## Benchmark 指标体系

### 主指标

| 指标 | 说明 |
|------|------|
| ROUGE-1/2/L/Lsum | n-gram重叠与LCS匹配 |
| BERTScore | 语义级相似度 |
| METEOR | 同义词感知匹配 |

### 事实性指标

| 指标 | 说明 |
|------|------|
| NLI Entailment Ratio | RoBERTa-MNLI判断每句是否被原文蕴含 |
| Hallucination Rate | 1 - 蕴含率 |
| 幻觉分类 | 区分内在/外在/矛盾幻觉 |
| n-gram Overlap | 生成摘要与原文的bigram/trigram重叠 |
| Novelty Ratio | 摘要中不出现在原文中的词汇比例 |

### 辅助指标

| 指标 | 说明 |
|------|------|
| Compression Ratio | 摘要长度/原文长度 |
| JS Divergence | 预测与参考的bigram分布JS散度 |
| Repetition Ratio | 4-gram重复率 |

---

## 运行方式

```bash
pip install -r requirements.txt

# 实验1: 多模型对比
python src/run_experiments.py --mode exp1 --dataset arxiv --max_samples 5000

# 实验2: 消融实验（核心）
python src/run_experiments.py --mode ablation --ablation_type all
python src/run_experiments.py --mode ablation --ablation_type no_sae
python src/run_experiments.py --mode ablation --ablation_type no_fgca
python src/run_experiments.py --mode ablation --ablation_type no_cfl
python src/run_experiments.py --mode ablation --ablation_type baseline

# 实验3: 事实性分析
python src/run_experiments.py --mode exp3

# 实验4: 上下文长度影响
python src/run_experiments.py --mode exp4

# 实验5: 参数敏感性
python src/run_experiments.py --mode sensitivity

# 实验6: 截断策略
python src/run_experiments.py --mode exp6

# 完整流水线
python src/run_experiments.py --mode full
```

---

## 论文结构

1. **引言**: 长文档摘要的重要性与幻觉问题
2. **相关工作**: 预训练摘要模型、长上下文方法、幻觉检测与缓解
3. **方法**:
   - 3.1 问题定义与基线模型 (LED)
   - 3.2 Section-Aware Embedding (SAE)
   - 3.3 Faithfulness-Guided Cross-Attention (FGCA)
   - 3.4 Contrastive Factuality Loss (CFL)
   - 3.5 训练与推理流程
4. **实验**:
   - 4.1 数据集与评估指标
   - 4.2 多模型对比 (Exp1)
   - 4.3 消融实验 (Exp2) — 模块消融
   - 4.4 事实性分析 (Exp3)
   - 4.5 上下文长度影响 (Exp4)
   - 4.6 参数敏感性 (Exp5)
   - 4.7 截断策略对比 (Exp6)
5. **分析与讨论**: SAE章节嵌入可视化、FGCA门控分布分析、CFL对比学习效果
6. **结论**

---

## 关键参考文献

1. Beltagy et al., "Longformer: The Long-Document Transformer" (2020) — LED基线
2. See et al., "PRIMERA: Pyramid-based Multi-Document Summarization" (2022)
3. Kryscinski et al., "Evaluating Factuality in Summarization" (2020)
4. Maynez et al., "On Faithfulness and Factuality in Abstractive Summarization" (2020)
5. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023)
6. Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023)
7. Chen et al., "Enhancing the Language Model with Monotonic Attention for Abstractive Summarization" (2023)
8. Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (2020)
9. Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training" (2019)
10. Zhang et al., "PEGASUS: Pre-training with Extracted Gap-sentences" (2020)

---

## 文件结构

```
end/
├── EXPERIMENT_PLAN.md
├── data/                    # 数据集缓存
├── models/                  # 模型保存
├── results/                 # 实验结果+图表
│   └── ablation/            # 消融实验结果
├── src/
│   ├── models/              # ★ 创新模块实现
│   │   ├── __init__.py
│   │   ├── led_fact.py        # LED-FaCT主模型
│   │   ├── section_embedding.py  # SAE模块
│   │   ├── faithfulness_gate.py  # FGCA模块
│   │   └── contrastive_loss.py   # CFL损失
│   ├── config.py            # 模型&训练配置
│   ├── data_utils.py        # 数据加载+章节检测
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   ├── benchmark.py         # 全面指标
│   ├── hallucination.py     # NLI幻觉检测
│   ├── ablation.py          # ★ 模块消融实验
│   ├── sensitivity.py       # 参数敏感性分析
│   ├── analyze.py           # 可视化
│   └── run_experiments.py   # 一键运行
├── notebooks/
├── paper/
└── requirements.txt
```

---

## GPU需求估算

| 模型 | 显存(训练) | 显存(推理) | 预估训练时间 |
|------|-----------|-----------|-------------|
| BART-Large-CNN | ~8GB | ~4GB | 2-4h |
| PEGASUS-arXiv | ~10GB | ~5GB | 3-5h |
| LED-Base-16384 | ~16GB | ~8GB | 8-12h |
| LED-FaCT (Full) | ~18GB | ~10GB | 10-14h |
| LED-FaCT (各消融) | ~16-18GB | ~8-10GB | 各8-14h |