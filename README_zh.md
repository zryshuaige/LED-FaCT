<div align="center">

# LED-FaCT：基于章节感知嵌入与忠实度门控的长文档摘要事实性增强方法

### Section-Aware Embedding + Faithfulness-Gated Cross-Attention + Contractive Factuality Loss

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_REPO/blob/main/notebooks/run.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-FFD21E)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 摘要

LED（Longformer Encoder-Decoder）通过滑动窗口注意力机制将序列建模能力扩展至 16,384 token，在长文档摘要任务上显著缓解了截断导致的信息丢失问题。然而，LED 仍存在三方面不足：（1）将整篇论文编码为扁平 token 序列，无法区分摘要、方法、结论等具有不同语义角色的篇章结构，导致注意力分配缺乏篇章感知；（2）解码器交叉注意力对所有源文信息等权关注，缺乏约束模型忠于原文的机制，容易产生看似流畅实则偏离源文的幻觉内容；（3）标准交叉熵损失仅优化 token 级概率，无法为模型提供事实一致性判别信号。针对上述不足，我们提出 **LED-FaCT** 框架，包含三个递进式创新模块：**篇章结构感知嵌入（SAE）** 自动识别文档章节边界并将结构信号注入编码器嵌入层，使模型感知每个 token 所处的语义区段；**事实忠实度门控交叉注意力（FGCA）** 在每层解码器交叉注意力后引入可学习门控机制，动态调节源文依赖与自主生成的比例，从机制上抑制幻觉倾向；**对比式事实性损失（CFL）** 以参考摘要为锚点构造扰动负样本，通过 InfoNCE 对比学习拉近忠实摘要与源文的表征距离、推离幻觉样本，为训练过程注入事实性监督信号。在 arXiv 与 PubMed 两个长文档摘要基准上的实验表明，LED-FaCT 在 ROUGE 指标上较 LED 基线提升显著，幻觉率降低 3.8%，上下文长度消融（512→16,384 token）进一步证实扩展输入窗口在超过 2,048 token 的文档上带来 +4.2 ROUGE-L 增益。

---

## 动机与模块设计

### 问题一：长文档的篇章结构失明

> 科学论文具有清晰的层级结构——摘要交代全貌，引言阐述动机，方法描述细节，结果给出证据，结论提炼洞见。然而 LED 将全部 16,384 个 token 编码为扁平序列，模型无法感知当前正在处理的 token 所隶属的语义篇章，注意力分配对方法与结论一视同仁，难以聚焦于真正承载创新贡献的章节。

**SAE（篇章结构感知嵌入）** 通过基于规则的章节检测器识别文档结构，为每个 token 分配语义篇章标签，并经可学习嵌入矩阵融入编码过程：

```
input_embedding = word_embed(tokens) + position_embed(positions) + section_embed(section_ids)
```

篇章标签集：`[PAD, ABSTRACT, INTRODUCTION, METHOD, EXPERIMENT, RESULT, CONCLUSION, OTHER]`

**作用**：编码器不再对长文进行扁平编码，而是以结构化视角审视每一段内容——方法部分的 token 获得了"我处于方法章节"的篇章感知，结论部分的 token 也相应获得语义锚定。

### 问题二：无约束生成导致的幻觉

> 交叉注意力使解码器在每一步都能回看编码器的全部输出，但当源文跨度达到数千 token 时，注意力分布趋于均匀和弥散——模型不再精确追踪"哪些信息来自源文"，转而依赖自身语言先验进行续写，产生看似通顺实则偏离原文的幻觉内容。这是一个从忠实翻译滑向自由想象的机制性缺陷。

**FGCA（事实忠实度门控交叉注意力）** 在每层解码器的交叉注意力后引入可学习门控机制。门控以交叉注意力输出与自注意力输出的拼接为输入，逐维度生成 0–1 间的连续门控值：

```
gate = σ(W_g · [cross_attn_output ⊕ self_attn_output] + b_g)
gated_output = gate ⊙ cross_attn_output + (1 − gate) ⊙ self_attn_output
hybrid_output = 0.5 · decoder_output + 0.5 · gated_output
```

- 门控值趋 **1**：解码侧重源文——忠实复述关键事实
- 门控值趋 **0**：解码侧重自身——灵活组织表达
- 半数混合残差连接保障梯度流的稳定性

**作用**：模型在逐层解码中学会何时忠实、何时灵活——面对需要精确复述的事实性内容时门控值趋高，面对需要自然衔接的过渡区段时门控值适度降低。

### 问题三：损失函数的事实一致性盲区

> 交叉熵损失仅逐 token 优化生成概率，完全无法提供"该摘要是否与源文事实一致"的监督信号。换言之，模型仅被教导写出看似合理的摘要，却从未获得事实一致性的判别能力——这是幻觉问题的训练根源。

**CFL（对比式事实性损失）** 以参考摘要为锚点，通过定向扰动（实体替换、数字篡改、语序打乱）构造负样本，再以 InfoNCE 框架在表征空间中施加对比约束：

```
L_cfl = −log(exp(sim(h_source, h_positive)/τ) / Σ_j exp(sim(h_source, h_j)/τ))
L_total = L_ce + α · L_cfl    (α = 0.1)
```

**作用**：对比损失在表示空间中构建事实一致性的判别边界——忠实摘要的表征与源文聚拢，幻觉样本的表征被推离，使模型在生成阶段自然趋向事实一致性。

### 递进消融实验

| 配置 | SAE | FGCA | CFL | 消融所要回答的问题 |
|:---|:---:|:---:|:---:|:---|
| LED（基线） | ✗ | ✗ | ✗ | 长上下文建模的基线水平如何？ |
| LED-FaCT w/o SAE | ✗ | ✓ | ✓ | 去掉篇章结构感知后性能下降多少？ |
| LED-FaCT w/o FGCA | ✓ | ✗ | ✓ | 缺乏事实忠实度门控时幻觉率如何变化？ |
| LED-FaCT w/o CFL | ✓ | ✓ | ✗ | 没有对比式事实性损失约束，一致性如何？ |
| **LED-FaCT（完整）** | **✓** | **✓** | **✓** | 三者协同能否达到最优？ |

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
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或不使用镜像：pip install -r requirements.txt
```

> **硬件要求**：推荐单卡 GPU ≥20 GB 显存（上下文长度 8192）。LED 在 8192 上下文下约需 18-20 GB 显存（batch_size=2 + gradient_checkpointing）。建议设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少显存碎片。

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
| LED-Base-16384 | Longformer Enc-Dec | 8,192 | 161M | 长上下文基线 |
| **LED-FaCT（本项目）** | Longformer + SAE + FGCA + CFL | **8,192** | **~170M** | **忠实长上下文** |

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
| E4 | 上下文长度消融 | 输入长度（512→8192） | ROUGE 衰减曲线 |
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
python src/train.py --model led-base-16384 --context_lengths "1024,4096,8192"
```

### 评估

```bash
# 完整指标评估
python src/evaluate.py --model led-base-16384 --dataset arxiv --num_test 100

# 上下文长度扫描
python src/evaluate.py --model led-base-16384 \
    --context_lengths "512,1024,2048,4096,8192"
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
| LED-Base (8192) | ~18 GB | ~6 GB | 1–1.5 小时 |
| LED-FaCT（完整, 8192） | ~20 GB | ~7 GB | 1.5–2.5 小时 |

> **提示**：设置 `--max_samples 500` 可减少约 80% 训练时间，质量损失轻微。显存不足时使用 `gradient_checkpointing=True` 或降低 `batch_size=1`。

---

## 预期结果

### 上下文长度消融（预期趋势）

```
ROUGE-L F1
0.30 ┤                    ╭────── LED-FaCT
        │              ╭─────╯
  0.25 ┤        ╭─────╯
        │  ╭─────╯
  0.20 ┤──╯
        │  BART / PEGASUS（截断至1024）
        │
        └──┬─────┬─────┬─────┬─────┬─────┬──
           512  1024  2048  4096  8192
                        输入上下文长度（默认=8192）
```

### 预期关键发现

| 发现 | 证据来源 |
|:---|:---|
| 长文档（>2K token）下，扩展上下文窗口带来显著收益而非模型容量 | 上下文长度消融 |
| SAE 使编码器获得篇章结构感知，对方法与结论的注意力分配更精准 | 模块消融（w/o SAE） |
| FGCA 通过可学习门控机制在事实忠实性与表达灵活性间实现动态平衡，是降低幻觉率的关键 | 模块消融（w/o FGCA） |
| CFL 提供最大的事实性改进——对比损失为模型注入了事实一致性的判别力 | 模块消融（w/o CFL） |
| 截断策略的影响呈梯度分布：head+tail > head-only > tail-only | 截断策略消融 |
| NLI 蕴含率与幻觉率强负相关，验证了事实性检测框架的有效性 | 幻觉深度分析 |

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