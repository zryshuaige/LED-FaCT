# SUMM-Lens 实验方案

## 论文题目

**中文：** SUMM-Lens：长文档摘要的零训练推理期增强 —— 基于链式密度与 NLI 重排的轻量级方法

**English:** SUMM-Lens: Zero-Training Inference-Time Enhancements for Long-Document Summarization via Chain-of-Density Prompting and NLI Reranking

---

## 模块一览

| 模块 | 全称 | 针对问题 | 训练？ | 灵感来源 |
|:---|:---|:---|:---:|:---|
| **CoD** | Chain-of-Density Prompting | 单次生成的摘要稀疏，遗漏关键实体/数字 | ✗ | Adams et al., EMNLP 2023 |
| **NLR** | NLI-Rerank | 单条采样不可靠，需要从多候选中选最忠实的 | ✗ | Laban et al., TACL 2022 |

详细设计见 [README.md](README.md) 与 [README_zh.md](README_zh.md)。

---

## 实验设计

### E1：基线阶梯对比（2019 → 2024）

| 模型 | 预训练 | 参数 |
|:---|:---|:---:|
| BART-Large-CNN | CNN/DailyMail | 400M |
| DistilBART-CNN | CNN/DailyMail | 306M |
| PEGASUS-arXiv | arXiv | 568M |
| LED-large-arXiv | arXiv（长文档专用） | 460M |
| **Qwen2.5-1.5B-Instruct** | 通用指令微调 | 1.5B |

所有模型从 HuggingFace 直接 `from_pretrained` 加载，**仅推理，零训练**。

### E2：模块消融（同一主干 = Qwen2.5-1.5B-Instruct）

| 配置 | CoD | NLR | 回答的问题 |
|:---|:---:|:---:|:---|
| Qwen2.5-Vanilla | ✗ | ✗ | 2024 LLM 在零样本下能达到什么水平？ |
| + CoD | ✓ | ✗ | 单独的迭代密化是否提升 ROUGE / 忠实度？ |
| + NLR | ✗ | ✓ | 单独的 NLI 候选选择是否提升？ |
| **+ CoD + NLR** | **✓** | **✓** | 两者是否互补？ |

### E3：忠实度分析

NLI 蕴含率（RoBERTa-large-MNLI）、幻觉率、按候选打分的分布对比。

> 旧方案中的"上下文长度敏感性 / 学习率敏感性 / 截断策略"扫描在零训练设定下不再适用，已删除。

---

## 评估指标

**质量：** ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum，BERTScore F1，METEOR

**忠实度：** NLI 蕴含率，逐候选蕴含分（仅 NLR 配置）

**辅助：** JS 散度（bigram），4-gram 重复率，压缩比，新颖度

---

## 运行

### 0. 准备环境（一次性）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

中国大陆默认走 `HF_ENDPOINT=https://hf-mirror.com`（已在 `data_utils.py` 内置）。
首次运行任意模型会自动下载权重到 HF 缓存（默认 `~/.cache/huggingface/hub`）。

### 1. 冒烟测试（先跑这个）

```bash
python src/run_experiments.py --mode quick_test --dataset arxiv
```

5 个样本、3 个代表模型（BART / Qwen2.5-Vanilla / SUMM-Lens-Full），跳过 BERTScore/METEOR。
CPU 几分钟可完成。验证管线工作正常后再继续后面的实验。

### 2. 单模型评估（E1 / E2 任意单点）

任意 baseline 或消融配置都可以单独跑，方便逐项调试或在不同时段拆开运行：

```bash
# E1 baseline 阶梯（任选其一）
python src/run_experiments.py --mode single --model bart-large-cnn       --dataset arxiv --num_test 100
python src/run_experiments.py --mode single --model distilbart-cnn-12-6  --dataset arxiv --num_test 100
python src/run_experiments.py --mode single --model pegasus-arxiv        --dataset arxiv --num_test 100
python src/run_experiments.py --mode single --model led-large-arxiv      --dataset arxiv --num_test 100
python src/run_experiments.py --mode single --model qwen2.5-1.5b         --dataset arxiv --num_test 100

# E2 消融配置（任选其一）
python src/run_experiments.py --mode single --model summlens-cod   --dataset arxiv --num_test 100
python src/run_experiments.py --mode single --model summlens-nlr   --dataset arxiv --num_test 100
python src/run_experiments.py --mode single --model summlens-full  --dataset arxiv --num_test 100
```

CPU 慢的机器可加 `--skip_bertscore --skip_meteor` 只算 ROUGE：

```bash
python src/run_experiments.py --mode single --model qwen2.5-1.5b \
    --dataset arxiv --num_test 100 --skip_bertscore --skip_meteor
```

输出位置：`results/<model>_<dataset>/eval_results.json`（指标）+ `predictions.json`（前 50 条样例）。

### 3. E1 · 5 个 baseline 一次跑完

```bash
python src/run_experiments.py --mode baseline --dataset arxiv  --num_test 100
python src/run_experiments.py --mode baseline --dataset pubmed --num_test 100
```

依次评估 BART / DistilBART / PEGASUS-arXiv / LED-arXiv / Qwen2.5-Vanilla 五个 baseline。

### 4. E2 · 4 配置消融一次跑完

```bash
python src/run_experiments.py --mode ablation --dataset arxiv  --num_test 100
python src/run_experiments.py --mode ablation --dataset pubmed --num_test 100
```

依次评估 Vanilla / +CoD / +NLR / +CoD+NLR 四个配置（共享同一份 NLI 模型权重）。

### 5. E3 · 忠实度分析

E3 的 NLI 蕴含率与逐候选打分**自动随 E2 一起记录**，已写入 `results/<run_dir>/summlens-*/eval_results.json` 的 `benchmark` 字段。**无需单独命令**，跑完 E2 即可。

如需在已有预测上重新跑忠实度分析，可直接调用：

```bash
python -c "
import sys, json; sys.path.insert(0, 'src')
from hallucination import HallucinationDetector
det = HallucinationDetector()
preds = json.load(open('results/summlens-full_arxiv/predictions.json', 'r', encoding='utf-8'))
sources = [p['input_preview'] for p in preds]
hyps    = [p['prediction']    for p in preds]
import numpy as np
probs = det.check_entailment_batch(sources, hyps, batch_size=8)
ent = [p['entailment'] for p in probs]
print('NLI mean entailment:', float(np.mean(ent)))
"
```

### 6. 一键端到端（E1 + E2 + 自动出图）

```bash
python src/run_experiments.py --mode all --dataset arxiv  --num_test 100
python src/run_experiments.py --mode all --dataset pubmed --num_test 100
```

5 个 baseline + 4 个消融配置 + 全套图表，一次跑完。
输出位置：`results/<timestamp>/`，含

- `eval_results.json`（每模型一份）
- `predictions.json`（每模型 50 条样例）
- `all_results.json`（聚合）
- `figures/rouge_comparison.{png,pdf,csv}`（E1 主图）
- `figures/ablation_comparison.{png,pdf,csv}`（E2 主图）
- `figures/faithfulness.{png,pdf}`（BERTScore + JS 散度）
- `figures/results_table.tex`（LaTeX 表）

### 7. 仅重新出图（已有 eval_results 的情况）

```bash
python src/analyze.py --results_dir results/<timestamp> --output_dir results/<timestamp>/figures --dataset arxiv
```

不重跑模型、只读 JSON 出图。在调整图样式或补充新模型后非常方便。

### 8. Notebook 版（含交互式案例对比）

```bash
jupyter notebook notebooks/run.ipynb
```

或在 Colab 中打开 —— 第 1 个 cell 会自动 `git clone` + `pip install`。Notebook 内含 CoD/NLR 模块单步演示与并排候选对比。

---

## 参数与运行时建议

| 场景 | --num_test | 预计时长（T4 GPU） | 预计时长（CPU） |
|:---|:---:|:---:|:---:|
| quick_test | 5 | ~1 分钟 | ~5 分钟 |
| single (BART/PEGASUS) | 100 | ~3 分钟 | ~30 分钟 |
| single (Qwen2.5-Vanilla) | 100 | ~5 分钟 | ~1 小时 |
| single (summlens-full) | 100 | ~25 分钟 | ~4 小时 |
| ablation 全套 | 100 | ~40 分钟 | ~6 小时 |
| all (E1+E2) | 100 | ~70 分钟 | ~10 小时 |

**建议**：先 `quick_test` 验证；正式跑 `single` 拆开跑，每个模型独立写入 `results/<model>_<dataset>/`，崩了不影响其他；最后 `--mode all` 用一致 timestamp 重跑做最终汇总（同一目录便于出图与论文截图）。

---

## 参考文献

1. Adams G, et al. *From Sparse to Dense: GPT-4 Summarization with the Chain of Density Prompt.* EMNLP 2023.
2. Laban P, et al. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 2022.
3. Qwen Team. *Qwen2.5 Technical Report.* 2024.
4. Lewis M, et al. *BART: Denoising Sequence-to-Sequence Pre-training.* ACL 2020.
5. Zhang J, et al. *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.* ICML 2020.
6. Beltagy I, et al. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020.
