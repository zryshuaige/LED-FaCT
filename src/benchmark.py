import os
import json
import logging
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
from tqdm import tqdm

from config import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
    for pred, ref in zip(predictions, references):
        if not pred.strip():
            for key in scores:
                scores[key].append(0.0)
            continue
        score = scorer.score(ref, pred)
        scores["rouge1"].append(score["rouge1"].fmeasure)
        scores["rouge2"].append(score["rouge2"].fmeasure)
        scores["rougeL"].append(score["rougeL"].fmeasure)
        scores["rougeLsum"].append(score["rougeLsum"].fmeasure)

    return {
        k: {"precision": float(np.mean(v)), "recall": float(np.mean(v)), "fmeasure": float(np.mean(v))}
        for k, v in scores.items()
    }


def compute_bertscore(predictions: List[str], references: List[str], lang: str = "en") -> Dict[str, float]:
    import bert_score
    logger.info(f"Computing BERTScore for {len(predictions)} samples...")
    P, R, F1 = bert_score.score(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def compute_meteor(predictions: List[str], references: List[str]) -> Dict[str, float]:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred)
        ref_tokens = nltk.word_tokenize(ref)
        try:
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)
        except Exception:
            scores.append(0.0)

    return {
        "meteor_mean": float(np.mean(scores)),
        "meteor_std": float(np.std(scores)),
        "meteor_median": float(np.median(scores)),
    }


def compute_js_divergence(predictions: List[str], references: List[str]) -> Dict[str, float]:
    def _get_ngram_dist(text, n=2):
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        dist = Counter(ngrams)
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()} if total > 0 else {}

    def _kl(p, q, epsilon=1e-10):
        return sum(p[k] * np.log2((p[k] + epsilon) / (q.get(k, 0) + epsilon)) for k in p)

    js_scores = []
    for pred, ref in zip(predictions, references):
        p_dist = _get_ngram_dist(pred)
        r_dist = _get_ngram_dist(ref)
        if not p_dist or not r_dist:
            js_scores.append(0.0)
            continue
        m = {k: 0.5 * (p_dist.get(k, 0) + r_dist.get(k, 0)) for k in set(p_dist) | set(r_dist)}
        kl_pm = _kl(p_dist, m)
        kl_qm = _kl(r_dist, m)
        js_scores.append(0.5 * (kl_pm + kl_qm))

    return {
        "js_divergence_mean": float(np.mean(js_scores)),
        "js_divergence_std": float(np.std(js_scores)),
    }


def compute_novelty_ratio(predictions: List[str], references: List[str], sources: List[str]) -> Dict[str, float]:
    def _ngram_set(text, n):
        tokens = text.lower().split()
        return set(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))

    novelty_1 = []
    novelty_2 = []
    for pred, ref, src in zip(predictions, references, sources):
        pred_1g = _ngram_set(pred, 1)
        src_1g = _ngram_set(src, 1)
        novelty_1.append(1.0 - len(pred_1g & src_1g) / max(len(pred_1g), 1))

        pred_2g = _ngram_set(pred, 2)
        src_2g = _ngram_set(src, 2)
        novelty_2.append(1.0 - len(pred_2g & src_2g) / max(len(pred_2g), 1))

    return {
        "novelty_unigram_mean": float(np.mean(novelty_1)),
        "novelty_bigram_mean": float(np.mean(novelty_2)),
    }


def compute_repetition_ratio(predictions: List[str]) -> Dict[str, float]:
    def _seq_rep(text, n=4):
        tokens = text.lower().split()
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        return 1.0 - len(set(ngrams)) / len(ngrams)

    rep_ratios = [_seq_rep(p) for p in predictions]
    return {
        "repetition_ratio_mean": float(np.mean(rep_ratios)),
        "repetition_ratio_std": float(np.std(rep_ratios)),
    }


def compute_length_stats(texts: List[str]) -> Dict[str, float]:
    lengths = [len(text.split()) for text in texts]
    return {
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)) if lengths else 0,
        "max_length": int(np.max(lengths)) if lengths else 0,
    }


def compute_compression_ratio(predictions: List[str], sources: List[str]) -> Dict[str, float]:
    pred_lens = [len(p.split()) for p in predictions]
    src_lens = [len(s.split()) for s in sources]
    ratios = [p / max(s, 1) for p, s in zip(pred_lens, src_lens)]
    return {
        "compression_ratio_mean": float(np.mean(ratios)),
        "compression_ratio_std": float(np.std(ratios)),
    }


def full_benchmark(
    predictions: List[str],
    references: List[str],
    sources: List[str] = None,
    compute_bert: bool = True,
    compute_met: bool = True,
) -> Dict:
    results = {}

    logger.info("Computing ROUGE...")
    results["rouge"] = compute_rouge(predictions, references)

    if compute_bert:
        logger.info("Computing BERTScore...")
        try:
            results["bertscore"] = compute_bertscore(predictions, references)
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")

    if compute_met:
        logger.info("Computing METEOR...")
        try:
            results["meteor"] = compute_meteor(predictions, references)
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")

    logger.info("Computing JS Divergence...")
    results["js_divergence"] = compute_js_divergence(predictions, references)

    results["repetition"] = compute_repetition_ratio(predictions)

    results["pred_length"] = compute_length_stats(predictions)
    results["ref_length"] = compute_length_stats(references)

    if sources:
        results["novelty"] = compute_novelty_ratio(predictions, references, sources)
        results["compression_ratio"] = compute_compression_ratio(predictions, sources)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="./results/benchmark_results.json")
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = [d["prediction"] for d in data]
    refs = [d["reference"] for d in data]
    srcs = [d.get("input_full", d.get("input")) for d in data]

    results = full_benchmark(preds, refs, srcs)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2))