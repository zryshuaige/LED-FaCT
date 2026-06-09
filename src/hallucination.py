import os
import json
import logging
from typing import List, Dict, Tuple
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


class HallucinationDetector:
    def __init__(self, nli_model_name="roberta-large-mnli", device=None):
        if device is None:
            device = get_device()
        self.device = device

        logger.info(f"Loading NLI model: {nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model = self.nli_model.to(device)
        self.nli_model.eval()

    def check_entailment(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        probs_dict = {}
        for idx, label in NLI_LABEL_MAP.items():
            if idx < len(probs):
                probs_dict[label] = probs[idx].item()

        return probs_dict

    def check_entailment_batch(self, premises: List[str], hypotheses: List[str], batch_size=16) -> List[Dict[str, float]]:
        results = []
        for i in tqdm(range(0, len(premises), batch_size), desc="NLI entailment check"):
            batch_premises = premises[i : i + batch_size]
            batch_hypotheses = hypotheses[i : i + batch_size]

            inputs = self.nli_tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                padding="max_length",
            ).to(self.device)

            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            for j in range(len(batch_premises)):
                probs_dict = {}
                for idx, label in NLI_LABEL_MAP.items():
                    if idx < probs.shape[1]:
                        probs_dict[label] = probs[j, idx].item()
                results.append(probs_dict)

        return results

    def detect_hallucinations(
        self,
        source_texts: List[str],
        generated_summaries: List[str],
        sentence_level: bool = True,
    ) -> Dict:
        logger.info(f"Running hallucination detection on {len(source_texts)} samples")

        entailment_scores = []
        contradiction_rates = []
        neutral_rates = []
        entailment_rates = []

        if not sentence_level:
            results = self.check_entailment_batch(source_texts, generated_summaries)
            for r in results:
                entailment_scores.append(r.get("entailment", 0))
                contradiction_rates.append(r.get("contradiction", 0))
                neutral_rates.append(r.get("neutral", 0))
                entailment_rates.append(r.get("entailment", 0))
        else:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            all_premises = []
            all_hypotheses = []
            sentence_counts = []

            for source, summary in zip(source_texts, generated_summaries):
                summary_sents = nltk.sent_tokenize(summary)
                sentence_counts.append(len(summary_sents))
                for sent in summary_sents:
                    truncated_source = source[:4000] if len(source) > 4000 else source
                    all_premises.append(truncated_source)
                    all_hypotheses.append(sent)

            if all_premises:
                sent_results = self.check_entailment_batch(all_premises, all_hypotheses, batch_size=32)

                idx = 0
                for count in sentence_counts:
                    doc_entailment = []
                    doc_contradiction = []
                    doc_neutral = []
                    for _ in range(count):
                        if idx < len(sent_results):
                            doc_entailment.append(sent_results[idx].get("entailment", 0))
                            doc_contradiction.append(sent_results[idx].get("contradiction", 0))
                            doc_neutral.append(sent_results[idx].get("neutral", 0))
                        idx += 1
                    entailment_rates.append(float(np.mean(doc_entailment)) if doc_entailment else 0.0)
                    contradiction_rates.append(float(np.mean(doc_contradiction)) if doc_contradiction else 0.0)
                    neutral_rates.append(float(np.mean(doc_neutral)) if doc_neutral else 0.0)

                    avg_score = float(np.mean([
                        s.get("entailment", 0) for s in sent_results[idx - count:idx]
                    ])) if count > 0 else 0.0
                    entailment_scores.append(avg_score)

        hallucination_rate = float(np.mean([1 - s for s in entailment_rates]))
        factuality_rate = float(np.mean(entailment_rates))

        results_dict = {
            "factuality_rate": factuality_rate,
            "hallucination_rate": hallucination_rate,
            "mean_entailment": float(np.mean(entailment_rates)),
            "mean_contradiction": float(np.mean(contradiction_rates)),
            "mean_neutral": float(np.mean(neutral_rates)),
            "num_samples": len(source_texts),
            "per_sample_entailment": entailment_rates,
            "per_sample_contradiction": contradiction_rates,
        }

        return results_dict

    def classify_hallucination_type(
        self,
        source_texts: List[str],
        generated_summaries: List[str],
    ) -> Dict:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        hallucination_types = {
            "intrinsic": 0,
            "extrinsic": 0,
            "contradictory": 0,
        }
        total_sentences = 0

        for source, summary in zip(source_texts, generated_summaries):
            summary_sents = nltk.sent_tokenize(summary)
            for sent in summary_sents:
                total_sentences += 1
                result = self.check_entailment(source[:4000], sent)

                if result.get("contradiction", 0) > 0.5:
                    hallucination_types["contradictory"] += 1
                elif result.get("neutral", 0) > 0.6:
                    source_lower = source.lower()
                    has_common = False
                    for word in sent.lower().split()[:10]:
                        if len(word) > 4 and word in source_lower:
                            has_common = True
                            break
                    if has_common:
                        hallucination_types["intrinsic"] += 1
                    else:
                        hallucination_types["extrinsic"] += 1

        return {
            "total_sentences": total_sentences,
            "hallucination_types": hallucination_types,
            "intrinsic_rate": hallucination_types["intrinsic"] / max(total_sentences, 1),
            "extrinsic_rate": hallucination_types["extrinsic"] / max(total_sentences, 1),
            "contradictory_rate": hallucination_types["contradictory"] / max(total_sentences, 1),
        }


def ngram_overlap(source: str, summary: str, n: int = 2) -> float:
    from collections import Counter

    def get_ngrams(text, n):
        tokens = text.lower().split()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    source_ngrams = get_ngrams(source, n)
    summary_ngrams = get_ngrams(summary, n)

    if not summary_ngrams:
        return 0.0

    overlap = sum((source_ngrams & summary_ngrams).values())
    total = sum(summary_ngrams.values())

    return overlap / total if total > 0 else 0.0


def compute_factuality_metrics(
    source_texts: List[str],
    generated_summaries: List[str],
    references: List[str] = None,
) -> Dict:
    logger.info("Computing n-gram overlap factuality metrics...")

    bigram_overlaps = [ngram_overlap(s, g, n=2) for s, g in zip(source_texts, generated_summaries)]
    trigram_overlaps = [ngram_overlap(s, g, n=3) for s, g in zip(source_texts, generated_summaries)]

    metrics = {
        "bigram_overlap_mean": float(np.mean(bigram_overlaps)),
        "bigram_overlap_std": float(np.std(bigram_overlaps)),
        "trigram_overlap_mean": float(np.mean(trigram_overlaps)),
        "trigram_overlap_std": float(np.std(trigram_overlaps)),
    }

    if references:
        ref_bi_overlaps = [ngram_overlap(s, r, n=2) for s, r in zip(source_texts, references)]
        metrics["reference_bigram_overlap_mean"] = float(np.mean(ref_bi_overlaps))
        novelty_gap = float(np.mean(bigram_overlaps) - np.mean(ref_bi_overlaps))
        metrics["novelty_gap"] = novelty_gap

    return metrics


def evaluate_hallucination_for_model(
    model_name: str,
    source_texts: List[str],
    generated_summaries: List[str],
    references: List[str] = None,
    use_nli: bool = True,
    output_dir: str = "./results",
):
    logger.info(f"Evaluating hallucination for model: {model_name}")

    ngram_metrics = compute_factuality_metrics(source_texts, generated_summaries, references)

    results = {"model": model_name, "ngram_metrics": ngram_metrics}

    if use_nli:
        detector = HallucinationDetector()
        nli_results = detector.detect_hallucinations(
            source_texts, generated_summaries, sentence_level=True
        )
        results["nli_metrics"] = nli_results

        type_results = detector.classify_hallucination_type(source_texts, generated_summaries)
        results["hallucination_types"] = type_results

    result_path = os.path.join(output_dir, f"hallucination_{model_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hallucination detection for summaries")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--model_name", type=str, default="unknown")
    parser.add_argument("--use_nli", action="store_true", help="Use NLI model for hallucination detection")
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = [d["input"] for d in data]
    predictions = [d["prediction"] for d in data]
    references = [d.get("reference") for d in data]
    if all(r is None for r in references):
        references = None

    results = evaluate_hallucination_for_model(
        model_name=args.model_name,
        source_texts=sources,
        generated_summaries=predictions,
        references=references,
        use_nli=args.use_nli,
        output_dir=args.output_dir,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))