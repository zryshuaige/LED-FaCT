import random
import re
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SummaryPerturbator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        self.entity_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'),
            re.compile(r'\b[A-Z]{2,}\b'),
        ]

        self.month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        self.swap_words = {
            "increase": "decrease", "decrease": "increase",
            "improve": "worsen", "worsen": "improve",
            "high": "low", "low": "high",
            "large": "small", "small": "large",
            "significant": "insignificant", "insignificant": "significant",
            "positive": "negative", "negative": "positive",
            "better": "worse", "worse": "better",
            "more": "less", "less": "more",
            "most": "least", "least": "most",
            "achieve": "fail", "fail": "achieve",
            "effective": "ineffective", "ineffective": "effective",
            "proposed": "rejected", "rejected": "proposed",
            "outperform": "underperform", "underperform": "outperform",
            "superior": "inferior", "inferior": "superior",
            "important": "unimportant", "unimportant": "important",
        }

    def _swap_numbers(self, text: str) -> str:
        numbers = self.number_pattern.findall(text)
        if not numbers:
            return text

        result = text
        for num in numbers:
            multiplier = self.rng.choice([2, 3, 0.5, 0.1, 10])
            try:
                new_val = float(num) * multiplier
                if '.' in num:
                    new_str = f"{new_val:.1f}"
                else:
                    new_str = str(int(new_val))
            except (ValueError, OverflowError):
                continue
            result = result.replace(num, new_str, 1)
        return result

    def _swap_entities(self, text: str) -> str:
        entities = []
        for pattern in self.entity_patterns:
            entities.extend(pattern.findall(text))

        if len(entities) < 2:
            return text

        result = text
        shuffled = entities.copy()
        self.rng.shuffle(shuffled)
        for orig, repl in zip(entities, shuffled):
            if orig != repl:
                result = result.replace(orig, repl, 1)
        return result

    def _swap_antonyms(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            lower = word.lower().rstrip(".,;:!?")
            if lower in self.swap_words:
                replacement = self.swap_words[lower]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement + word[len(lower):].lstrip(lower))
            else:
                result.append(word)
        return " ".join(result)

    def perturb(self, text: str, strategy: str = "mixed") -> str:
        if strategy == "numbers":
            return self._swap_numbers(text)
        elif strategy == "entities":
            return self._swap_entities(text)
        elif strategy == "antonyms":
            return self._swap_antonyms(text)
        elif strategy == "mixed":
            strategy_choice = self.rng.choice(["numbers", "entities", "antonyms"])
            return self.perturb(text, strategy=strategy_choice)
        else:
            return self._swap_numbers(text)

    def perturb_batch(self, texts: List[str], strategy: str = "mixed") -> List[str]:
        return [self.perturb(text, strategy=strategy) for text in texts]


class ContrastiveFactualityLoss(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        projection_dim: int = 128,
        temperature: float = 0.07,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

        self.perturbator = SummaryPerturbator()

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        perturbed_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pooled = decoder_hidden_states.mean(dim=1)
        projected = self.projection_head(pooled)
        projected = F.normalize(projected, dim=-1)

        if perturbed_labels is not None:
            batch_size = projected.shape[0] // 2
            pos_repr = projected[:batch_size]
            neg_repr = projected[batch_size:]

            sim_matrix = torch.matmul(pos_repr, neg_repr.T) / self.temperature

            labels_contrastive = torch.arange(batch_size, device=projected.device)
            contrastive_loss = F.cross_entropy(sim_matrix, labels_contrastive)

            metrics = {
                "contrastive_loss": contrastive_loss.item(),
                "sim_pos_mean": torch.diag(sim_matrix).mean().item(),
                "sim_neg_mean": (sim_matrix.sum(dim=1) - torch.diag(sim_matrix)).mean().item() / max(batch_size - 1, 1),
            }

            return contrastive_loss, metrics
        else:
            return torch.tensor(0.0, device=decoder_hidden_states.device), {"contrastive_loss": 0.0}

    def create_perturbed_batch(
        self,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        perturbed_texts = self.perturbator.perturb_batch(texts, strategy="mixed")
        perturbed_ids = tokenizer(
            perturbed_texts,
            max_length=input_ids.shape[1],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]
        perturbed_ids = perturbed_ids.to(input_ids.device)
        return perturbed_ids