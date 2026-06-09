import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn


SECTION_LABELS = {
    0: "PAD",
    1: "ABSTRACT",
    2: "INTRODUCTION",
    3: "METHOD",
    4: "EXPERIMENT",
    5: "RESULT",
    6: "CONCLUSION",
    7: "OTHER",
}

LABEL_TO_ID = {v: k for k, v in SECTION_LABELS.items()}
NUM_SECTION_TYPES = len(SECTION_LABELS)


SECTION_PATTERNS = [
    (r"(?i)(?:^|\n)\s*abstract\b", "ABSTRACT"),
    (r"(?i)(?:^|\n)\s*introduction\b", "INTRODUCTION"),
    (r"(?i)(?:^|\n)\s*(?:related\s+work|background|literature\s+review)\b", "INTRODUCTION"),
    (r"(?i)(?:^|\n)\s*(?:method|methodology|approach|model|proposed\s+method|framework)\b", "METHOD"),
    (r"(?i)(?:^|\n)\s*(?:experiment|experimental\s+setup|evaluation|experimental\s+results|setup)\b", "EXPERIMENT"),
    (r"(?i)(?:^|\n)\s*(?:result|results|discussion|analysis|empirical|findings)\b", "RESULT"),
    (r"(?i)(?:^|\n)\s*(?:conclusion|conclusions|summary|future\s+work|final)\b", "CONCLUSION"),
]


@dataclass
class SectionSpan:
    label: str
    label_id: int
    start_char: int
    end_char: int


class SectionDetector:
    def __init__(self):
        self.compiled_patterns = [
            (re.compile(pattern), label) for pattern, label in SECTION_PATTERNS
        ]

    def detect_sections(self, text: str) -> List[SectionSpan]:
        matches = []
        for pattern, label in self.compiled_patterns:
            for m in pattern.finditer(text):
                matches.append((m.start(), label))

        matches.sort(key=lambda x: x[0])

        seen_labels = set()
        unique_matches = []
        for pos, label in matches:
            if label not in seen_labels:
                unique_matches.append((pos, label))
                seen_labels.add(label)

        if not unique_matches:
            return [SectionSpan("OTHER", LABEL_TO_ID["OTHER"], 0, len(text))]

        spans = []
        for i, (start, label) in enumerate(unique_matches):
            end = unique_matches[i + 1][0] if i + 1 < len(unique_matches) else len(text)
            spans.append(SectionSpan(label, LABEL_TO_ID[label], start, end))

        if spans[0].start_char > 0:
            spans.insert(0, SectionSpan("OTHER", LABEL_TO_ID["OTHER"], 0, spans[0].start_char))

        return spans

    def text_to_section_ids(self, text: str, tokenizer, max_length: int) -> torch.Tensor:
        if not text or not text.strip():
            return torch.zeros(max_length, dtype=torch.long)

        spans = self.detect_sections(text)
        char_to_section = {}
        for span in spans:
            for c in range(span.start_char, min(span.end_char, len(text))):
                char_to_section[c] = span.label_id

        default_section = LABEL_TO_ID["OTHER"]
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
        token_offsets = None
        if hasattr(tokenizer, "encode_plus"):
            enc = tokenizer(text, truncation=False, return_offsets_mapping=True,
                            add_special_tokens=True)
            if "offset_mapping" in enc:
                token_offsets = enc["offset_mapping"]

        section_ids = []
        for i, tok_id in enumerate(tokens):
            if i >= max_length:
                break
            if i == 0 or (token_offsets and token_offsets[i] == (0, 0)):
                section_ids.append(LABEL_TO_ID["PAD"])
            elif token_offsets:
                start_off, end_off = token_offsets[i]
                mid_char = (start_off + end_off) // 2
                section_ids.append(char_to_section.get(mid_char, default_section))
            else:
                frac = i / max(len(tokens) - 1, 1)
                char_pos = int(frac * len(text))
                section_ids.append(char_to_section.get(char_pos, default_section))

        pad_len = max_length - len(section_ids)
        if pad_len > 0:
            section_ids.extend([LABEL_TO_ID["PAD"]] * pad_len)

        return torch.tensor(section_ids[:max_length], dtype=torch.long)

    def batch_text_to_section_ids(
        self,
        texts: List[str],
        tokenizer,
        max_length: int,
    ) -> torch.Tensor:
        all_section_ids = []
        for text in texts:
            all_section_ids.append(self.text_to_section_ids(text, tokenizer, max_length))
        return torch.stack(all_section_ids)


class SectionAwareEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_section_types: int = NUM_SECTION_TYPES,
        section_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.section_embed_dim = section_embed_dim
        self.num_section_types = num_section_types

        self.section_embedding = nn.Embedding(num_section_types, section_embed_dim)
        self.section_projection = nn.Linear(section_embed_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.section_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.section_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.section_projection.bias)

    def forward(
        self,
        input_embeds: torch.Tensor,
        section_ids: torch.Tensor,
    ) -> torch.Tensor:
        section_emb = self.section_embedding(section_ids)
        section_proj = self.section_projection(section_emb)
        section_proj = self.dropout(section_proj)
        output = self.layer_norm(input_embeds + section_proj)
        return output