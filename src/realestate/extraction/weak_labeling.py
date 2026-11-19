"""Sinh nhãn BIO "bạc" (silver) từ Rule Engine để khởi tạo dữ liệu huấn luyện PhoBERT.

Quy trình đề xuất (mục 8.2, 10.3):
1. Rule Engine gán nhãn tự động toàn bộ tin rao -> data/annotations/silver_bio.jsonl
2. Nhóm gán nhãn hiệu chỉnh thủ công một tập con -> gold_bio.jsonl (dùng làm tập kiểm thử)
3. Huấn luyện PhoBERT trên silver (+ gold train), đánh giá trên gold test.

Token hóa ở mức âm tiết (tách theo khoảng trắng & dấu câu) để giữ offset ký tự. Có thể thay
bằng tách từ VnCoreNLP/underthesea nếu muốn bám sát cách tiền huấn luyện của PhoBERT.
"""
from __future__ import annotations

import re
from typing import Iterable

from .labels import ENTITY_TYPES
from .rule_engine import RuleEngine, Span

_TOKEN = re.compile(r"\w+(?:[.,]\d+)*|[^\w\s]", re.U)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def spans_to_bio(text: str, spans: Iterable[Span]) -> tuple[list[str], list[str]]:
    tokens = tokenize_with_offsets(text)
    tags = ["O"] * len(tokens)
    for sp in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if sp.label not in ENTITY_TYPES:
            continue
        idx = [i for i, (_, s, e) in enumerate(tokens) if s < sp.end and e > sp.start]
        if not idx or any(tags[i] != "O" for i in idx):
            continue  # không ghi đè thực thể đã gán
        tags[idx[0]] = f"B-{sp.label}"
        for i in idx[1:]:
            tags[i] = f"I-{sp.label}"
    return [t for t, _, _ in tokens], tags


def bio_to_spans(tokens: list[tuple[str, int, int]], tags: list[str],
                 probs: list[float] | None = None) -> list[dict]:
    """Gộp nhãn BIO thành span (start, end, label, confidence) trên văn bản gốc."""
    out: list[dict] = []
    cur = None
    for i, ((_, s, e), tag) in enumerate(zip(tokens, tags)):
        p = probs[i] if probs else 1.0
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur["label"] != tag[2:])):
            if cur:
                out.append(cur)
            cur = {"label": tag[2:], "start": s, "end": e, "probs": [p]}
        elif tag.startswith("I-") and cur:
            cur["end"] = e
            cur["probs"].append(p)
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    for sp in out:
        sp["confidence"] = sum(sp["probs"]) / len(sp["probs"])
        del sp["probs"]
    return out


def weak_label(text: str, engine: RuleEngine | None = None, min_conf: float = 0.6) -> dict:
    engine = engine or RuleEngine()
    spans = [s for s in engine.find_spans(text) if s.confidence >= min_conf]
    tokens, tags = spans_to_bio(text, spans)
    return {"tokens": tokens, "tags": tags}
