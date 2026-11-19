"""Độ đo đánh giá (mục 13).

- Trích xuất: Precision / Recall / F1 theo span nghiêm ngặt (micro, macro, theo nhãn)
              và theo trường sau chuẩn hóa.
- Phân tích yêu cầu: accuracy thuộc tính/toán tử, macro-F1 HARD/SOFT.
- Xếp hạng: Precision@K, Recall@K, NDCG@K, MRR.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------- NER (strict span)
def bio_spans(tags: Sequence[str]) -> set[tuple[int, int, str]]:
    spans, start, label = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and t[2:] != label):
            if label is not None:
                spans.add((start, i, label))
            start, label = (i, t[2:]) if t != "O" else (None, None)
    return spans


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def span_prf(gold: Iterable[Sequence[str]], pred: Iterable[Sequence[str]]) -> dict:
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for g, p in zip(gold, pred):
        gs, ps = bio_spans(g), bio_spans(p)
        for s in ps & gs:
            per[s[2]][0] += 1
        for s in ps - gs:
            per[s[2]][1] += 1
        for s in gs - ps:
            per[s[2]][2] += 1
    by_label = {k: _prf(*v) for k, v in sorted(per.items())}
    tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
    macro = sum(x["f1"] for x in by_label.values()) / len(by_label) if by_label else 0.0
    return {"micro": _prf(tp, fp, fn), "macro_f1": round(macro, 4), "by_label": by_label}


def field_prf(gold: list[dict[str, Any]], pred: list[dict[str, Any]], fields: list[str],
              rel_tol: float = 0.02) -> dict:
    """So khớp giá trị đã chuẩn hóa theo từng trường (None = không có)."""
    out = {}
    for f in fields:
        tp = fp = fn = 0
        for g, p in zip(gold, pred):
            gv, pv = g.get(f), p.get(f)
            if gv is None and pv is None:
                continue
            if gv is not None and pv is not None and _eq(gv, pv, rel_tol):
                tp += 1
            else:
                if pv is not None:
                    fp += 1
                if gv is not None:
                    fn += 1
        out[f] = _prf(tp, fp, fn)
    return out


def _eq(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)
    if isinstance(a, list) or isinstance(b, list):
        return set(a if isinstance(a, list) else [a]) == set(b if isinstance(b, list) else [b])
    return str(a).strip().lower() == str(b).strip().lower()


# ---------------------------------------------------------------- query parsing
def query_parsing_scores(gold: list[list[dict]], pred: list[list[dict]]) -> dict:
    """Mỗi phần tử: danh sách tiêu chí {attribute, operator, group, target_value}."""
    attr_tp = attr_fp = attr_fn = 0
    op_ok = op_n = 0
    group_pairs: list[tuple[str, str]] = []
    unconfirmed_auto = 0
    for g, p in zip(gold, pred):
        gmap = {c["attribute"]: c for c in g}
        pmap = {c["attribute"]: c for c in p}
        for a in pmap:
            if a in gmap:
                attr_tp += 1
                op_n += 1
                op_ok += pmap[a].get("operator") == gmap[a].get("operator")
                group_pairs.append((gmap[a].get("group"), pmap[a].get("group")))
                if gmap[a].get("needs_confirmation") and not pmap[a].get("needs_confirmation"):
                    unconfirmed_auto += 1
            else:
                attr_fp += 1
        attr_fn += sum(1 for a in gmap if a not in pmap)
    f1s = []
    for lab in ("HARD", "SOFT"):
        tp = sum(1 for g, p in group_pairs if g == lab and p == lab)
        fp = sum(1 for g, p in group_pairs if g != lab and p == lab)
        fn = sum(1 for g, p in group_pairs if g == lab and p != lab)
        f1s.append(_prf(tp, fp, fn)["f1"])
    return {"attribute": _prf(attr_tp, attr_fp, attr_fn), "operator_accuracy": round(op_ok / op_n, 4) if op_n else 0.0,
            "hard_soft_macro_f1": round(sum(f1s) / 2, 4), "should_confirm_but_auto_decided": unconfirmed_auto}


# ---------------------------------------------------------------- ranking
def precision_at_k(ranked: Sequence[str], rel: dict[str, int], k: int, threshold: int = 2) -> float:
    top = ranked[:k]
    return sum(1 for d in top if rel.get(d, 0) >= threshold) / k if k else 0.0


def recall_at_k(ranked: Sequence[str], rel: dict[str, int], k: int, threshold: int = 2) -> float:
    total = sum(1 for v in rel.values() if v >= threshold)
    if not total:
        return 0.0
    return sum(1 for d in ranked[:k] if rel.get(d, 0) >= threshold) / total


def dcg(gains: Sequence[float]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], rel: dict[str, int], k: int) -> float:
    gains = [rel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg else 0.0


def mrr(ranked: Sequence[str], rel: dict[str, int], threshold: int = 2) -> float:
    for i, d in enumerate(ranked):
        if rel.get(d, 0) >= threshold:
            return 1 / (i + 1)
    return 0.0


def ranking_report(runs: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks=(5, 10)) -> dict:
    out: dict[str, float] = {}
    qids = [q for q in qrels if q in runs]
    for k in ks:
        out[f"P@{k}"] = round(sum(precision_at_k(runs[q], qrels[q], k) for q in qids) / max(1, len(qids)), 4)
        out[f"R@{k}"] = round(sum(recall_at_k(runs[q], qrels[q], k) for q in qids) / max(1, len(qids)), 4)
        out[f"NDCG@{k}"] = round(sum(ndcg_at_k(runs[q], qrels[q], k) for q in qids) / max(1, len(qids)), 4)
    out["MRR"] = round(sum(mrr(runs[q], qrels[q]) for q in qids) / max(1, len(qids)), 4)
    out["queries"] = len(qids)
    return out
