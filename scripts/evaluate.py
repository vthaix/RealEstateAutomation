"""Đánh giá (mục 13).

1) Trích xuất — so sánh Rule-only / BERT-only / BERT+Rule trên tập gold BIO:
    python scripts/evaluate.py ner --gold data/annotations/gold_test.jsonl

2) Phân tích yêu cầu — so sánh parser với nhãn tiêu chí:
    python scripts/evaluate.py query --gold data/annotations/queries_gold.jsonl --parser rule

3) Xếp hạng — Precision@K, Recall@K, NDCG@K + ablation (equal/priority, λ):
    python scripts/evaluate.py ranking --qrels data/annotations/qrels.jsonl
"""
import argparse
import json

import _bootstrap  # noqa: F401

from realestate.config import settings
from realestate.evaluation.metrics import query_parsing_scores, ranking_report, span_prf
from realestate.extraction.labels import ENTITY_TYPES
from realestate.extraction.rule_engine import RuleEngine
from realestate.extraction.weak_labeling import spans_to_bio
from realestate.utils.io import read_jsonl


def eval_ner(gold_path: str) -> dict:
    from realestate.extraction.ner_model import PhoBertNER
    from realestate.extraction.rule_engine import Span

    rows = list(read_jsonl(gold_path))
    gold = [r["tags"] for r in rows]
    engine = RuleEngine()
    report = {}
    # Rule-only
    pred = [spans_to_bio(r["text"], [s for s in engine.find_spans(r["text"]) if s.label in ENTITY_TYPES])[1]
            for r in rows]
    report["rule_only"] = span_prf(gold, pred)
    ner = PhoBertNER.try_load(settings.ner_model_dir)
    if ner:
        bert_pred, hybrid_pred = [], []
        for r in rows:
            bs = ner.predict(r["text"])
            b_spans = [Span(b["label"], b["start"], b["end"], b["text"]) for b in bs]
            bert_pred.append(spans_to_bio(r["text"], b_spans)[1])
            rs = [s for s in engine.find_spans(r["text"]) if s.label in ENTITY_TYPES]
            extra = [s for s in rs if not any(s.start < b.end and s.end > b.start for b in b_spans)]
            hybrid_pred.append(spans_to_bio(r["text"], b_spans + extra)[1])
        report["bert_only"] = span_prf(gold, bert_pred)
        report["bert_rule"] = span_prf(gold, hybrid_pred)
    else:
        report["note"] = "Chưa có model PhoBERT -> chỉ đánh giá Rule-only"
    return report


def eval_query(gold_path: str, parser_mode: str) -> dict:
    from realestate.query import QueryParser

    rows = list(read_jsonl(gold_path))  # {"query": str, "criteria": [...]}
    qp = QueryParser(mode=parser_mode, model=settings.llm_model, has_credentials=settings.has_llm_credentials)
    preds = [[c.model_dump(mode="json") for c in qp.parse(r["query"], mode=parser_mode).criteria] for r in rows]
    return query_parsing_scores([r["criteria"] for r in rows], preds)


def eval_ranking(qrels_path: str, k: int) -> dict:
    from realestate.ranking.recommender import build_recommender

    rows = list(read_jsonl(qrels_path))  # {"qid", "query", "judgments": {property_id: 0..3}}
    rec = build_recommender()
    qrels = {r["qid"]: r["judgments"] for r in rows}
    configs = {"A_equal_lambda0": ("equal", 0.0), "B_priority_lambda0": ("priority", 0.0),
               "D_priority_lambda0.1": ("priority", 0.1)}
    out = {}
    parsed = {r["qid"]: rec.parser.parse(r["query"]) for r in rows}  # phân tích 1 lần, dùng chung cho mọi cấu hình
    for name, (w, lam) in configs.items():
        runs = {}
        for r in rows:
            res = rec.recommend(parsed[r["qid"]].model_copy(deep=True), top_k=k, weighting=w, lambda_aux=lam)
            runs[r["qid"]] = [x.property["id"] for x in res.results]
        out[name] = ranking_report(runs, qrels, ks=(5, k))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ner"); a.add_argument("--gold", required=True)
    b = sub.add_parser("query"); b.add_argument("--gold", required=True); b.add_argument("--parser", default="rule")
    c = sub.add_parser("ranking"); c.add_argument("--qrels", required=True); c.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    if args.cmd == "ner":
        res = eval_ner(args.gold)
    elif args.cmd == "query":
        res = eval_query(args.gold, args.parser)
    else:
        res = eval_ranking(args.qrels, args.k)
    print(json.dumps(res, ensure_ascii=False, indent=2))
