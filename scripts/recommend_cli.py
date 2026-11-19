"""Gợi ý từ dòng lệnh.

    python scripts/recommend_cli.py "Tôi cần một căn nhà ở Quận 3, diện tích 60 m2 thì càng tốt, phòng tắm có thể 2"
    python scripts/recommend_cli.py --parser rule --weighting equal "căn hộ Hà Nội tối đa 4 tỷ, 2 phòng ngủ"
"""
import argparse
import json

import _bootstrap  # noqa: F401

from realestate.ranking.explain import fmt_vnd
from realestate.ranking.recommender import build_recommender

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--parser", choices=["auto", "llm", "rule"], default=None)
    ap.add_argument("--weighting", choices=["equal", "priority"], default=None)
    ap.add_argument("--lambda-aux", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rec = build_recommender()
    res = rec.recommend(args.query, top_k=args.top, weighting=args.weighting, lambda_aux=args.lambda_aux,
                        parser_mode=args.parser)
    if args.json:
        print(res.model_dump_json(indent=2))
        raise SystemExit
    print(f"\n== Tiêu chí (parser={res.parsed_query.parser}, trọng số={res.weighting}) ==")
    for c in res.parsed_query.criteria:
        print(f"  [{c.group.value:4}] {c.attribute:14} {c.operator.value:10} {c.target_value!s:>20} "
              f"min={c.target_min} max={c.target_max} level={c.preference_level and c.preference_level.value} "
              f"conf={c.confidence}{'  (cần xác nhận)' if c.needs_confirmation else ''}")
    print("  weights:", json.dumps(res.weights, ensure_ascii=False))
    for n in res.notes + res.parsed_query.warnings:
        print("  !", n)
    print(f"\n== {res.total_passed_hard}/{res.total_candidates} tin thỏa điều kiện bắt buộc ==")
    for i, r in enumerate(res.results, 1):
        p = r.property
        sc = "—" if r.score is None else f"{r.score:.3f} (cov {r.coverage:.2f})"
        print(f"\n{i}. [{sc}] {p.get('title')}\n   {p.get('district')}, {p.get('province')} | "
              f"{fmt_vnd(p.get('price_vnd'))} | {p.get('area_m2')} m2 | {p.get('url')}")
        for line in r.explanation:
            print("   ", line)
    if res.insufficient_data:
        print(f"\n(+ {len(res.insufficient_data)} tin chưa đủ dữ liệu để xác minh điều kiện bắt buộc)")
