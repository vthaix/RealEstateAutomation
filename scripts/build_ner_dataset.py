"""Sinh tập dữ liệu BIO cho PhoBERT từ Rule Engine (silver) và chia train/dev/test.

    python scripts/build_ner_dataset.py --test-ratio 0.15 --dev-ratio 0.1

Đầu ra (data/annotations/):
  silver_train.jsonl, silver_dev.jsonl  -> huấn luyện
  to_annotate_test.jsonl                -> nhóm gán nhãn hiệu chỉnh thủ công thành gold_test.jsonl
Tin trùng được loại TRƯỚC khi chia tập để tránh rò rỉ dữ liệu (mục 10.1).
"""
import argparse
import random

import _bootstrap  # noqa: F401

from realestate.config import settings
from realestate.extraction.rule_engine import RuleEngine
from realestate.extraction.weak_labeling import weak_label
from realestate.preprocessing.cleaner import clean_text, deduplicate
from realestate.preprocessing.source_parsers import parse_listing
from realestate.utils.io import load_raw_listings, write_jsonl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(settings.raw_data_path))
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    listings, n_dup = deduplicate(load_raw_listings(args.input))
    random.Random(args.seed).shuffle(listings)
    engine = RuleEngine()
    rows = []
    for li in listings:
        p = parse_listing(clean_text(li.text), li.url)
        text = f"{p.title or ''} . {p.description}".strip(" .")
        row = weak_label(text, engine, args.min_conf)
        row.update({"id": li.id, "text": text})
        rows.append(row)

    n = len(rows)
    n_test, n_dev = int(n * args.test_ratio), int(n * args.dev_ratio)
    out = settings.annotations_dir
    write_jsonl(out / "to_annotate_test.jsonl", rows[:n_test])
    write_jsonl(out / "silver_dev.jsonl", rows[n_test:n_test + n_dev])
    write_jsonl(out / "silver_train.jsonl", rows[n_test + n_dev:])
    print(f"Loại {n_dup} tin trùng. test={n_test} dev={n_dev} train={n - n_test - n_dev} -> {out}")
