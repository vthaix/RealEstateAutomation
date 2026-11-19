"""Tinh chỉnh PhoBERT cho NER bất động sản (cần: pip install -r requirements-ml.txt).

    python scripts/train_ner.py --epochs 5
    python scripts/train_ner.py --train data/annotations/silver_train.jsonl --eval data/annotations/gold_test.jsonl
"""
import argparse
import json

import _bootstrap  # noqa: F401

from realestate.config import settings
from realestate.extraction.ner_model import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(settings.annotations_dir / "silver_train.jsonl"))
    ap.add_argument("--eval", default=str(settings.annotations_dir / "silver_dev.jsonl"))
    ap.add_argument("--base-model", default=settings.ner_base_model)
    ap.add_argument("--output", default=str(settings.ner_model_dir))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()
    metrics = train(args.train, args.eval, args.base_model, args.output, args.epochs, args.batch_size, args.lr,
                    settings.ner_max_length)
    print(json.dumps(metrics, indent=2))
