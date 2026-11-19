"""Chạy pipeline: dữ liệu thô -> trích xuất (rule | bert | hybrid) -> data/processed/properties.jsonl + Knowledge Graph.

    python scripts/ingest.py                                   # xây lại từ đầu với RAW_DATA_PATH
    python scripts/ingest.py --input data/raw/moi.json --append # BỔ SUNG bộ data mới vào dữ liệu đang có
    python scripts/ingest.py --input data/raw/moi.json          # chỉ dùng bộ mới (ghi đè dữ liệu cũ)
    python scripts/ingest.py --mode rule --limit 500

Đầu vào: JSON mảng hoặc JSONL, mỗi tin cần tối thiểu trường văn bản (text|content|description)
và nên có url (url|link); _id/id tùy chọn.
"""
import argparse
import json

import _bootstrap  # noqa: F401

from realestate.pipeline import run_ingest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="file dữ liệu thô (mặc định RAW_DATA_PATH)")
    ap.add_argument("--append", action="store_true", help="bổ sung vào dữ liệu + KG hiện có thay vì ghi đè")
    ap.add_argument("--mode", choices=["rule", "bert", "hybrid"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kg", default=None, help="đường dẫn lưu KG (.pkl)")
    ap.add_argument("--out", default=None, help="đường dẫn properties.jsonl")
    args = ap.parse_args()
    report = run_ingest(args.input, args.mode, args.limit, args.kg, args.out, append=args.append)
    print(json.dumps(report, ensure_ascii=False, indent=2))
