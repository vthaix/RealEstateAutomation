"""LUỒNG 2 — Xử lý dữ liệu: data/batches -> PhoBERT/Rule Engine -> data/processed + Knowledge Graph -> Neo4j.

    python scripts/process.py              # bổ sung tin MỚI trong data/batches (bỏ qua tin đã xử lý)
    python scripts/process.py --rebuild    # xử lý lại TOÀN BỘ từ đầu (sau khi sửa luật JSON)
    python scripts/process.py --no-neo4j   # không đẩy sang Neo4j

Web (luồng 3) tự nạp lại Knowledge Graph mới, không cần khởi động lại.
"""
import argparse
import json

import _bootstrap  # noqa: F401

from realestate.config import settings
from realestate.pipeline import run_ingest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="file hoặc thư mục dữ liệu thô (mặc định data/batches)")
    ap.add_argument("--rebuild", action="store_true", help="xử lý lại toàn bộ, ghi đè kết quả cũ")
    ap.add_argument("--mode", choices=["rule", "bert", "hybrid"], default=None)
    ap.add_argument("--no-neo4j", action="store_true")
    args = ap.parse_args()

    report = run_ingest(args.input, args.mode, append=not args.rebuild)
    print(json.dumps({k: report[k] for k in ("records", "duplicates_with_existing", "total_records_after",
                                             "extraction_mode", "coverage")}, ensure_ascii=False, indent=2))

    if not args.no_neo4j:
        try:
            from realestate.kg.neo4j_export import export_to_neo4j
            from realestate.kg.store import KnowledgeGraph

            res = export_to_neo4j(KnowledgeGraph.load(settings.kg_path), settings.neo4j_uri, settings.neo4j_user,
                                  settings.neo4j_password, reset=args.rebuild)
            print("Neo4j:", json.dumps(res))
        except Exception as e:  # Neo4j không chạy -> vẫn xong luồng xử lý
            print(f"(Bỏ qua Neo4j: {type(e).__name__}: {e})")
