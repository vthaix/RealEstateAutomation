"""Đẩy Knowledge Graph sang Neo4j (đọc NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD trong .env).

    python scripts/export_neo4j.py           # MERGE, chạy lại không nhân đôi (dùng sau ingest --append)
    python scripts/export_neo4j.py --reset   # xóa sạch DB rồi nạp lại (dùng sau ingest ghi đè)
"""
import argparse
import json

import _bootstrap  # noqa: F401

from realestate.config import settings
from realestate.kg.neo4j_export import export_to_neo4j
from realestate.kg.store import KnowledgeGraph

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="xóa toàn bộ dữ liệu Neo4j trước khi nạp")
    ap.add_argument("--kg", default=str(settings.kg_path))
    ap.add_argument("--database", default=None, help="tên database (mặc định của server)")
    args = ap.parse_args()
    kg = KnowledgeGraph.load(args.kg)
    res = export_to_neo4j(kg, settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password,
                          reset=args.reset, database=args.database)
    print(json.dumps(res, indent=2))
