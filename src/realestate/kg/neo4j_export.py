"""Đẩy Knowledge Graph (NetworkX) sang Neo4j — cần `pip install neo4j`.

    python scripts/export_neo4j.py            # MERGE (chạy lại nhiều lần không bị nhân đôi)
    python scripts/export_neo4j.py --reset    # xóa sạch DB rồi nạp lại

Mỗi nút có label theo loại (Property, Province, District, Ward, Street, Project, PropertyType,
Transaction, Direction, LegalStatus, Position, Feature, Source) và khóa duy nhất `id`.
Ví dụ truy vấn trong Neo4j Browser:
    MATCH (p:Property)-[:LOCATED_IN]->(:Ward)-[:PART_OF]->(d:District {key:'3'})
    WHERE p.price_vnd <= 10e9 RETURN p.title, p.price_vnd, p.area_m2 LIMIT 20
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

from .store import KnowledgeGraph

log = logging.getLogger(__name__)

LABELS = {"Property", "Province", "District", "Ward", "Street", "Project", "PropertyType", "Transaction",
          "Direction", "LegalStatus", "Position", "Feature", "Source"}


def _props(node_id: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in ("type", "record") or v is None:
            continue
        key = "listing_id" if k == "id" else k  # 'id' của bản ghi gốc không được đè khóa nút
        out[key] = v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False)
    out["id"] = node_id
    if d.get("type") == "Property" and d.get("record"):
        rec = json.loads(d["record"])
        # danh sách -> mảng chuỗi (Neo4j hỗ trợ list thuần), dấu vết trích xuất -> chuỗi JSON
        out["positions"] = rec.get("positions") or []
        out["features"] = rec.get("features") or []
        out["warnings"] = rec.get("warnings") or []
        out["description"] = (rec.get("description") or "")[:4000]
        out["fields_json"] = json.dumps(rec.get("fields", {}), ensure_ascii=False)
    return out


def export_to_neo4j(kg: KnowledgeGraph, uri: str, user: str, password: str, batch: int = 1000,
                    reset: bool = False, database: str | None = None) -> dict:
    from neo4j import GraphDatabase

    nodes: dict[str, list[dict]] = defaultdict(list)
    ntype: dict[str, str] = {}
    for n, d in kg.g.nodes(data=True):
        t = d.get("type")
        if t in LABELS:
            nodes[t].append(_props(n, d))
            ntype[n] = t
    edges: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for u, v, rel in kg.g.edges(keys=True):
        if u in ntype and v in ntype and rel.replace("_", "").isalpha():
            edges[(ntype[u], rel, ntype[v])].append({"u": u, "v": v})

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    with driver.session(database=database) as s:
        if reset:
            log.info("Xóa dữ liệu cũ trong Neo4j")
            s.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
        for t in nodes:  # ràng buộc duy nhất = index -> MATCH theo id nhanh
            s.run(f"CREATE CONSTRAINT {t.lower()}_id IF NOT EXISTS FOR (n:{t}) REQUIRE n.id IS UNIQUE")
        for t in ("District", "Province", "Ward"):
            s.run(f"CREATE INDEX {t.lower()}_key IF NOT EXISTS FOR (n:{t}) ON (n.key)")
        s.run("CREATE INDEX property_price IF NOT EXISTS FOR (n:Property) ON (n.price_vnd)")
        for t, rows in nodes.items():
            for i in range(0, len(rows), batch):
                s.run(f"UNWIND $rows AS r MERGE (n:{t} {{id: r.id}}) SET n = r", rows=rows[i:i + batch])
            log.info("Nút %s: %d", t, len(rows))
        for (tu, rel, tv), rows in edges.items():
            for i in range(0, len(rows), batch):
                s.run(f"UNWIND $rows AS r MATCH (a:{tu} {{id: r.u}}) MATCH (b:{tv} {{id: r.v}}) "
                      f"MERGE (a)-[:{rel}]->(b)", rows=rows[i:i + batch])
        counts = s.run("MATCH (n) WITH count(n) AS n MATCH ()-[r]->() RETURN n, count(r) AS r").single()
    driver.close()
    return {"nodes_sent": sum(map(len, nodes.values())), "edges_sent": sum(map(len, edges.values())),
            "neo4j_nodes": counts["n"], "neo4j_relationships": counts["r"]}
