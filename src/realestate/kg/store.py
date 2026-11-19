"""Knowledge Graph lưu trữ bất động sản (NetworkX, lưu file pickle; có thể export sang Neo4j).

Lược đồ:
  (Property)-[:LOCATED_IN]->(Ward)-[:PART_OF]->(District)-[:PART_OF]->(Province)
  (Property)-[:ON_STREET]->(Street)        (Property)-[:IN_PROJECT]->(Project)
  (Property)-[:HAS_TYPE]->(PropertyType)  (Property)-[:FOR_TRANSACTION]->(Transaction)
  (Property)-[:FACING]->(Direction)       (Property)-[:HAS_LEGAL_STATUS]->(LegalStatus)
  (Property)-[:HAS_POSITION]->(Position)  (Property)-[:HAS_FEATURE]->(Feature)
  (Property)-[:FROM_SOURCE]->(Source)
"""
from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

import networkx as nx

from ..rules.loader import lexicon_labels
from ..schemas import PropertyRecord
from ..utils.text import location_key, slug
from .location import location_level


class KnowledgeGraph:
    def __init__(self, graph: Optional[nx.MultiDiGraph] = None):
        self.g: nx.MultiDiGraph = graph if graph is not None else nx.MultiDiGraph()
        self._cache: dict[str, PropertyRecord] = {}

    # ------------------------------------------------------------- build
    def _node(self, nid: str, ntype: str, **attrs) -> str:
        if nid not in self.g:
            self.g.add_node(nid, type=ntype, **attrs)
        return nid

    def _edge(self, u: str, v: str, rel: str) -> None:
        if not self.g.has_edge(u, v, key=rel):
            self.g.add_edge(u, v, key=rel, rel=rel)

    def add_property(self, rec: PropertyRecord) -> str:
        pid = f"property:{rec.id}"
        attrs = rec.summary()
        self.g.add_node(pid, type="Property", record=rec.model_dump_json(), **{
            k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool)) or v is None
        })
        self._cache.pop(rec.id, None)

        # --- phân cấp địa danh
        loc_parent = None
        if rec.province:
            pk = location_key(rec.province)
            loc_parent = self._node(f"province:{pk}", "Province", name=rec.province, key=pk)
        if rec.district:
            dk = location_key(rec.district)
            prov_k = location_key(rec.province) if rec.province else "_"
            d = self._node(f"district:{prov_k}/{dk}", "District", name=rec.district, key=dk)
            if loc_parent:
                self._edge(d, loc_parent, "PART_OF")
            loc_parent = d
        if rec.ward:
            wk = location_key(rec.ward)
            w = self._node(f"ward:{loc_parent or '_'}/{wk}", "Ward", name=rec.ward, key=wk)
            if loc_parent:
                self._edge(w, loc_parent, "PART_OF")
            loc_parent = w
        if loc_parent:
            self._edge(pid, loc_parent, "LOCATED_IN")
        if rec.street:
            s = self._node(f"street:{slug(rec.province or '')}/{slug(rec.street)}", "Street", name=rec.street,
                           key=location_key(rec.street))
            self._edge(pid, s, "ON_STREET")
        if rec.project:
            pj = self._node(f"project:{slug(rec.project)}", "Project", name=rec.project, key=location_key(rec.project))
            self._edge(pid, pj, "IN_PROJECT")

        # --- thuộc tính phân loại
        ptype_labels = lexicon_labels("PROPERTY_TYPE")
        legal_labels = lexicon_labels("LEGAL_STATUS")
        pos_labels = lexicon_labels("POSITION")
        feat_labels = lexicon_labels("FEATURE")
        if rec.property_type:
            self._edge(pid, self._node(f"ptype:{rec.property_type}", "PropertyType",
                                       name=ptype_labels.get(rec.property_type, rec.property_type)), "HAS_TYPE")
        if rec.transaction_type:
            self._edge(pid, self._node(f"transaction:{rec.transaction_type}", "Transaction",
                                       name=rec.transaction_type), "FOR_TRANSACTION")
        if rec.direction:
            self._edge(pid, self._node(f"direction:{slug(rec.direction)}", "Direction", name=rec.direction), "FACING")
        if rec.legal_status:
            self._edge(pid, self._node(f"legal:{rec.legal_status}", "LegalStatus",
                                       name=legal_labels.get(rec.legal_status, rec.legal_status)), "HAS_LEGAL_STATUS")
        for p in rec.positions:
            self._edge(pid, self._node(f"position:{p}", "Position", name=pos_labels.get(p, p)), "HAS_POSITION")
        for f in rec.features:
            self._edge(pid, self._node(f"feature:{f}", "Feature", name=feat_labels.get(f, f)), "HAS_FEATURE")
        self._edge(pid, self._node(f"source:{rec.source}", "Source", name=rec.source), "FROM_SOURCE")
        return pid

    def add_many(self, records: Iterable[PropertyRecord]) -> int:
        n = 0
        for r in records:
            self.add_property(r)
            n += 1
        return n

    # ------------------------------------------------------------- query
    def get_property(self, pid: str) -> Optional[PropertyRecord]:
        pid = pid.removeprefix("property:")
        if pid in self._cache:
            return self._cache[pid]
        data = self.g.nodes.get(f"property:{pid}")
        if not data:
            return None
        rec = PropertyRecord.model_validate_json(data["record"])
        self._cache[pid] = rec
        return rec

    def property_ids(self) -> list[str]:
        return [n.removeprefix("property:") for n, d in self.g.nodes(data=True) if d.get("type") == "Property"]

    def properties(self) -> Iterator[PropertyRecord]:
        for pid in self.property_ids():
            rec = self.get_property(pid)
            if rec:
                yield rec

    def find_location_nodes(self, target: str) -> list[str]:
        """Nút địa danh khớp chuỗi người dùng ('Quận 3', 'Thủ Đức', 'Hà Nội')."""
        level = location_level(target)
        key = location_key(target)
        types = {"district": {"District"}, "ward": {"Ward"}, "province": {"Province"},
                 "street": {"Street"}}.get(level or "", {"Province", "District", "Ward", "Project", "Street"})
        return [n for n, d in self.g.nodes(data=True) if d.get("type") in types and d.get("key") == key]

    def properties_in_location(self, loc_node: str) -> set[str]:
        """Duyệt ngược PART_OF / LOCATED_IN / ON_STREET / IN_PROJECT để lấy mọi BĐS thuộc khu vực."""
        out: set[str] = set()
        stack = [loc_node]
        seen = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for u, _, rel in self.g.in_edges(n, keys=True):
                if self.g.nodes[u].get("type") == "Property":
                    out.add(u.removeprefix("property:"))
                elif rel == "PART_OF":
                    stack.append(u)
        return out

    def candidate_ids_for_locations(self, targets: list[str]) -> Optional[set[str]]:
        """Giao các tập BĐS theo từng ràng buộc địa điểm. None nếu không có địa điểm nào khớp nút KG."""
        result: Optional[set[str]] = None
        for t in targets:
            nodes = self.find_location_nodes(t)
            if not nodes:
                return None  # để bộ lọc thuộc tính quyết định (tránh loại nhầm)
            ids: set[str] = set()
            for n in nodes:
                ids |= self.properties_in_location(n)
            result = ids if result is None else result & ids
        return result

    def gazetteer(self) -> list[dict]:
        """Danh sách địa danh có trong KG (phục vụ parser luật & gợi ý cho LLM)."""
        out = []
        for n, d in self.g.nodes(data=True):
            if d.get("type") in ("Province", "District", "Ward", "Project"):
                out.append({"node": n, "type": d["type"], "name": d.get("name"), "key": d.get("key"),
                            "count": len(self.properties_in_location(n)) if d["type"] != "Ward" else None})
        return out

    def neighbors(self, pid: str) -> list[dict]:
        node = f"property:{pid.removeprefix('property:')}"
        return [{"rel": rel, "node": v, "type": self.g.nodes[v].get("type"), "name": self.g.nodes[v].get("name")}
                for _, v, rel in self.g.out_edges(node, keys=True)]

    def stats(self) -> dict:
        types = Counter(d.get("type") for _, d in self.g.nodes(data=True))
        rels = Counter(k for _, _, k in self.g.edges(keys=True))
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "node_types": dict(types), "relations": dict(rels)}

    # ------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:  # ghi file tạm rồi đổi tên: web đang chạy không đọc phải file dở
            pickle.dump(self.g, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def export_json(self, path: str | Path) -> None:
        try:
            data = nx.node_link_data(self.g, edges="edges")
        except TypeError:  # networkx < 3.4
            data = nx.node_link_data(self.g)
        for n in data["nodes"]:
            n.pop("record", None)
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        with open(path, "rb") as f:
            return cls(pickle.load(f))
