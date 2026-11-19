"""Bộ phân tích yêu cầu dựa trên luật (fallback khi không có LLM, và baseline để so sánh).

Toàn bộ từ khóa/regex nằm trong `rules/query_rules.json`; giá trị số & thuộc tính được
nhận diện bằng chính RuleEngine của nhánh trích xuất (extraction_rules.json).
"""
from __future__ import annotations

import re
from typing import Optional

from ..extraction.rule_engine import RuleEngine, Span
from ..kg.location import location_level
from ..rules.loader import RuleSet, load_query_rules
from ..schemas import Criterion, Group, Operator, ParsedQuery, PreferenceLevel
from ..utils.text import collapse_ws, fold, location_key, nfc

SPAN_TO_ATTR = {
    "AREA": "area", "BEDROOM": "bedrooms", "BATHROOM": "bathrooms", "FLOOR": "floors",
    "FRONTAGE": "frontage", "DIRECTION": "direction", "LEGAL_STATUS": "legal_status",
    "POSITION": "position", "FEATURE": "feature", "PROPERTY_TYPE": "property_type",
    "TRANSACTION_TYPE": "transaction_type",
}
NUMERIC = {"price", "area", "bedrooms", "bathrooms", "floors", "frontage"}


def _contains(text: str, phrase: str) -> bool:
    """Khớp cụm từ có ranh giới chữ (tránh 'chỉ' khớp trong 'địa chỉ')."""
    return re.search(rf"(?<![0-9a-zà-ỹđ]){re.escape(phrase)}(?![0-9a-zà-ỹđ])", text, re.I) is not None


class RuleQueryParser:
    def __init__(self, rules: RuleSet | None = None, engine: RuleEngine | None = None,
                 gazetteer: Optional[list[dict]] = None):
        self.r = rules or load_query_rules()
        self.engine = engine or RuleEngine()
        self.split_rx = self.r.compile(self.r["clause_split_regex"])
        self.range_rx = self.r.compile(self.r["range"]["regex"])
        self.value_patterns = [(p, self.r.compile(p["regex"])) for p in self.r["attribute_value_patterns"]]
        loc = self.r["location"]
        self.loc_rx = self.r.compile(loc["regex"])
        self.loc_stop = set(loc["stop_words"])
        self.set_gazetteer(gazetteer or [])

    def set_gazetteer(self, gazetteer: list[dict]) -> None:
        """Gazetteer lấy từ Knowledge Graph: [{name, type, key}]"""
        self.gaz: dict[str, list[dict]] = {}
        for g in gazetteer:
            k = g.get("key") or location_key(g.get("name") or "")
            if k and not k.isdigit() and len(k) >= 3:
                self.gaz.setdefault(k, []).append(g)
        self.gaz_keys = sorted(self.gaz, key=len, reverse=True)

    # ================================================================ public
    def parse(self, query: str) -> ParsedQuery:
        q = collapse_ws(nfc(query))
        criteria: list[Criterion] = []
        for cs, ce in self._clauses(q):
            clause = q[cs:ce]
            criteria += self._parse_clause(clause)
        criteria = self._dedupe(criteria)
        pq = ParsedQuery(query=query, criteria=criteria, parser="rule")
        if not criteria:
            pq.warnings.append("Không nhận diện được tiêu chí nào — vui lòng mô tả rõ hơn.")
        return pq

    # ============================================================== clauses
    def _clauses(self, q: str) -> list[tuple[int, int]]:
        out, start = [], 0
        for m in self.split_rx.finditer(q):
            if m.start() > start:
                out.append((start, m.start()))
            start = m.end()
        if start < len(q):
            out.append((start, len(q)))
        return out

    def _flags(self, clause: str) -> dict:
        low = clause.lower()
        level = None
        for lv in self.r["priority_levels"]:
            if any(_contains(low, p) for p in lv["patterns"]):
                level = PreferenceLevel(lv["level"])
                break
        return {
            "hard": [c for c in self.r["hard_cues"] if _contains(low, c)],
            "soft": [c for c in self.r["soft_cues"] if _contains(low, c)],
            "neg": [c for c in self.r["negation_cues"] if _contains(low, c)],
            "level": level,
        }

    # ============================================================== entities
    def _entities(self, clause: str) -> list[dict]:
        """Các (attribute, value, start, end) tìm được trong mệnh đề."""
        ents: list[dict] = []
        # RANGE trước: 'từ 3 đến 5 tỷ', '60-80m2'
        for m in self.range_rx.finditer(clause):
            unit = (m.group("ub") or m.group("ua") or "").lower()
            if unit not in self.r["range"]["units"]:
                continue
            attr, mult = self.r["range"]["units"][unit]
            a = self.engine.to_float(m.group("a"), unit)
            b = self.engine.to_float(m.group("b"), unit)
            if a is None or b is None:
                continue
            ua = (m.group("ua") or "").lower()
            mult_a = self.r["range"]["units"].get(ua, [attr, mult])[1] if ua else mult
            lo, hi = sorted([a * mult_a, b * mult])
            ents.append({"attr": attr, "value": None, "min": lo, "max": hi, "start": m.start(), "end": m.end(),
                         "op": "RANGE", "conf": 0.9})

        def free(s: int, e: int) -> bool:
            return not any(s < x["end"] and e > x["start"] for x in ents)

        spans: list[Span] = self.engine.find_spans(clause)
        for sp in spans:
            if not free(sp.start, sp.end):
                continue
            if sp.label == "PRICE":
                role = sp.meta.get("role")
                if role in ("price_per_m2", "other_money"):
                    continue
                ents.append({"attr": "price", "value": sp.value, "start": sp.start, "end": sp.end,
                             "conf": 0.9 if sp.confidence >= 0.5 else 0.6, "raw": sp.text})
            elif sp.label == "FLOOR" and sp.meta.get("floor_level"):
                continue
            elif sp.label in SPAN_TO_ATTR:
                attr = SPAN_TO_ATTR[sp.label]
                if attr == "property_type" and fold(sp.text) in {fold(x) for x in self.r["property_type_generic"]}:
                    continue
                ents.append({"attr": attr, "value": sp.value, "start": sp.start, "end": sp.end, "conf": 0.9,
                             "raw": sp.text})
        # mẫu "thuộc tính + giá trị" đặc thù câu hỏi: 'phòng tắm có thể 2', 'giá 5'
        for cfg, rx in self.value_patterns:
            for m in rx.finditer(clause):
                if not free(m.start("n"), m.end("n")) or any(
                        e["attr"] == cfg["attribute"] and m.start() < e["end"] and m.end() > e["start"] for e in ents):
                    continue
                attr = cfg["attribute"]
                v = self.engine.to_float(m.group("n"))
                if v is None:
                    continue
                if attr == "price":
                    unit = (m.groupdict().get("u") or cfg.get("unit_default", "tỷ")).lower()
                    v *= 1e6 if unit in ("triệu", "tr") else 1e9
                elif attr in ("bedrooms", "bathrooms", "floors"):
                    v = int(v)
                ents.append({"attr": attr, "value": v, "start": m.start(), "end": m.end(), "conf": 0.8,
                             "raw": m.group(0), "value_pos": m.start("n")})
        ents += self._locations(clause, ents)
        return sorted(ents, key=lambda e: e["start"])

    def _locations(self, clause: str, taken: list[dict]) -> list[dict]:
        loc_cfg = self.r["location"]
        low = clause.lower()
        f = fold(clause)  # cùng độ dài với clause (bỏ dấu không đổi số ký tự với NFC tiếng Việt)
        out: list[dict] = []

        def overlaps(s, e):
            return any(s < x["end"] and e > x["start"] for x in out + taken if x.get("attr") == "location")

        # 1) regex có tiền tố hành chính
        aliases = {fold(k): v for k, v in loc_cfg["province_aliases"].items()}
        for m in self.loc_rx.finditer(clause):
            if m.group("num"):
                p2 = m.group("prefix2").lower()
                target = f"{'Quận' if p2 == 'q' else 'Phường'} {int(m.group('num'))}"
                out.append({"attr": "location", "value": target, "start": m.start(), "end": m.end(), "conf": 0.95})
                continue
            prefix = m.group("prefix").lower()
            name = m.group("name")
            if name.isdigit():
                target = f"{prefix.title()} {int(name)}"
                out.append({"attr": "location", "value": target, "start": m.start(), "end": m.end(), "conf": 0.95})
                continue
            if prefix.startswith(("tp", "thành phố", "tỉnh")):
                alias = next((v for k, v in aliases.items() if fold(name).startswith(k)), None)
                if alias:
                    out.append({"attr": "location", "value": alias, "start": m.start(), "end": m.end(), "conf": 0.95})
                    continue
            words = []
            for w in name.split(" "):
                if w.lower() in self.loc_stop:
                    break
                words.append(w)
            if not words:
                continue
            nm = " ".join(words[:4])
            verified = self._gaz_lookup(nm)
            if prefix in loc_cfg["generic_prefixes"] and not verified:
                continue
            if verified:
                nm = verified
            end = m.start("name") + len(" ".join(words[:4]))
            label = "" if prefix in loc_cfg["generic_prefixes"] else prefix.title().replace("Tp", "TP").replace("Q.", "Quận") + " "
            if verified and location_level(verified):
                value = verified  # tên trong KG đã có tiền tố ('Quận Bình Thạnh')
            else:
                value = (label + (nm if verified else nm.title())).strip()
            out.append({"attr": "location", "value": value,
                        "start": m.start(), "end": end, "conf": 0.9 if verified or not self.gaz else 0.7})
        # 2) gazetteer từ KG (không cần tiền tố): 'thủ đức', 'bình thạnh', 'hà nội'
        for key in self.gaz_keys:
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", f):
                if overlaps(m.start(), m.end()) or len(key.split()) < 2 and self.gaz[key][0]["type"] != "Province":
                    continue
                out.append({"attr": "location", "value": self.gaz[key][0]["name"], "start": m.start(),
                            "end": m.end(), "conf": 0.85})
        # 3) bí danh tỉnh/thành
        for alias, prov in loc_cfg["province_aliases"].items():
            for m in re.finditer(rf"(?<![0-9a-zà-ỹđ]){re.escape(alias)}(?![0-9a-zà-ỹđ])", low):
                if not overlaps(m.start(), m.end()):
                    out.append({"attr": "location", "value": prov, "start": m.start(), "end": m.end(), "conf": 0.9})
        return out

    def _gaz_lookup(self, name: str) -> Optional[str]:
        k = location_key(name)
        if k in self.gaz:
            return self.gaz[k][0]["name"]
        # khớp tiền tố dài nhất ('bình thạnh gần chợ' -> 'bình thạnh')
        for key in self.gaz_keys:
            if k.startswith(key + " "):
                return self.gaz[key][0]["name"]
        return None

    # ============================================================ criterion
    def _operator(self, clause: str, ent: dict, attr: str) -> tuple[str, Optional[str]]:
        if ent.get("op"):
            return ent["op"], None
        low = clause.lower()
        s = ent.get("value_pos", ent["start"])
        before = low[max(0, s - 22): s]
        after = low[ent["end"]: ent["end"] + 18]
        ops = self.r["operators"]
        if attr in NUMERIC:
            for op in ("EQ", "UNRESOLVED", "LTE", "GTE", "APPROX"):
                for cue in sorted(ops[op], key=len, reverse=True):
                    if before.rstrip().endswith(cue) or _contains(before[-12:], cue):
                        return op, cue
            for op, cues in self.r["operator_suffix"].items():
                if any(after.lstrip().startswith(c) for c in cues):
                    return op, cues[0]
        return self.r["default_operator"].get(attr, "EQ"), None

    def _parse_clause(self, clause: str) -> list[Criterion]:
        flags = self._flags(clause)
        out: list[Criterion] = []
        dg = self.r["default_group"]
        strong = self.r["operator_hardness"]["strong"]
        for ent in self._entities(clause):
            attr = ent["attr"]
            op, cue = self._operator(clause, ent, attr)
            conf = ent.get("conf", 0.9)
            note = None
            # --- nhóm HARD / SOFT
            if flags["neg"] and attr not in NUMERIC:
                op, group = "NOT_IN", Group.HARD
            elif cue in strong:
                group = Group.HARD
            elif flags["hard"] and not flags["soft"]:
                group = Group.HARD
            elif flags["soft"]:
                group = Group.SOFT
                if attr == "location" and op == "IN_AREA":
                    note = "Địa điểm nêu kèm tín hiệu linh hoạt ('gần', 'ưu tiên') -> xếp SOFT"
            else:
                # 'tối đa/dưới...' (có từ khóa toán tử) -> HARD; toán tử mặc định không làm tiêu chí thành bắt buộc
                g = dg["by_attribute"].get(attr) or (dg["by_operator"].get(op) if cue else None) or dg["fallback"]
                group = Group(g)
                conf = min(conf, 0.85)
            if op == "UNRESOLVED":
                conf = min(conf, 0.58)
                note = "Cách nói linh hoạt ('có thể ...') chưa đủ căn cứ để xác định toán tử"
            level = flags["level"] if group == Group.SOFT else None
            target = ent.get("value")
            if op in ("IN_SET", "NOT_IN") and target is not None and not isinstance(target, list):
                target = [target]
            out.append(Criterion(
                attribute=attr, target_value=target, target_min=ent.get("min"), target_max=ent.get("max"),
                unit={"price": "VND", "area": "m2", "frontage": "m"}.get(attr), operator=Operator(op), group=group,
                preference_level=level, confidence=round(conf, 2), evidence_span=clause.strip(),
                needs_confirmation=op == "UNRESOLVED" or conf < 0.6, note=note,
            ))
        return out

    @staticmethod
    def _dedupe(criteria: list[Criterion]) -> list[Criterion]:
        seen: dict[tuple, Criterion] = {}
        for c in criteria:
            key = (c.attribute, str(c.target_value).lower(), c.operator, c.target_min, c.target_max)
            if key not in seen or c.confidence > seen[key].confidence:
                seen[key] = c
        # gộp IN_SET cùng thuộc tính & cùng nhóm: 'sổ hồng hoặc sổ đỏ'
        merged: dict[tuple, Criterion] = {}
        out: list[Criterion] = []
        for c in seen.values():
            if c.operator in (Operator.IN_SET, Operator.NOT_IN) and isinstance(c.target_value, list):
                k = (c.attribute, c.operator, c.group)
                if k in merged:
                    merged[k].target_value = sorted(set(merged[k].target_value) | set(c.target_value))
                    continue
                merged[k] = c
            out.append(c)
        return out
