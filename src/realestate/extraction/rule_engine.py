"""Rule Engine (mục 10.4) — bộ THÔNG DỊCH luật.

Toàn bộ luật (regex, từ khóa ngữ cảnh, từ điển, ngưỡng) nằm trong
`src/realestate/rules/extraction_rules.json`. Module này chỉ nạp file luật và áp dụng,
không chứa luật cứng nào; sửa/thêm luật => sửa JSON.

Hai vai trò:
1. Phát hiện (Rule-only baseline + sinh nhãn yếu cho PhoBERT): `find_spans(text)`
2. Chuẩn hóa hậu kỳ cho đoạn PhoBERT nhận diện: `normalize(label, span_text, context_before)`
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..rules.loader import RuleSet, load_extraction_rules
from ..utils.text import fold, fold_keep


@dataclass
class Span:
    label: str
    start: int
    end: int
    text: str
    value: Any = None
    unit: Optional[str] = None
    confidence: float = 0.9
    meta: dict = field(default_factory=dict)


class RuleEngine:
    def __init__(self, rules: RuleSet | None = None):
        self.rules = rules or load_extraction_rules()
        r = self.rules
        self.thousand_units = {u.lower() for u in r["number"]["thousand_separator_units"]}
        # money
        self.money_cfg = r["money"]
        self.money_patterns = [(p, r.compile(p["regex"])) for p in self.money_cfg["patterns"]]
        self.money_suffix = r.compile(self.money_cfg["suffix_regex"])
        self.approx_cfg = self.money_cfg.get("approx_cues")
        self.approx_range_rx = (r.compile(self.approx_cfg["range_before_regex"])
                                if self.approx_cfg and self.approx_cfg.get("range_before_regex") else None)
        # area / dims / numeric
        self.area_cfg = r["area"]
        self.area_rx = r.compile(self.area_cfg["regex"])
        self.dims_cfg = r["dimensions"]
        self.dims_rx = r.compile(self.dims_cfg["regex"])
        self.numeric = [(e, r.compile(e["regex"])) for e in r["numeric_entities"]]
        self.number_words = {k.lower(): v for k, v in r.get("number_words", {}).items()}
        self.penalties = [(p, r.compile(p["before_regex"])) for p in r.get("numeric_context_penalties", [])]
        self.floor_cfg = r["floor_composite"]
        self.floor_rx = r.compile(self.floor_cfg["regex"])
        self.floor_require = [r.compile(x) for x in self.floor_cfg["require_any"]]
        # direction / location / transaction
        self.dir_cfg = r["direction"]
        self.dir_rx = r.compile(self.dir_cfg["regex"])
        self.dir_values = {fold(v): v for v in self.dir_cfg["values"]}
        self.dir_values.update({fold(k): v for k, v in self.dir_cfg.get("abbreviations", {}).items()})
        self.loc_cfg = r["location"]
        self.loc_rx = r.compile(self.loc_cfg["regex"])
        self.loc_prefix = r.compile(self.loc_cfg["prefix_regex"])
        self.trans_cfg = r["transaction"]
        self.trans_items = [(it["code"], r.lexicon_regex(it["patterns"])) for it in self.trans_cfg["items"]]
        # lexicons
        self.lexicons: dict[str, list[tuple[str, re.Pattern]]] = {}
        self.lexicon_conf: dict[str, float] = {}
        for cat, lex in r["lexicons"].items():
            self.lexicons[cat] = [(it["code"], r.lexicon_regex(it["patterns"])) for it in lex["items"]]
            self.lexicon_conf[cat] = lex.get("confidence", 0.85)

    # ================================================================ numbers
    def to_float(self, num: str, unit: str = "") -> Optional[float]:
        """'2,99' -> 2.99 ; '1.200' (triệu/m2) -> 1200 ; '1.234.567' -> 1234567."""
        s = (num or "").strip()
        if not s:
            return None
        seps = re.findall(r"[.,]", s)
        try:
            if len(seps) >= 2:
                if len(set(seps)) == 1:
                    return float(re.sub(r"[.,]", "", s))
                last = max(s.rfind("."), s.rfind(","))
                return float(re.sub(r"[.,]", "", s[:last]) + "." + s[last + 1:])
            if len(seps) == 1:
                int_part, frac = re.split(r"[.,]", s)
                if len(frac) == 3 and unit.lower() in self.thousand_units and len(int_part) <= 3:
                    return float(int_part + frac)
                return float(f"{int_part}.{frac}")
            return float(s)
        except ValueError:
            return None

    # ================================================================= money
    def _money_value(self, cfg: dict, m: re.Match) -> Optional[float]:
        mode = cfg["mode"]
        if mode == "int_plus":
            return int(m.group("a")) * cfg["multiplier_a"] + int(m.group("b")) * cfg["multiplier_b"]
        if mode == "decimal_join":
            return float(f"{m.group('a')}.{m.group('b')}") * cfg["multiplier"]
        if mode == "number":
            v = self.to_float(m.group("a"), cfg.get("number_unit", ""))
            return v * cfg["multiplier"] if v is not None else None
        if mode == "masked":
            # '2.x tỷ' -> 2 tỷ ; '5xx triệu' -> 500 triệu (giá che số, luôn là xấp xỉ)
            mult = cfg["unit_multipliers"][m.group("unit").lower()]
            a = self.to_float(m.group("a"))
            if a is None:
                return None
            if m.group("sep") or mult >= 1e9 or re.search(r"[.,]", m.group("a")):
                return a * mult
            return a * 10 ** len(m.group("x")) * mult
        raise ValueError(f"money mode không hỗ trợ: {mode}")

    def parse_money(self, text: str) -> Optional[tuple[float, str]]:
        for cfg, rx in self.money_patterns:
            m = rx.search(text)
            if m:
                v = self._money_value(cfg, m)
                if v:
                    return v, cfg["name"]
        return None

    def classify_money(self, context_before: str, per: Optional[str], context_after: str = "") -> tuple[str, float]:
        """Vai trò ngữ nghĩa của biểu thức tiền: price | deposit | other_money | rent_price | price_per_m2 | unknown_money.

        Cue gần biểu thức tiền nhất thắng ('giảm còn 3ty880' -> 'còn' -> price);
        cue phía sau ('136 triệu vốn ban đầu') ghi đè thành other_money.
        """
        cfg = self.money_cfg
        rc = cfg["role_confidence"]
        if per:
            p = fold(per).replace("/", "").strip()
            for rule in cfg["per_unit_roles"]:
                if any(p.startswith(c) for c in rule["contains"]):
                    return rule["role"], rule["confidence"]
        cues = cfg["cues"]
        after = re.split(r"[,.;!?]", fold_keep(context_after).lstrip(), maxsplit=1)[0][: cfg.get("after_window", 0)]
        after_other = any(re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", after)
                          for c in cues.get("after_other", [])) if after else False
        ctx = fold_keep(context_before)[-cfg["context_window"]:]
        near = ctx[-cfg["near_window"]:]

        def last(text: str, lst: list[str]) -> int:
            return max((text.rfind(c) + len(c) for c in lst if c in text), default=-1)

        pos = {"deposit": last(near, cues["deposit"]), "other_money": last(near, cues["other_money"]),
               "price": last(near, cues["price"])}
        role = max(pos, key=lambda k: (pos[k], {"deposit": 2, "other_money": 1, "price": 0}[k]))
        if pos[role] >= 0:
            if role != "deposit" and after_other:
                return "other_money", rc["other_near"]
            return role, rc[{"deposit": "deposit_near", "other_money": "other_near", "price": "price_near"}[role]]
        if after_other:
            return "other_money", rc["other_near"]
        if any(c in ctx for c in cues["deposit"]):
            return "deposit", rc["deposit_far"]
        return "unknown_money", rc["unknown"]

    def _money_spans(self, text: str) -> list[Span]:
        rc = self.money_cfg["role_confidence"]
        out: list[Span] = []
        taken: list[tuple[int, int]] = []
        for cfg, rx in self.money_patterns:
            for m in rx.finditer(text):
                if any(m.start() < e and m.end() > s for s, e in taken):
                    continue
                value = self._money_value(cfg, m)
                if not value or value <= 0:
                    continue
                suf = self.money_suffix.match(text, m.end())
                per = suf.group("per") if suf else None
                end = suf.end() if (suf and per) else m.end()
                role, conf = self.classify_money(text[max(0, m.start() - 60): m.start()], per, text[end: end + 30])
                approx = False
                if cfg.get("confidence"):  # mẫu tự mang độ tin cậy riêng (vd giá che số)
                    conf = min(conf, cfg["confidence"])
                    approx = True
                if self.approx_cfg and role in ("price", "unknown_money"):
                    b = fold_keep(text[max(0, m.start() - self.approx_cfg["window"]): m.start()])
                    if any(c in b for c in self.approx_cfg["before"]) or (
                            self.approx_range_rx and self.approx_range_rx.search(text[max(0, m.start() - 12): m.start()])):
                        conf = min(conf, self.approx_cfg["confidence"])
                        approx = True
                unit_used = (m.groupdict().get("u") or "").lower()
                if unit_used and unit_used in cfg.get("ambiguous_units", []):
                    conf = min(conf, rc["ambiguous_unit_max"])
                    if role != "price":
                        conf = min(conf, rc["ambiguous_unit_no_price_context"])
                label = "DEPOSIT" if role == "deposit" else "PRICE"
                out.append(Span(label, m.start(), end, text[m.start():end], value, "VND", conf,
                                {"role": role, "kind": cfg["name"], "per": per, "approx": approx}))
                taken.append((m.start(), end))
        return out

    # ================================================================== area
    def _area_spans(self, text: str) -> list[Span]:
        cfg = self.area_cfg
        out: list[Span] = []
        for m in self.area_rx.finditer(text):
            unit = m.group("u").lower()
            v = self.to_float(m.group("n"), "m2")
            if v is None or v <= 0:
                continue
            v *= cfg["unit_multipliers"].get(unit, 1)
            before = text[max(0, m.start() - 8): m.start()]
            if any(x in before for x in cfg["exclude_if_preceded_by"]):
                continue
            ctx = fold_keep(text[max(0, m.start() - cfg["context_window"]): m.start()])
            kind = "area"
            for k, cues in cfg["kind_cues"].items():
                if any(c in ctx for c in cues):
                    kind = k
            out.append(Span("AREA", m.start(), m.end(), m.group(0), v, "m2", cfg["confidence"], {"kind": kind}))

        d = self.dims_cfg
        for m in self.dims_rx.finditer(text):
            w, ln = self.to_float(m.group("w")), self.to_float(m.group("l"))
            if not w or not ln:
                continue
            w, ln = min(w, ln), max(w, ln)
            if w > d["max_width"] or ln > d["max_length"]:
                continue
            out.append(Span("FRONTAGE", m.start("w"), m.end("w"), m.group("w"), w, "m", d["confidence"], {"dims": True}))
            out.append(Span("LENGTH", m.start("l"), m.end("l"), m.group("l"), ln, "m", d["confidence"],
                            {"dims": True, "product": round(w * ln, 2)}))
        return out

    # ======================================================== numeric entities
    def _to_int(self, raw: str) -> Optional[int]:
        raw = raw.strip().lower()
        return int(raw) if raw.isdigit() else self.number_words.get(raw)

    def _apply_penalties(self, span: Span, text: str) -> Span:
        """Hạ độ tin cậy theo ngữ cảnh phía trước (đếm theo tầng, dự án nhiều loại căn, tòa nhà khác...)."""
        before = text[max(0, span.start - 60): span.start]
        for cfg, rx in self.penalties:
            if span.label in cfg["labels"] and rx.search(before):
                span.confidence = min(span.confidence, cfg["confidence"])
                span.meta["penalty"] = cfg.get("note")
        return span

    def _numeric_spans(self, text: str) -> list[Span]:
        out: list[Span] = []
        for cfg, rx in self.numeric:
            for m in rx.finditer(text):
                raw = m.group("n")
                v: Any = self._to_int(raw) if cfg.get("type") == "int" else self.to_float(raw)
                if v is None or not (cfg.get("min", 0) <= v <= cfg.get("max", float("inf"))):
                    continue
                out.append(self._apply_penalties(Span(cfg["label"], m.start(), m.end(), m.group(0), v, cfg.get("unit"),
                                                      cfg.get("confidence", 0.85), dict(cfg.get("meta", {}))), text))
        cfg = self.floor_cfg
        for m in self.floor_rx.finditer(text):
            s = m.group(0).strip()
            if not any(rx.search(s) for rx in self.floor_require):
                continue
            n = int(m.group("t") or 1) + int(m.group("l"))
            if cfg.get("count_mezzanine") and m.group("lung") is not None:
                n += int(m.group("lung") or 1)
            out.append(self._apply_penalties(Span("FLOOR", m.start(), m.end(), m.group(0), n, None, cfg["confidence"],
                                                  {"composite": True}), text))
        return out

    # ======================================================== categorical
    def _direction_spans(self, text: str) -> list[Span]:
        cc = self.dir_cfg["cue_confidence"]
        out = []
        for m in self.dir_rx.finditer(text):
            d = self.dir_values.get(fold(m.group("d")))
            if not d:
                continue
            whole = fold(m.group(0))
            conf = cc["default"]
            for k, c in cc.items():
                if k != "default" and k in whole:
                    conf = min(conf, c)
            out.append(Span("DIRECTION", m.start(), m.end(), m.group(0), d, None, conf))
        return out

    def _trim_location(self, s: str) -> Optional[str]:
        m = self.loc_prefix.match(s)
        if not m:
            return None
        prefix, rest = s[: m.end()], s[m.end():]
        dm = re.match(r"\d{1,2}", rest)
        if dm:
            return prefix + dm.group(0)
        if not self.loc_cfg.get("require_capitalized_name", True):
            return s.strip()
        words = []
        for w in rest.split(" "):
            if w and w[0].isupper():
                words.append(w)
            else:
                break
        return (prefix + " ".join(words)).strip() if words else None

    def _location_spans(self, text: str) -> list[Span]:
        out = []
        for m in self.loc_rx.finditer(text):
            loc = self._trim_location(m.group(0))
            if loc:
                out.append(Span("LOCATION", m.start(), m.start() + len(loc), loc, loc.strip(" ,.-"),
                                confidence=self.loc_cfg["confidence"]))
        return out

    def _lexicon_spans(self, text: str, items: list[tuple[str, re.Pattern]], label: str, conf: float) -> list[Span]:
        out = []
        for code, rx in items:
            for m in rx.finditer(text):
                out.append(Span(label, m.start(), m.end(), m.group(0), code, None, conf))
        return out

    # ================================================================ public
    def find_spans(self, text: str) -> list[Span]:
        spans: list[Span] = []
        spans += self._money_spans(text)
        spans += self._area_spans(text)
        spans += self._numeric_spans(text)
        spans += self._direction_spans(text)
        spans += self._location_spans(text)
        for cat, items in self.lexicons.items():
            spans += self._lexicon_spans(text, items, cat, self.lexicon_conf[cat])
        spans += self._lexicon_spans(text, self.trans_items, "TRANSACTION_TYPE", self.trans_cfg["confidence"])
        return self.remove_overlaps(spans)

    @staticmethod
    def remove_overlaps(spans: list[Span]) -> list[Span]:
        """Cùng nhãn mà chồng lấn -> giữ span dài/tin cậy hơn. Khác nhãn được phép chồng lấn
        (vd 'ô tô để trong nhà' vừa là POSITION vừa là FEATURE)."""
        spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
        kept: list[Span] = []
        for s in spans:
            clash = next((k for k in kept if k.label == s.label and s.start < k.end and s.end > k.start), None)
            if clash is None:
                kept.append(s)
            elif (s.end - s.start, s.confidence) > (clash.end - clash.start, clash.confidence):
                kept[kept.index(clash)] = s
        return sorted(kept, key=lambda s: s.start)

    def normalize(self, label: str, span_text: str, context_before: str = "") -> Optional[Span]:
        """Chuẩn hóa một đoạn do PhoBERT nhận diện thành giá trị có kiểu (hậu xử lý BERT)."""
        txt = span_text.strip()
        if label in ("PRICE", "DEPOSIT"):
            pm = self.parse_money(txt)
            if not pm:
                return None
            value, kind = pm
            per = re.search(r"/\s*(m2|m²|tháng|năm)", txt, re.I)
            role, conf = self.classify_money(context_before, per.group(0) if per else None)
            if label == "DEPOSIT":
                role = "deposit"
            return Span(label, 0, len(span_text), span_text, value, "VND", conf, {"role": role, "kind": kind})
        if label == "LOCATION":
            return Span(label, 0, len(span_text), span_text, txt.strip(" ,.-"), None, 0.8)
        if label in ("BEDROOM", "BATHROOM", "FLOOR") and re.fullmatch(r"\d{1,2}", txt):
            return Span(label, 0, len(span_text), span_text, int(txt), None, 0.6)
        if label in ("FRONTAGE", "LENGTH") and re.fullmatch(r"\d+(?:[.,]\d+)?\s*m?", txt):
            v = self.to_float(txt.rstrip(" m"))
            return Span(label, 0, len(span_text), span_text, v, "m", 0.7) if v else None
        # Bối cảnh tối thiểu để luật kích hoạt (vd 'ngang 4m' cần cụm 'ngang')
        probe = (context_before[-15:] + " " + span_text) if label in ("FRONTAGE", "LENGTH", "DIRECTION") else span_text
        offset = len(probe) - len(span_text)
        found = [s for s in self.find_spans(probe) if s.label == label and s.end > offset]
        if not found:
            return None
        best = max(found, key=lambda s: (s.end - s.start, s.confidence))
        best.start, best.end, best.text = 0, len(span_text), span_text
        return best
