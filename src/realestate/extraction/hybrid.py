"""Pipeline trích xuất lai BERT/PhoBERT + Rule Engine (mục 9, 10).

Tin rao -> Tiền xử lý -> (header bán cấu trúc) + (PhoBERT NER) + (Rule Engine) -> hợp nhất -> Validate

3 cấu hình phục vụ ablation (mục 8.3, 13.4-E):
- rule   : chỉ Rule Engine
- bert   : chỉ PhoBERT, chuẩn hóa tối thiểu (đọc số), không kiểm tra ngữ cảnh
- hybrid : PhoBERT nhận diện đoạn, Rule Engine chuẩn hóa + kiểm tra ngữ cảnh + bổ sung đoạn bị bỏ sót

Nguyên tắc hợp nhất (mục 10.5, Phụ lục B):
- Không âm thầm ghi đè: giá trị khác nhau giữa các nguồn được lưu ở `alternatives`, trạng thái CONFLICT.
- Thiếu thì để None (UNKNOWN), không tự điền.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..preprocessing.cleaner import clean_text
from ..preprocessing.source_parsers import ParsedListing, parse_listing
from ..rules.loader import load_extraction_rules
from ..schemas import ExtractedField, FieldStatus, PropertyRecord, RawListing
from ..utils.text import canonical_province, fold
from .ner_model import PhoBertNER
from .rule_engine import RuleEngine, Span
from .validator import validate_record

log = logging.getLogger(__name__)

SOURCE_RANK = {"header": 3, "bert+rule": 2, "rule": 1, "bert": 1}
SINGLE_FIELDS = [
    "price_vnd", "price_per_m2_vnd", "deposit_vnd", "area_m2", "land_area_m2", "usable_area_m2",
    "frontage_m", "length_m", "bedrooms", "bathrooms", "floors", "floor_level", "direction",
    "legal_status", "property_type", "transaction_type",
]
LIST_FIELDS = {"POSITION": "positions", "FEATURE": "features"}


@dataclass
class Candidate:
    field: str
    value: Any
    source: str
    confidence: float
    span: Optional[Span] = None
    kind: Optional[str] = None
    order: int = 0
    role_rank: int = 0
    extra: dict = field(default_factory=dict)


class HybridExtractor:
    def __init__(self, mode: str = "hybrid", engine: RuleEngine | None = None, ner: PhoBertNER | None = None):
        self.engine = engine or RuleEngine()
        self.ner = ner
        self.rules = load_extraction_rules()
        self.mode = mode
        if mode in ("bert", "hybrid") and ner is None:
            log.warning("Mode=%s nhưng chưa có model PhoBERT -> chạy Rule-only", mode)
            self.mode = "rule"
        tol = self.rules["validation"]["conflict_tolerance"]
        self.tolerance: dict[str, float] = tol
        self.role_order: list[str] = self.rules["money"]["price_selection"]["role_order"]

    # ------------------------------------------------------------------ API
    def extract(self, raw: RawListing) -> PropertyRecord:
        text = clean_text(raw.text)
        parsed = parse_listing(text, raw.url)
        body = f"{parsed.title or ''} . {parsed.description}".strip(" .")

        cands: list[Candidate] = self._header_candidates(parsed)
        spans = self._detect(body)
        cands += self._span_candidates(spans, parsed)

        rec = PropertyRecord(
            id=raw.id, url=raw.url, source=parsed.source, title=parsed.title,
            listing_code=parsed.header.get("listing_code"), posted_date=parsed.header.get("posted_date"),
            description=parsed.description, extraction_mode=self.mode,
        )
        self._resolve(rec, cands, body)
        self._fill_location(rec, parsed, spans)
        for label, attr in LIST_FIELDS.items():
            codes = sorted({s.value for s in spans if s.label == label})
            setattr(rec, attr, codes)
            if codes:
                ev = [s for s in spans if s.label == label]
                rec.fields[attr] = ExtractedField(
                    name=attr, value=codes, source=self._span_source(ev[0]), confidence=max(s.confidence for s in ev),
                    evidence="; ".join(sorted({s.text for s in ev}))[:300],
                )
        validate_record(rec, self.rules)
        return rec

    # ----------------------------------------------------------- detection
    def _detect(self, body: str) -> list[Span]:
        if self.mode == "rule":
            spans = self.engine.find_spans(body)
            for s in spans:
                s.meta["source"] = "rule"
            return spans

        bert_raw = self.ner.predict(body)  # type: ignore[union-attr]
        bert_spans: list[Span] = []
        for b in bert_raw:
            ctx = body[max(0, b["start"] - 60): b["start"]]
            if self.mode == "bert":
                sp = self._naive_normalize(b["label"], b["text"])
            else:
                sp = self.engine.normalize(b["label"], b["text"], ctx)
            if sp is None:
                continue
            sp.start, sp.end, sp.text = b["start"], b["end"], b["text"]
            sp.confidence = round(min(1.0, (sp.confidence + b["confidence"]) / 2 if self.mode == "hybrid" else b["confidence"]), 3)
            sp.meta["source"] = "bert" if self.mode == "bert" else "bert+rule"
            bert_spans.append(sp)
        if self.mode == "bert":
            return bert_spans

        # hybrid: bổ sung đoạn luật tìm được mà BERT bỏ sót (và nhãn ngoài tập NER như FEATURE)
        rule_spans = self.engine.find_spans(body)
        merged = list(bert_spans)
        for rs in rule_spans:
            overlap = [b for b in bert_spans if rs.start < b.end and rs.end > b.start]
            if not overlap:
                rs.meta["source"] = "rule"
                rs.confidence *= 0.9
                merged.append(rs)
            else:
                for b in overlap:
                    if b.label != rs.label and rs.label not in ("FEATURE",):
                        # BERT và luật gán nhãn khác nhau cho cùng đoạn -> ghi nhận xung đột
                        b.meta.setdefault("label_conflict", []).append({"rule_label": rs.label, "value": rs.value})
                    if rs.label == "FEATURE":
                        rs.meta["source"] = "rule"
                        merged.append(rs)
        return sorted(merged, key=lambda s: s.start)

    def _naive_normalize(self, label: str, text: str) -> Optional[Span]:
        """Chuẩn hóa tối thiểu cho cấu hình BERT-only: không dùng ngữ cảnh/từ điển."""
        if label in ("PRICE", "DEPOSIT"):
            pm = self.engine.parse_money(text)
            if not pm:
                return None
            return Span(label, 0, len(text), text, pm[0], "VND", 0.8,
                        {"role": "deposit" if label == "DEPOSIT" else "price"})
        if label in ("LOCATION", "DIRECTION", "LEGAL_STATUS", "POSITION", "PROPERTY_TYPE", "TRANSACTION_TYPE"):
            return Span(label, 0, len(text), text, text.strip(), None, 0.8)
        m = re.search(r"\d+(?:[.,]\d+)?", text)
        if not m:
            return None
        v = self.engine.to_float(m.group(0), "m2" if label == "AREA" else "")
        if label in ("BEDROOM", "BATHROOM", "FLOOR"):
            v = int(v) if v is not None else None
        return Span(label, 0, len(text), text, v, None, 0.8) if v is not None else None

    @staticmethod
    def _span_source(s: Span) -> str:
        return s.meta.get("source", "rule")

    # ---------------------------------------------------------- candidates
    def _header_candidates(self, p: ParsedListing) -> list[Candidate]:
        h = p.header
        out: list[Candidate] = []
        conf = 0.95

        def add(field_: str, value: Any, raw: str | None = None):
            if value is not None:
                out.append(Candidate(field_, value, "header", conf, extra={"raw": raw}))

        add("transaction_type", h.get("transaction_type"))
        add("property_type", h.get("property_type"))
        if h.get("price_text"):
            pm = self.engine.parse_money(h["price_text"])
            if pm:
                field_ = "price_vnd"
                out.append(Candidate(field_, pm[0], "header", conf, extra={"raw": h["price_text"]}))
        if h.get("price_per_m2_text"):
            pm = self.engine.parse_money(h["price_per_m2_text"])
            if pm:
                add("price_per_m2_vnd", pm[0], h["price_per_m2_text"])
        if h.get("area"):
            a = self.engine.to_float(h["area"], "m2")
            add("area_m2", a * h.get("area_multiplier", 1) if a is not None else None, h["area"])
        # trường dạng chữ trong khối chi tiết -> chuẩn hóa bằng chính Rule Engine
        if h.get("legal_text"):
            sp = self.engine.normalize("LEGAL_STATUS", h["legal_text"])
            if sp:
                add("legal_status", sp.value, h["legal_text"])
        if h.get("direction_text"):
            sp = self.engine.normalize("DIRECTION", "hướng " + h["direction_text"])
            if sp:
                add("direction", sp.value, h["direction_text"])
        if h.get("dims_text"):
            dims = [s for s in self.engine.find_spans(h["dims_text"]) if s.label in ("FRONTAGE", "LENGTH")]
            for s in dims:
                add("frontage_m" if s.label == "FRONTAGE" else "length_m", s.value, h["dims_text"])
        if h.get("bedrooms"):
            add("bedrooms", int(h["bedrooms"]), h["bedrooms"])
        if h.get("bathrooms"):
            add("bathrooms", int(h["bathrooms"]), h["bathrooms"])
        if h.get("floors"):
            n = int(h["floors"])
            # Với căn hộ, "Tầng : 15" là vị trí tầng, không phải số tầng
            add("floor_level" if h.get("property_type") == "apartment" else "floors", n, h["floors"])
        return out

    def _span_candidates(self, spans: list[Span], parsed: ParsedListing) -> list[Candidate]:
        out: list[Candidate] = []
        is_rent = parsed.header.get("transaction_type") == "rent"
        ignore = set(self.rules["validation"].get("ignore_text_fields_for_types", {})
                     .get(parsed.header.get("property_type"), []))
        for i, s in enumerate(spans):
            src = self._span_source(s)

            def c(f, v=s.value, _s=s, _src=src, _i=i, **kw):
                if f not in ignore:
                    out.append(Candidate(f, v, _src, _s.confidence, span=_s, order=_i, **kw))
            if s.label == "PRICE":
                role = s.meta.get("role", "price")
                if role == "price_per_m2":
                    c("price_per_m2_vnd")
                elif role == "rent_price":
                    if is_rent:
                        c("price_vnd", role_rank=5)
                elif role in self.role_order:
                    c("price_vnd", role_rank=len(self.role_order) - self.role_order.index(role))
            elif s.label == "DEPOSIT":
                c("deposit_vnd")
            elif s.label == "AREA":
                kind = s.meta.get("kind", "area")
                if kind == "land":
                    c("land_area_m2", kind=kind)
                elif kind == "usable":
                    c("usable_area_m2", kind=kind)
                c("area_m2", kind=kind, role_rank={"area": 3, "land": 2, "usable": 1}[kind])
            elif s.label == "LENGTH" and s.meta.get("product"):
                c("length_m")
                out.append(Candidate("area_m2", s.meta["product"], src, 0.5, span=s, order=i, kind="dims", role_rank=0))
            elif s.label == "FLOOR":
                c("floor_level" if s.meta.get("floor_level") else "floors",
                  role_rank=2 if s.meta.get("composite") else 1)
            elif s.label in ("FRONTAGE", "LENGTH"):
                c("frontage_m" if s.label == "FRONTAGE" else "length_m", role_rank=0 if s.meta.get("dims") else 1)
            elif s.label == "BEDROOM":
                c("bedrooms")
            elif s.label == "BATHROOM":
                c("bathrooms")
            elif s.label == "DIRECTION":
                c("direction")
            elif s.label == "LEGAL_STATUS":
                # ưu tiên loại sổ cụ thể hơn "co_so" (chỉ nói chung chung)
                c("legal_status", role_rank=0 if s.value == "co_so" else 1)
            elif s.label == "PROPERTY_TYPE":
                c("property_type")
            elif s.label == "TRANSACTION_TYPE":
                c("transaction_type")
        return out

    # ------------------------------------------------------------- resolve
    def _resolve(self, rec: PropertyRecord, cands: list[Candidate], body: str) -> None:
        by_field: dict[str, list[Candidate]] = {}
        for c in cands:
            by_field.setdefault(c.field, []).append(c)
        for f in SINGLE_FIELDS:
            lst = by_field.get(f)
            if not lst:
                continue
            lst.sort(key=lambda c: (-SOURCE_RANK.get(c.source, 0), -c.role_rank, -c.confidence, c.order))
            best = lst[0]
            alternatives = []
            status = FieldStatus.OK
            for other in lst[1:]:
                if self._same(f, best.value, other.value):
                    continue
                alt = {"value": other.value, "source": other.source, "confidence": round(other.confidence, 3),
                       "evidence": other.span.text if other.span else other.extra.get("raw")}
                if alt not in alternatives:
                    alternatives.append(alt)
                # mâu thuẫn đáng kể giữa 2 nguồn khác nhau, cùng loại ngữ nghĩa
                if (other.source != best.source and other.confidence >= 0.8 and other.kind in (None, "area")
                        and best.kind in (None, "area") and other.role_rank >= best.role_rank - 1
                        and f not in ("property_type", "transaction_type", "legal_status")):
                    status = FieldStatus.CONFLICT
            if best.span is not None and best.span.meta.get("label_conflict"):
                status = FieldStatus.CONFLICT
                alternatives += best.span.meta["label_conflict"]
            if best.confidence < 0.5 and status == FieldStatus.OK:
                status = FieldStatus.NEEDS_REVIEW
            sp = best.span
            ev = body[max(0, sp.start - 40): sp.end + 40] if sp else best.extra.get("raw")
            rec.fields[f] = ExtractedField(
                name=f, value=best.value, unit=(sp.unit if sp else None), raw=(sp.text if sp else best.extra.get("raw")),
                evidence=ev, start=sp.start if sp else None, end=sp.end if sp else None, source=best.source,
                confidence=round(best.confidence, 3), status=status, alternatives=alternatives[:5],
            )
            setattr(rec, f, best.value)

    def _same(self, f: str, a: Any, b: Any) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            tol = self.tolerance.get(f, 0.05)
            return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)
        return a == b

    # ------------------------------------------------------------ location
    def _fill_location(self, rec: PropertyRecord, parsed: ParsedListing, spans: list[Span]) -> None:
        loc = parsed.location
        rec.province = canonical_province(loc.get("province"))
        rec.district = loc.get("district")
        rec.ward = loc.get("ward")
        rec.street = loc.get("street")
        rec.project = loc.get("project")
        rec.address = loc.get("address")
        src = "header" if rec.province else None
        # bổ sung từ mô tả khi header thiếu
        for s in spans:
            if s.label != "LOCATION":
                continue
            k = fold(s.value)
            if not rec.street and (k.startswith("duong ") or k.startswith("pho ")):
                rec.street = s.value
            elif not rec.district and (k.startswith("quan ") or k.startswith("huyen ") or k.startswith("q.")):
                rec.district = s.value
                src = src or "rule"
            elif not rec.ward and (k.startswith("phuong ") or k.startswith("xa ") or k.startswith("p.")):
                rec.ward = s.value
            elif not rec.province and (k.startswith("tinh ") or k.startswith("tp") or k.startswith("thanh pho ")):
                rec.province = canonical_province(s.value)
                src = src or "rule"
        if rec.province or rec.district:
            rec.fields["location"] = ExtractedField(
                name="location",
                value={"province": rec.province, "district": rec.district, "ward": rec.ward, "street": rec.street},
                source=src or "rule", confidence=0.95 if src == "header" else 0.6,
            )
