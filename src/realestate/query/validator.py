"""Kiểm tra hậu kỳ kết quả phân tích yêu cầu (mục 11.1, 18 — chống LLM tự tạo giá trị).

- evidence_span phải là chuỗi con của câu người dùng; nếu không -> giảm tin cậy + cần xác nhận
- kiểm tra miền giá trị, quy đổi đơn vị giá bị nhầm (5 -> 5 tỷ) và gắn cờ
- HARD không mang preference_level; RANGE phải có min/max
- tiêu chí độ tin cậy thấp / UNRESOLVED -> needs_confirmation
"""
from __future__ import annotations

from ..rules.loader import load_extraction_rules, lexicon_labels
from ..schemas import Criterion, Group, Operator, ParsedQuery
from ..utils.text import fold

DIRECTIONS = {fold(d): d for d in load_extraction_rules()["direction"]["values"]}


def _validate_one(c: Criterion, qfold: str, warnings: list[str]) -> Criterion | None:
    ranges = load_extraction_rules()["validation"]["ranges"]
    # --- bằng chứng ngôn ngữ
    if not c.evidence_span or fold(c.evidence_span) not in qfold:
        c.confidence = round(min(c.confidence, 0.6) * 0.8, 2)
        c.needs_confirmation = True
        warnings.append(f"Tiêu chí '{c.attribute}' không có bằng chứng nguyên văn trong câu -> cần xác nhận")

    # --- chuẩn hóa giá trị theo thuộc tính
    if c.attribute == "price":
        for k in ("target_value", "target_min", "target_max"):
            v = getattr(c, k)
            if isinstance(v, (int, float)) and 0 < v < 100_000:
                # LLM/luật trả số "5" -> hiểu là tỷ, nhưng đánh dấu để người dùng xác nhận
                setattr(c, k, v * 1e9 if v < 1000 else v * 1e6)
                c.needs_confirmation = True
                c.note = (c.note or "") + " Đã quy đổi đơn vị giá, cần xác nhận."
        lo, hi = ranges["price_vnd"]
        v = c.target_value if isinstance(c.target_value, (int, float)) else (c.target_max or c.target_min)
        if isinstance(v, (int, float)) and not (lo <= v <= hi):
            warnings.append(f"Giá {v:,.0f} VND ngoài miền hợp lệ -> cần xác nhận")
            c.needs_confirmation = True
    elif c.attribute == "direction":
        vals = c.target_value if isinstance(c.target_value, list) else [c.target_value]
        norm = [DIRECTIONS.get(fold(str(v))) for v in vals]
        if None in norm:
            warnings.append(f"Hướng không hợp lệ: {vals}")
            return None
        c.target_value = norm if c.operator in (Operator.IN_SET, Operator.NOT_IN) else norm[0]
    elif c.attribute in ("legal_status", "position", "feature", "property_type"):
        cat = {"legal_status": "LEGAL_STATUS", "position": "POSITION", "feature": "FEATURE",
               "property_type": "PROPERTY_TYPE"}[c.attribute]
        valid = set(lexicon_labels(cat))
        vals = c.target_value if isinstance(c.target_value, list) else [c.target_value]
        kept = [v for v in vals if v in valid]
        if not kept:
            warnings.append(f"Giá trị '{vals}' không thuộc danh mục {c.attribute} -> bỏ")
            return None
        c.target_value = kept if (c.operator in (Operator.IN_SET, Operator.NOT_IN) or len(kept) > 1) else kept[0]
        if c.attribute in ("legal_status", "property_type") and c.operator not in (Operator.IN_SET, Operator.NOT_IN):
            c.operator = Operator.IN_SET
            c.target_value = kept
    elif c.attribute in ("bedrooms", "bathrooms", "floors"):
        for k in ("target_value", "target_min", "target_max"):
            v = getattr(c, k)
            if isinstance(v, float):
                setattr(c, k, int(round(v)))

    # --- toán tử
    if c.operator == Operator.RANGE:
        if c.target_min is None or c.target_max is None:
            if isinstance(c.target_value, (int, float)):
                c.operator = Operator.APPROX
            else:
                return None
        elif c.target_min > c.target_max:
            c.target_min, c.target_max = c.target_max, c.target_min
    if c.operator in (Operator.GTE, Operator.LTE, Operator.APPROX, Operator.EQ) and c.target_value is None:
        c.target_value = c.target_min if c.operator == Operator.GTE else c.target_max
        if c.target_value is None:
            return None
    if c.attribute == "location" and c.operator not in (Operator.IN_AREA, Operator.NOT_IN):
        c.operator = Operator.IN_AREA

    # --- nhóm / mức ưu tiên
    if c.group == Group.HARD:
        c.preference_level = None
    if c.operator == Operator.UNRESOLVED or c.confidence < 0.6:
        c.needs_confirmation = True
    c.confidence = max(0.0, min(1.0, c.confidence))
    return c


def validate_parsed_query(pq: ParsedQuery) -> ParsedQuery:
    qfold = fold(pq.query)
    out: list[Criterion] = []
    seen: set[tuple] = set()
    for c in pq.criteria:
        c2 = _validate_one(c, qfold, pq.warnings)
        if c2 is None:
            continue
        key = (c2.attribute, c2.operator, str(c2.target_value), c2.target_min, c2.target_max)
        if key in seen:
            continue
        seen.add(key)
        out.append(c2)
    pq.criteria = out
    return pq
