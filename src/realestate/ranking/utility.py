"""Hàm kiểm tra điều kiện bắt buộc h_j(p) và hàm tiện ích s_i(p) ∈ [0,1] (mục 12.1, 12.3).

Tham số (dung sai, bảng tương đồng hướng/pháp lý/loại hình) lấy từ rules/query_rules.json.
Trả về None khi bản ghi thiếu dữ liệu -> không được coi là thỏa / không thỏa.
"""
from __future__ import annotations

from typing import Any, Optional

from ..kg.location import location_matches
from ..rules.loader import load_query_rules
from ..schemas import Criterion, FieldStatus, Operator, PropertyRecord
from ..utils.text import fold


def _rules():
    return load_query_rules()


def tolerance(attr: str, target: float) -> float:
    t = _rules()["tolerances"].get(attr, {"relative": 0.1, "absolute": 1})
    return max(t["absolute"], t["relative"] * abs(target or 0))


def _as_list(v: Any) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _expand_legal(codes: list[str]) -> set[str]:
    groups = _rules()["legal_similarity"]["groups"]
    out: set[str] = set()
    for c in codes:
        out |= set(groups.get(c, [c]))
    return out


def _direction_score(a: str, b: str) -> float:
    cfg = _rules()["direction_similarity"]
    ring = [fold(x) for x in cfg["ring"]]
    fa, fb = fold(a), fold(b)
    if fa == fb:
        return 1.0
    if fa in ring and fb in ring:
        d = abs(ring.index(fa) - ring.index(fb))
        if min(d, len(ring) - d) == 1:
            return cfg["adjacent_score"]
    return 0.0


def categorical_score(attr: str, value: Any, targets: list) -> Optional[float]:
    if value is None or value == [] or (attr == "property_type" and value == "other"):
        return None
    if attr == "direction":
        return max((_direction_score(value, t) for t in targets), default=0.0)
    if attr == "legal_status":
        return 1.0 if value in _expand_legal(targets) else 0.0
    if attr == "property_type":
        sim = _rules()["property_type_similarity"]
        return max((1.0 if value == t else sim.get(t, {}).get(value, 0.0) for t in targets), default=0.0)
    if attr in ("position", "feature"):
        vals = set(_as_list(value))
        return 1.0 if vals & set(targets) else None  # tin rao không nhắc ≠ không có
    return 1.0 if value in targets else 0.0


def numeric_target(c: Criterion) -> Optional[float]:
    v = c.target_value
    if isinstance(v, list):
        v = v[0] if v else None
    if isinstance(v, (int, float)):
        return float(v)
    return None


_FIELD_OF = {"price": "price_vnd", "area": "area_m2", "frontage": "frontage_m"}


def _needs_review(rec: PropertyRecord, attr: str) -> bool:
    f = rec.fields.get(_FIELD_OF.get(attr, attr))
    return f is not None and f.status == FieldStatus.NEEDS_REVIEW


def check_hard(c: Criterion, rec: PropertyRecord) -> Optional[bool]:
    """h_j(p): True / False / None (thiếu dữ liệu để xác minh)."""
    attr = c.attribute
    if attr == "location":
        targets = _as_list(c.target_value)
        res = [location_matches(rec, str(t)) for t in targets]
        if any(r is True for r in res):
            hit = True
        elif all(r is False for r in res):
            hit = False
        else:
            hit = None
        if c.operator == Operator.NOT_IN and hit is not None:
            return not hit
        return hit

    x = rec.get(attr)
    if attr in ("price", "area", "bedrooms", "bathrooms", "floors", "frontage"):
        if x is None or _needs_review(rec, attr):
            return None  # giá trị chưa đáng tin -> không coi là đã thỏa / vi phạm điều kiện bắt buộc
        t = numeric_target(c)
        op = c.operator
        if op == Operator.RANGE:
            return (c.target_min or 0) <= x <= (c.target_max if c.target_max is not None else float("inf"))
        if t is None:
            return None
        if op == Operator.GTE:
            return x >= t
        if op == Operator.LTE:
            return x <= t
        if op == Operator.EQ:
            return abs(x - t) < 1e-9 if attr in ("bedrooms", "bathrooms", "floors") else abs(x - t) <= tolerance(attr, t) / 2
        return abs(x - t) <= tolerance(attr, t)  # APPROX / UNRESOLVED

    targets = _as_list(c.target_value)
    s = categorical_score(attr, x, targets)
    if s is None:
        return None
    hit = s >= 1.0
    return (not hit) if c.operator == Operator.NOT_IN else hit


def utility(c: Criterion, rec: PropertyRecord) -> Optional[float]:
    """s_i(p) ∈ [0,1]; None = thiếu dữ liệu."""
    attr = c.attribute
    if attr == "location":
        h = check_hard(c, rec)
        return None if h is None else float(h)
    x = rec.get(attr)
    if attr in ("price", "area", "bedrooms", "bathrooms", "floors", "frontage"):
        if x is None:
            return None
        op = c.operator
        if op == Operator.RANGE:
            lo = c.target_min if c.target_min is not None else float("-inf")
            hi = c.target_max if c.target_max is not None else float("inf")
            if lo <= x <= hi:
                return 1.0
            bound = lo if x < lo else hi
            return max(0.0, 1 - abs(x - bound) / tolerance(attr, bound))
        t = numeric_target(c)
        if t is None or t == 0:
            return None
        if op == Operator.GTE:
            return min(1.0, x / t)                                      # s = min(1, x/t)
        if op == Operator.LTE:
            return 1.0 if x <= t else max(0.0, 1 - (x - t) / tolerance(attr, t))
        return max(0.0, 1 - abs(x - t) / tolerance(attr, t))           # s = max(0, 1 - |x-t|/δ)

    targets = _as_list(c.target_value)
    s = categorical_score(attr, x, targets)
    if s is None:
        return None
    return 1.0 - s if c.operator == Operator.NOT_IN else s
