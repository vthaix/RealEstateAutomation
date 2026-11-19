"""Sinh giải thích từ dữ liệu & phép tính (mục 12.7) — không dùng LLM để bịa lý do."""
from __future__ import annotations

from typing import Any, Optional

from ..rules.loader import lexicon_labels
from ..schemas import Criterion, CriterionScore, PropertyRecord

ATTR_VI = {
    "location": "địa điểm", "price": "giá", "area": "diện tích", "bedrooms": "số phòng ngủ",
    "bathrooms": "số phòng tắm/WC", "floors": "số tầng", "frontage": "chiều ngang", "direction": "hướng",
    "legal_status": "pháp lý", "property_type": "loại hình", "transaction_type": "hình thức giao dịch",
    "position": "vị trí tiếp cận", "feature": "tiện ích",
}
OP_VI = {"EQ": "=", "GTE": "≥", "LTE": "≤", "APPROX": "≈", "RANGE": "trong khoảng", "IN_AREA": "thuộc",
         "IN_SET": "thuộc", "NOT_IN": "không thuộc", "HAS": "có", "UNRESOLVED": "≈ (chưa rõ)"}


def fmt_vnd(v: Optional[float]) -> str:
    if v is None:
        return "?"
    if v >= 1e9:
        return f"{v / 1e9:,.2f} tỷ".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{v / 1e6:,.0f} triệu".replace(",", ".")


def fmt_value(attr: str, v: Any) -> str:
    if v is None or v == []:
        return "chưa có dữ liệu"
    if attr == "price":
        return fmt_vnd(v)
    if attr == "area":
        return f"{v:g} m²"
    if attr == "frontage":
        return f"{v:g} m"
    labels = {"legal_status": "LEGAL_STATUS", "position": "POSITION", "feature": "FEATURE",
              "property_type": "PROPERTY_TYPE"}.get(attr)
    if labels:
        lab = lexicon_labels(labels)
        vals = v if isinstance(v, list) else [v]
        return ", ".join(lab.get(x, str(x)) for x in vals)
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return str(v)


def fmt_target(c: Criterion | CriterionScore, attr: str) -> str:
    op = c.operator if isinstance(c.operator, str) else c.operator.value
    if op == "RANGE":
        lo = getattr(c, "target_min", None)
        hi = getattr(c, "target_max", None)
        tgt = getattr(c, "target", None)
        if lo is None and isinstance(tgt, list) and len(tgt) == 2:
            lo, hi = tgt
        return f"{fmt_value(attr, lo)} – {fmt_value(attr, hi)}"
    t = c.target_value if isinstance(c, Criterion) else c.target
    return f"{OP_VI.get(op, op)} {fmt_value(attr, t)}"


def explain(rec: PropertyRecord, hard: list[Criterion], hard_status: dict[str, Optional[bool]],
            breakdown: list[CriterionScore], mentioned: set[str]) -> tuple[list[str], dict]:
    lines: list[str] = []
    ok = [f"{ATTR_VI.get(c.attribute, c.attribute)} {fmt_target(c, c.attribute)}"
          for c, k in zip(hard, hard_status) if hard_status[k] is True]
    if ok:
        lines.append("✔ Thỏa điều kiện bắt buộc: " + "; ".join(ok))
    unk = [f"{ATTR_VI.get(c.attribute, c.attribute)}" for c, k in zip(hard, hard_status) if hard_status[k] is None]
    if unk:
        lines.append("⚠ Chưa đủ dữ liệu để xác minh điều kiện bắt buộc: " + ", ".join(unk))

    known = [b for b in breakdown if b.utility is not None]
    for b in sorted(known, key=lambda b: -b.contribution)[:2]:
        if b.utility >= 0.5:
            lines.append(f"＋ {ATTR_VI.get(b.attribute, b.attribute)}: {fmt_value(b.attribute, b.property_value)} "
                         f"(mong muốn {fmt_target(b, b.attribute)}) → đóng góp {b.contribution:.2f}/{b.weight:.2f}")
    for b in known:
        if b.utility < 0.7:
            lines.append(f"－ {ATTR_VI.get(b.attribute, b.attribute)}: {fmt_value(b.attribute, b.property_value)} "
                         f"chưa sát mong muốn {fmt_target(b, b.attribute)} (tiện ích {b.utility:.2f})")
    missing = [ATTR_VI.get(b.attribute, b.attribute) for b in breakdown if b.utility is None]
    if missing:
        lines.append("？ Tin rao chưa có dữ liệu: " + ", ".join(missing) + " (tính mức trung tính, cần hỏi thêm người bán)")

    # Thông tin bổ trợ: hiển thị, không cộng điểm
    aux_attrs = ["price", "area", "bedrooms", "bathrooms", "floors", "frontage", "direction", "legal_status",
                 "position", "feature", "property_type"]
    auxiliary = {ATTR_VI[a]: fmt_value(a, rec.get(a)) for a in aux_attrs if a not in mentioned}
    if rec.warnings:
        auxiliary["cảnh báo dữ liệu"] = "; ".join(rec.warnings[:3])
    return lines, auxiliary
