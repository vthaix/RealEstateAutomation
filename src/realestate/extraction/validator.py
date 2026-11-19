"""Kiểm tra hợp lệ (mục 10.4-10.5). Ngưỡng lấy từ extraction_rules.json -> validation.

Giá trị bất thường KHÔNG bị sửa; chỉ đổi trạng thái sang NEEDS_REVIEW và ghi cảnh báo.
"""
from __future__ import annotations

from ..rules.loader import RuleSet, load_extraction_rules
from ..schemas import FieldStatus, PropertyRecord


def validate_record(rec: PropertyRecord, rules: RuleSet | None = None) -> PropertyRecord:
    rules = rules or load_extraction_rules()
    ranges: dict[str, list[float]] = rules["validation"]["ranges"]

    for f, (lo, hi) in ranges.items():
        key = f
        if f == "rent_price_vnd":
            if rec.transaction_type != "rent":
                continue
            key = "price_vnd"
        elif f == "price_vnd" and rec.transaction_type == "rent":
            continue
        v = getattr(rec, key, None)
        if v is None:
            continue
        if not (lo <= v <= hi):
            rec.warnings.append(f"{key}={v} ngoài miền hợp lệ [{lo}, {hi}]")
            if key in rec.fields:
                rec.fields[key].status = FieldStatus.NEEDS_REVIEW
                rec.fields[key].note = "Giá trị ngoài miền hợp lệ"

    # Nhất quán giá - diện tích (chỉ kiểm tra, không tự sửa)
    if rec.price_vnd and rec.area_m2 and rec.transaction_type != "rent":
        ppm = rec.price_vnd / rec.area_m2
        by_type = rules["validation"].get("price_per_m2_by_type", {})
        lo, hi = by_type.get(rec.property_type) or ranges.get("price_per_m2_vnd", [0, float("inf")])
        if not (lo <= ppm <= hi):
            rec.warnings.append(f"Đơn giá {ppm:,.0f} VND/m2 bất thường -> cần kiểm tra giá hoặc diện tích")
            for k in ("price_vnd", "area_m2"):
                if k in rec.fields and rec.fields[k].status == FieldStatus.OK:
                    rec.fields[k].status = FieldStatus.NEEDS_REVIEW
        if rec.price_per_m2_vnd is None:
            rec.price_per_m2_vnd = round(ppm)

    if rec.bathrooms and rec.bedrooms and rec.bathrooms > rec.bedrooms * 3:
        rec.warnings.append("Số WC lớn bất thường so với số phòng ngủ")
    if rec.frontage_m and rec.length_m and rec.area_m2:
        prod = rec.frontage_m * rec.length_m
        if prod > rec.area_m2 * 3 or prod < rec.area_m2 / 10:
            rec.warnings.append("Kích thước ngang x dài không khớp diện tích")
    for f in rec.fields.values():
        if f.status == FieldStatus.CONFLICT:
            rec.warnings.append(f"Xung đột giá trị ở trường {f.name}: giữ {f.value}, phương án khác {[a.get('value') for a in f.alternatives]}")
    return rec
