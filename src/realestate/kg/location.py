"""So khớp địa danh giữa yêu cầu người dùng và bản ghi BĐS."""
from __future__ import annotations

import re
from typing import Optional

from ..utils.text import fold, location_key

_LEVEL_PREFIX = [
    (re.compile(r"^(?:quan|huyen|thi xa)\b|^q[\s.]*\d"), "district"),
    (re.compile(r"^(?:phuong|xa|thi tran)\b|^p[\s.]*\d"), "ward"),
    (re.compile(r"^(?:tinh|tp|thanh pho)\b"), "province"),
    (re.compile(r"^(?:duong|pho)\b"), "street"),
]


def location_level(target: str) -> Optional[str]:
    k = fold(target)
    for rx, lvl in _LEVEL_PREFIX:
        if rx.match(k):
            return lvl
    return None


def _keys(name: Optional[str]) -> set[str]:
    if not name:
        return set()
    return {location_key(name)}


def location_matches(rec, target: str) -> Optional[bool]:
    """True/False nếu xác định được, None nếu bản ghi thiếu dữ liệu địa điểm."""
    level = location_level(target)
    tkey = location_key(target)
    if not tkey:
        return None
    levels = {
        "province": rec.province, "district": rec.district, "ward": rec.ward,
        "street": rec.street, "project": rec.project,
    }
    if not any(levels.values()):
        return None
    if level:
        val = levels.get(level)
        if val is None:
            # thiếu cấp này: nếu tên trùng ở cấp khác vẫn coi là chưa xác minh được
            return None
        return tkey in _keys(val)
    # không có tiền tố: khớp bất kỳ cấp nào (vd "Thủ Đức", "Hà Nội", "Vinhomes")
    for lvl, val in levels.items():
        if val and (tkey in _keys(val) or (lvl in ("project", "street") and tkey in fold(val))):
            return True
    return False
