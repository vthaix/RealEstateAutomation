"""Tiện ích xử lý chuỗi tiếng Việt."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt: 'Quận Bình Thạnh' -> 'Quan Binh Thanh'."""
    text = nfc(text).replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def fold(text: str) -> str:
    """Khóa so khớp: bỏ dấu, viết thường, gộp khoảng trắng."""
    return collapse_ws(strip_accents(text).lower())


def fold_keep(text: str) -> str:
    """Như fold() nhưng KHÔNG cắt khoảng trắng đầu/cuối (giữ ranh giới khi so cue ngữ cảnh)."""
    return _WS.sub(" ", strip_accents(text).lower())


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", fold(text)).strip("_")


# ---------------------------------------------------------------------------
# Chuẩn hóa địa danh
# ---------------------------------------------------------------------------
_PROVINCE_ALIASES = {
    "ho chi minh": "Hồ Chí Minh", "tp ho chi minh": "Hồ Chí Minh", "tphcm": "Hồ Chí Minh",
    "hcm": "Hồ Chí Minh", "sai gon": "Hồ Chí Minh", "saigon": "Hồ Chí Minh", "tp hcm": "Hồ Chí Minh",
    "ha noi": "Hà Nội", "hn": "Hà Nội", "da nang": "Đà Nẵng", "hai phong": "Hải Phòng",
    "can tho": "Cần Thơ",
}
_ADMIN_PREFIX = re.compile(
    r"^(thanh pho|tinh|tp\.?|quan|huyen|thi xa|thi tran|phuong|xa|q\.?|p\.?|h\.?)\s+", re.I
)


def location_key(name: str) -> str:
    """Khóa so khớp địa danh bỏ tiền tố hành chính.

    'Quận 3' -> '3', 'Q.3' -> '3', 'Huyện An Dương' -> 'an duong',
    'Thành phố Hồ Chí Minh' -> 'ho chi minh'.
    """
    k = fold(name).replace(".", " ").strip()
    k = collapse_ws(k)
    # Q3 / P12 viết liền
    m = re.fullmatch(r"(q|p|quan|phuong)\s*(\d{1,2})", k)
    if m:
        return m.group(2)
    prev = None
    while prev != k:
        prev = k
        k = _ADMIN_PREFIX.sub("", k).strip()
    k = re.sub(r"^0+(\d)", r"\1", k)  # 'Phường 05' -> '5'
    return k


def canonical_province(name: str | None) -> str | None:
    if not name:
        return None
    k = location_key(name)
    if k in _PROVINCE_ALIASES:
        return _PROVINCE_ALIASES[k]
    cleaned = re.sub(r"^(Thành phố|Thành Phố|Tỉnh|TP\.?)\s+", "", collapse_ws(name), flags=re.I)
    return cleaned or None
