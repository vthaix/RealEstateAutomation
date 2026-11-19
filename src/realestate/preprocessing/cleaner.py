"""Tiền xử lý: làm sạch ký tự, ẩn thông tin cá nhân, khử trùng lặp (mục 8.2, 10.1, 19)."""
from __future__ import annotations

import hashlib
import re

from ..schemas import RawListing
from ..utils.text import collapse_ws, fold, nfc

# Số điện thoại VN: 09xx xxx xxx, 0912.345.678, +84 912 345 678, 0912***678 ...
_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
_PHONE_MASKED = re.compile(r"(?<!\d)0\d{2,4}[\s.]?\*{2,}[\s.]?\d{0,4}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_ZERO_WIDTH = re.compile(r"[​-‏﻿]")
_BULLETS = re.compile(r"[•●▪►✓✔☑❖➤⭐🔥✅👉📍💥🌟]+")


def mask_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE_MASKED.sub("[PHONE]", text)
    text = _PHONE.sub("[PHONE]", text)
    return text


def clean_text(text: str, keep_bullets: bool = False) -> str:
    """Chuẩn hóa Unicode NFC, bỏ ký tự vô hình, gộp khoảng trắng — giữ nguyên nội dung."""
    text = nfc(text)
    text = _ZERO_WIDTH.sub("", text)
    if not keep_bullets:
        text = _BULLETS.sub(" ", text)
    text = text.replace("m²", "m2").replace("㎡", "m2")
    return collapse_ws(mask_pii(text))


def content_hash(text: str) -> str:
    """Hash nội dung đã fold để phát hiện tin trùng / gần trùng (bỏ số liệu nhiễu như mã tin)."""
    k = fold(text)
    k = re.sub(r"ma tin : \d+|ngay dang : [\d/]+|\d+ ngay truoc", " ", k)
    k = re.sub(r"[^a-z0-9 ]", " ", k)
    k = collapse_ws(k)
    return hashlib.sha1(k.encode()).hexdigest()


def deduplicate(listings: list[RawListing]) -> tuple[list[RawListing], int]:
    seen_url: set[str] = set()
    seen_hash: set[str] = set()
    out: list[RawListing] = []
    for li in listings:
        url = li.url.split("?")[0]
        h = content_hash(li.text)
        if url in seen_url or h in seen_hash:
            continue
        seen_url.add(url)
        seen_hash.add(h)
        out.append(li)
    return out, len(listings) - len(out)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Tách câu, trả về (start, end) để giữ nguyên offset trên văn bản gốc."""
    spans = []
    start = 0
    for m in re.finditer(r"(?<=[.!?;])\s+(?=[A-ZĐÀ-Ỹ\-–])|\s+-\s+", text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans
