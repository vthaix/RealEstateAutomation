"""Đọc / ghi dữ liệu."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from ..schemas import RawListing


TEXT_KEYS = ("text", "content", "description", "body", "raw_text")
URL_KEYS = ("url", "link", "href", "source_url")
ID_KEYS = ("_id", "id", "listing_id", "code")


def _first(obj: dict, keys: tuple[str, ...]):
    return next((obj[k] for k in keys if obj.get(k) not in (None, "")), None)


def load_raw_listings(path: str | Path) -> list[RawListing]:
    """Đọc dữ liệu thô. Hỗ trợ:
    - JSON mảng (export MongoDB): [{"_id": {"$oid": ...}, "url": ..., "text": ...}, ...]
    - JSONL: mỗi dòng 1 object
    Tên trường linh hoạt: text|content|description, url|link, _id|id. Không có id -> sinh từ hash URL+nội dung
    (ổn định giữa các lần chạy, tránh trùng id khi nạp nhiều bộ data).
    """
    path = Path(path)
    if path.is_dir():  # thư mục chứa nhiều lô dữ liệu: đọc mọi .json / .jsonl theo thứ tự tên
        out_all: list[RawListing] = []
        for p in sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl"))):
            out_all += load_raw_listings(p)
        return out_all
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()
    stripped = raw.lstrip()
    if stripped.startswith("["):
        data = json.loads(raw)
    else:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    out: list[RawListing] = []
    for obj in data:
        text = _first(obj, TEXT_KEYS) or ""
        if not text.strip():
            continue
        url = _first(obj, URL_KEYS) or ""
        oid = _first(obj, ID_KEYS)
        if isinstance(oid, dict):
            oid = oid.get("$oid")
        if not oid:
            oid = "h" + hashlib.sha1(f"{url}|{text[:500]}".encode("utf-8")).hexdigest()[:20]
        source = url.split("/")[2] if url.count("/") >= 2 else None
        out.append(RawListing(id=str(oid), url=url, text=text, source=source))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
