"""Luồng xử lý tin rao: JSON thô -> tiền xử lý -> trích xuất lai -> kiểm tra -> Knowledge Graph."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from .config import settings
from .extraction import build_extractor
from .kg.store import KnowledgeGraph
from .preprocessing.cleaner import clean_text, content_hash, deduplicate
from .preprocessing.source_parsers import parse_listing
from .schemas import FieldStatus, PropertyRecord
from .utils.io import load_raw_listings, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

COVERAGE_FIELDS = ["province", "district", "property_type", "transaction_type", "price_vnd", "area_m2", "bedrooms",
                   "bathrooms", "floors", "frontage_m", "direction", "legal_status"]


def coverage_report(records: list[PropertyRecord]) -> dict:
    n = len(records) or 1
    cov = {f: round(sum(1 for r in records if getattr(r, f) not in (None, [], "")) / n, 4) for f in COVERAGE_FIELDS}
    status = Counter(fv.status.value for r in records for fv in r.fields.values())
    sources = Counter(fv.source for r in records for fv in r.fields.values())
    return {"records": len(records), "coverage": cov, "field_status": dict(status), "field_sources": dict(sources),
            "records_with_warnings": sum(1 for r in records if r.warnings)}


def run_ingest(raw_path: Optional[str | Path] = None, mode: Optional[str] = None, limit: Optional[int] = None,
               kg_path: Optional[str | Path] = None, out_path: Optional[str | Path] = None,
               append: bool = False) -> dict:
    """append=False: xây lại từ đầu (ghi đè). append=True: bổ sung tin mới vào dữ liệu + KG hiện có,
    bỏ qua tin trùng URL/nội dung với dữ liệu cũ."""
    raw_path = Path(raw_path or settings.raw_data_path)
    out_path = Path(out_path or settings.properties_path)
    kg_path = Path(kg_path or settings.kg_path)
    listings = load_raw_listings(raw_path)
    if limit:
        listings = listings[:limit]
    listings, n_dup = deduplicate(listings)

    existing: list[dict] = []
    n_old_dup = 0
    if append and out_path.exists() and kg_path.exists():
        existing = list(read_jsonl(out_path))
        seen_url = {r["url"].split("?")[0] for r in existing}
        seen_id = {r["id"] for r in existing}
        seen_hash = {content_hash(f"{r.get('title') or ''} {r.get('description') or ''}") for r in existing}
        fresh = []
        for li in listings:
            p = parse_listing(clean_text(li.text), li.url)
            h = content_hash(f"{p.title or ''} {p.description}")
            if li.url.split("?")[0] in seen_url or li.id in seen_id or h in seen_hash:
                n_old_dup += 1
                continue
            fresh.append(li)
        listings = fresh
    log.info("Tin mới cần xử lý: %d (trùng trong file: %d, trùng dữ liệu cũ: %d)", len(listings), n_dup, n_old_dup)

    extractor = build_extractor(mode)
    records: list[PropertyRecord] = []
    for i, li in enumerate(listings):
        try:
            records.append(extractor.extract(li))
        except Exception as e:  # không để 1 tin lỗi làm hỏng cả lô
            log.exception("Lỗi trích xuất tin %s: %s", li.id, e)
        if (i + 1) % 500 == 0:
            log.info("Đã xử lý %d/%d", i + 1, len(listings))

    new_rows = [r.model_dump(mode="json") for r in records]
    write_jsonl(out_path, existing + new_rows)

    kg = KnowledgeGraph.load(kg_path) if (append and existing) else KnowledgeGraph()
    kg.add_many(records)
    kg.save(kg_path)

    report = coverage_report(records)
    report.update({"duplicates_removed": n_dup, "duplicates_with_existing": n_old_dup,
                   "total_records_after": len(existing) + len(new_rows), "append": append,
                   "extraction_mode": extractor.mode, "kg": kg.stats(),
                   "needs_review": sum(1 for r in records for f in r.fields.values()
                                       if f.status != FieldStatus.OK)})
    return report
