"""Lưu tin cào được.

JsonBatchPipeline (mặc định): mỗi web 1 file data/batches/<site>_<YYYYmmdd_HHMMSS>.json
  - ghi dần ra <file>.jsonl.part trong lúc cào (mất điện/Ctrl+C vẫn còn dữ liệu đã cào)
  - khi kết thúc chuyển thành mảng JSON đúng định dạng bộ dữ liệu gốc: [{_id, url, text, ...}]
  - chặn trùng URL trong lần chạy và giới hạn số tin mỗi web (limit)
MongoPipeline (tùy chọn): bật khi truyền -s MONGO_URI=mongodb://...
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from scrapy.exceptions import DropItem

log = logging.getLogger(__name__)


def _norm(url: str) -> str:
    return url.split("?")[0].split("#")[0].rstrip("/")


class JsonBatchPipeline:
    def open_spider(self, spider):
        self.out_dir = Path(spider.settings.get("BATCH_DIR"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.files: dict[str, object] = {}
        self.paths: dict[str, Path] = {}
        self.counts: dict[str, int] = {}
        self.seen: set[str] = set()

    def _handle(self, site: str):
        if site not in self.files:
            p = self.out_dir / f"{site}_{self.stamp}.jsonl.part"
            self.paths[site] = p
            self.files[site] = open(p, "a", encoding="utf-8")
            self.counts[site] = 0
        return self.files[site]

    def process_item(self, item, spider):
        site, url = item["site"], item["url"]
        key = _norm(url)
        if key in self.seen:
            raise DropItem(f"Trùng URL: {url}")
        f = self._handle(site)
        if self.counts[site] >= spider.limit:
            raise DropItem(f"[{site}] đã đủ {spider.limit} tin")
        self.seen.add(key)
        row = {"_id": hashlib.sha1(key.encode()).hexdigest()[:24], "url": url, "text": item["text"],
               "source": site, "crawled_at": item["crawled_at"]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        self.counts[site] += 1
        spider.on_saved(site, self.counts[site])
        return item

    def close_spider(self, spider):
        for site, f in self.files.items():
            f.close()
            part = self.paths[site]
            rows = [json.loads(line) for line in part.read_text(encoding="utf-8").splitlines() if line.strip()]
            final = part.with_name(part.name.replace(".jsonl.part", ".json"))
            final.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            part.unlink()
            log.info("[%s] Đã lưu %d tin -> %s", site, len(rows), final)


class MongoPipeline:
    """Tùy chọn: lưu thêm vào MongoDB (như phiên bản cũ). Bật bằng -s MONGO_URI=mongodb://127.0.0.1:27017"""

    def open_spider(self, spider):
        from pymongo import MongoClient

        self.client = MongoClient(spider.settings.get("MONGO_URI"))
        db = self.client[spider.settings.get("MONGO_DB", "RealEasteDB")]
        self.collection = db[spider.settings.get("MONGO_COLLECTION", "text")]
        self.collection.create_index("url", unique=True)

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
        self.collection.update_one({"url": item["url"]},
                                   {"$set": {"text": item["text"], "source": item["site"],
                                             "crawled_at": item["crawled_at"]}}, upsert=True)
        return item
