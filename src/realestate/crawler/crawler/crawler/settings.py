"""Cấu hình Scrapy — cào có trách nhiệm (mục 10.1, 19 đề cương)."""
import os
import sys
from pathlib import Path

# cho phép import package realestate (đọc cấu hình, luật) khi chạy `scrapy crawl` từ thư mục này
_SRC = Path(__file__).resolve().parents[4]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from realestate.config import settings as _app  # noqa: E402

BOT_NAME = "crawler"
SPIDER_MODULES = ["crawler.spiders"]
NEWSPIDER_MODULE = "crawler.spiders"

# Nơi lưu file JSON (mặc định data/batches — cùng chỗ ingest đọc)
_raw = Path(_app.raw_data_path)
BATCH_DIR = os.getenv("BATCH_DIR", str(_raw if _raw.suffix == "" else _raw.parent))

ITEM_PIPELINES = {"crawler.pipelines.JsonBatchPipeline": 300}
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    ITEM_PIPELINES["crawler.pipelines.MongoPipeline"] = 400

# --- Lịch sự với máy chủ
USER_AGENT = "Mozilla/5.0 (compatible; RealEstateResearchBot/1.0; +student research, non-commercial)"
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 1.0
RANDOMIZE_DOWNLOAD_DELAY = True
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 30
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]
DOWNLOAD_TIMEOUT = 60
COOKIES_ENABLED = False
TELNETCONSOLE_ENABLED = False

# Ưu tiên xử lý trang chi tiết trước khi mở thêm trang danh sách (bám sát giới hạn tin)
DEPTH_PRIORITY = 0
SCHEDULER_PRIORITY_QUEUE = "scrapy.pqueues.DownloaderAwarePriorityQueue"

LOG_LEVEL = os.getenv("CRAWL_LOG_LEVEL", "INFO")
FEED_EXPORT_ENCODING = "utf-8"
