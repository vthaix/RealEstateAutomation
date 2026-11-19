"""Data Collector (mục 10.1).

Crawler Scrapy: src/realestate/crawler/crawler/ (spider `realestate`), cấu hình web ở rules/crawl_sites.json.
Chạy: python scripts/crawl.py [--sites bds] [--limit 5000] [--ingest]
Kết quả: data/batches/<web>_<thời gian>.json -> scripts/ingest.py đọc trực tiếp.
"""
