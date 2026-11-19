"""Cào tin rao -> data/batches/<web>_<thời gian>.json (cấu hình web: src/realestate/rules/crawl_sites.json).

    python scripts/crawl.py                              # các web đang bật, cào HẾT tin (không giới hạn)
    python scripts/crawl.py --limit 5000                 # tối đa 5000 tin mỗi web
    python scripts/crawl.py --sites bds --limit 100      # thử nhanh
    python scripts/crawl.py --ingest                     # cào xong tự trích xuất bổ sung vào KG (+ Neo4j nếu có)
    python scripts/crawl.py --jobdir data/crawl_job      # có thể Ctrl+C rồi chạy lại để cào tiếp
    python scripts/crawl.py --finalize                   # đổi file .jsonl.part còn sót (bị dừng đột ngột) thành .json

Trong lúc cào, tin được ghi dần vào <web>_<thời gian>.jsonl.part; cào xong (hoặc Ctrl+C MỘT lần để dừng êm)
file tự đổi thành .json. Nhấn Ctrl+C hai lần / tắt máy -> file .part ở lại -> dùng --finalize.
"""
import json
import argparse
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT = ROOT / "src" / "realestate" / "crawler" / "crawler"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=None, help="vd: bds,radanhadat (mặc định: các web enabled)")
    ap.add_argument("--limit", type=int, default=None, help="số tin tối đa mỗi web; 0 hoặc bỏ trống = cào hết (theo default_limit trong crawl_sites.json)")
    ap.add_argument("--out", default=None, help="thư mục lưu JSON (mặc định data/batches)")
    ap.add_argument("--jobdir", default=None, help="thư mục lưu trạng thái để tạm dừng / chạy tiếp")
    ap.add_argument("--finalize", action="store_true", help="chỉ chuyển các *.jsonl.part còn sót thành .json rồi thoát")
    ap.add_argument("--ingest", action="store_true", help="cào xong chạy ingest --append (+ export Neo4j nếu kết nối được)")
    args = ap.parse_args()

    if args.out:
        os.environ["BATCH_DIR"] = str(Path(args.out).resolve())

    if args.finalize:
        from realestate.config import settings as app_settings

        raw = Path(os.environ.get("BATCH_DIR") or app_settings.raw_data_path)
        batch_dir = raw if raw.suffix == "" else raw.parent
        parts = sorted(batch_dir.glob("*.jsonl.part"))
        if not parts:
            print("Không có file .part nào.")
        for part in parts:
            rows = []
            for line in part.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass  # dòng cuối ghi dở khi bị dừng đột ngột
            final = part.with_name(part.name.replace(".jsonl.part", ".json"))
            final.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            part.unlink()
            print(f"{part.name} -> {final.name}: {len(rows)} tin")
        print("Lưu ý: chỉ chạy --finalize khi KHÔNG còn crawler nào đang chạy.")
        sys.exit(0)
    sys.path.insert(0, str(SCRAPY_PROJECT))
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "crawler.settings")

    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    settings = get_project_settings()
    if args.jobdir:
        settings.set("JOBDIR", args.jobdir)
    kwargs = {k: v for k, v in {"sites": args.sites, "limit": args.limit}.items() if v}
    process = CrawlerProcess(settings, install_root_handler=False)
    process.crawl("realestate", **kwargs)
    process.start()

    if args.ingest:
        batch_dir = os.environ.get("BATCH_DIR") or settings.get("BATCH_DIR")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "ingest.py"), "--input", batch_dir, "--append"], check=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_neo4j.py")])
        if r.returncode != 0:
            print("(Bỏ qua export Neo4j: chưa kết nối được Neo4j)")
