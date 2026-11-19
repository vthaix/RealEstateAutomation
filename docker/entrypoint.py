"""Container `app` = LUỒNG 3 (web gợi ý). Chỉ phục vụ web/API, không tự cào hay xử lý dữ liệu.

Luồng 1 (cào) và luồng 2 (xử lý) chạy ở terminal:
    docker compose run --rm app python scripts/crawl.py
    docker compose run --rm app python scripts/process.py
Web tự nạp lại Knowledge Graph khi luồng 2 chạy xong.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app/src")
from realestate.config import settings  # noqa: E402

if not Path(settings.kg_path).exists():
    print("CHƯA CÓ DỮ LIỆU ĐÃ XỬ LÝ. Web vẫn chạy nhưng chưa gợi ý được.\n"
          "Chạy luồng 2 ở terminal: docker compose run --rm app python scripts/process.py", flush=True)

os.execvp("uvicorn", ["uvicorn", "realestate.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"])
