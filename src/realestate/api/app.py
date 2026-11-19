"""LUỒNG 3 — REST API + giao diện web (Result Interface — mục 9).

Chạy:  uvicorn realestate.api.app:app --app-dir src   (sau khi đã chạy luồng 2: scripts/process.py)
Knowledge Graph được nạp lại tự động khi file data/kg/graph.pkl thay đổi.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..extraction import build_extractor
from ..kg.store import KnowledgeGraph
from ..ranking.recommender import Recommender, build_recommender
from ..schemas import ParsedQuery, RawListing, RecommendationResponse

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app = FastAPI(title="Hệ thống tìm kiếm & gợi ý nhà ở", version="1.0.0")


_cache: dict = {"mtime": None, "rec": None}
_lock = threading.Lock()


def get_recommender() -> Recommender:
    """Nạp KG một lần; tự nạp lại khi luồng 2 ghi đè graph.pkl (so thời điểm sửa file)."""
    p = Path(settings.kg_path)
    if not p.exists():
        raise HTTPException(503, "Chưa có dữ liệu đã xử lý. Hãy chạy luồng 2 ở terminal: python scripts/process.py")
    mtime = p.stat().st_mtime
    if _cache["rec"] is None or _cache["mtime"] != mtime:
        with _lock:
            if _cache["rec"] is None or _cache["mtime"] != mtime:
                _cache["rec"] = build_recommender(KnowledgeGraph.load(p))
                _cache["mtime"] = mtime
    return _cache["rec"]


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3, examples=["Tôi cần một căn nhà ở Quận 3, diện tích 60 m² thì càng tốt, phòng tắm có thể 2"])
    top_k: int = Field(10, ge=1, le=100)
    weighting: Literal["equal", "priority"] | None = None
    lambda_aux: float | None = Field(None, ge=0, le=0.5)
    parser: Literal["auto", "llm", "rule"] | None = None
    parsed_query: Optional[ParsedQuery] = Field(None, description="Gửi lại tiêu chí đã được người dùng xác nhận/hiệu chỉnh")


class ParseRequest(BaseModel):
    query: str
    parser: Literal["auto", "llm", "rule"] | None = None


class ExtractRequest(BaseModel):
    text: str
    url: str = ""
    mode: Literal["rule", "bert", "hybrid"] | None = None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/parse", response_model=ParsedQuery)
def parse(req: ParseRequest):
    return get_recommender().parser.parse(req.query, mode=req.parser)


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendRequest):
    rec = get_recommender()
    q = req.parsed_query or req.query
    return rec.recommend(q, top_k=req.top_k, weighting=req.weighting, lambda_aux=req.lambda_aux,
                         parser_mode=req.parser)


@app.post("/api/extract")
def extract(req: ExtractRequest):
    """Thử trích xuất một tin rao bất kỳ (demo nhánh PhoBERT + Rule Engine)."""
    ex = build_extractor(req.mode)
    rec = ex.extract(RawListing(id="adhoc", url=req.url, text=req.text))
    return rec.model_dump(mode="json")


@app.get("/api/properties/{pid}")
def get_property(pid: str):
    rec = get_recommender().kg.get_property(pid)
    if not rec:
        raise HTTPException(404, "Không tìm thấy")
    return {"property": rec.model_dump(mode="json"), "graph": get_recommender().kg.neighbors(pid)}


@app.get("/api/stats")
def stats():
    kg = get_recommender().kg
    return kg.stats()


@app.get("/api/locations")
def locations(q: str = "", limit: int = 30):
    from ..utils.text import fold

    gaz = get_recommender().kg.gazetteer()
    fq = fold(q)
    items = [g for g in gaz if g["type"] in ("Province", "District") and (not fq or fq in fold(g["name"] or ""))]
    return sorted(items, key=lambda g: -(g.get("count") or 0))[:limit]
