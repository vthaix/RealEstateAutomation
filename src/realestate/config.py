"""Cấu hình tập trung, đọc từ biến môi trường / file .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv là tùy chọn
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Settings:
    # --- Dữ liệu ---
    raw_data_path: Path = field(default_factory=lambda: _path("RAW_DATA_PATH", "data/batches"))
    processed_dir: Path = field(default_factory=lambda: _path("PROCESSED_DIR", "data/processed"))
    annotations_dir: Path = field(default_factory=lambda: _path("ANNOTATIONS_DIR", "data/annotations"))

    # --- Trích xuất ---
    # rule | bert | hybrid  (tương ứng 3 cấu hình Rule-only / BERT-only / BERT + Rule Engine)
    extraction_mode: str = field(default_factory=lambda: os.getenv("EXTRACTION_MODE", "hybrid"))
    ner_base_model: str = field(default_factory=lambda: os.getenv("NER_BASE_MODEL", "vinai/phobert-base-v2"))
    ner_model_dir: Path = field(default_factory=lambda: _path("NER_MODEL_DIR", "models/phobert-ner"))
    ner_max_length: int = field(default_factory=lambda: int(os.getenv("NER_MAX_LENGTH", "256")))

    # --- Knowledge graph ---
    kg_backend: str = field(default_factory=lambda: os.getenv("KG_BACKEND", "networkx"))  # networkx | neo4j
    kg_path: Path = field(default_factory=lambda: _path("KG_PATH", "data/kg/graph.pkl"))
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "realestate123"))

    # --- LLM query parser ---
    # auto: dùng LLM nếu có API key, ngược lại dùng bộ phân tích luật
    query_parser: str = field(default_factory=lambda: os.getenv("QUERY_PARSER", "auto"))  # auto | llm | rule
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-opus-5-5"))
    llm_effort: str = field(default_factory=lambda: os.getenv("LLM_EFFORT", "low"))

    # --- Xếp hạng ---
    weighting: str = field(default_factory=lambda: os.getenv("WEIGHTING", "priority"))  # equal | priority
    lambda_aux: float = field(default_factory=lambda: float(os.getenv("LAMBDA_AUX", "0.0")))
    missing_soft_score: float = field(default_factory=lambda: float(os.getenv("MISSING_SOFT_SCORE", "0.5")))
    default_top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "10")))

    @property
    def properties_path(self) -> Path:
        return self.processed_dir / "properties.jsonl"

    @property
    def has_llm_credentials(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


settings = Settings()
