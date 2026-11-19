from .hybrid import HybridExtractor
from .rule_engine import RuleEngine, Span

__all__ = ["HybridExtractor", "RuleEngine", "Span", "build_extractor"]


def build_extractor(mode: str | None = None) -> HybridExtractor:
    """Tạo extractor theo cấu hình (tự nạp PhoBERT nếu có model đã huấn luyện)."""
    from ..config import settings
    from .ner_model import PhoBertNER

    mode = mode or settings.extraction_mode
    ner = PhoBertNER.try_load(settings.ner_model_dir, settings.ner_max_length) if mode in ("bert", "hybrid") else None
    return HybridExtractor(mode=mode, ner=ner)
