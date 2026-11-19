"""PhoBERT token classification cho NER bất động sản (mục 10.3).

- Huấn luyện: `scripts/train_ner.py` (dùng `train()` bên dưới)
- Suy luận : `PhoBertNER.predict(text)` -> list span {label, start, end, text, confidence}

PhoBERT dùng tokenizer BPE "chậm" (không có word_ids()), nên căn nhãn được làm thủ công:
mỗi token âm tiết được mã hóa riêng, nhãn gán cho sub-token đầu, các sub-token sau = -100.
Torch/Transformers là phụ thuộc tùy chọn — nếu chưa cài hoặc chưa có model, hệ thống tự
chuyển sang chế độ Rule-only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .labels import BIO_LABELS, ID2LABEL, LABEL2ID
from .weak_labeling import bio_to_spans, tokenize_with_offsets

log = logging.getLogger(__name__)


def _lazy_imports():
    import torch  # noqa: F401
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: F401

    return torch, AutoTokenizer, AutoModelForTokenClassification


def encode_words(tokenizer, words: list[str], labels: Optional[list[str]], max_length: int):
    """Mã hóa danh sách từ, căn nhãn vào sub-token đầu tiên. Trả về (input_ids, label_ids, word_index)."""
    input_ids = [tokenizer.cls_token_id]
    label_ids = [-100]
    word_index = [-1]
    for wi, w in enumerate(words):
        sub = tokenizer.encode(w, add_special_tokens=False) or [tokenizer.unk_token_id]
        if len(input_ids) + len(sub) >= max_length - 1:
            break
        input_ids += sub
        label_ids += [LABEL2ID[labels[wi]] if labels else 0] + [-100] * (len(sub) - 1)
        word_index += [wi] + [-1] * (len(sub) - 1)
    input_ids.append(tokenizer.sep_token_id)
    label_ids.append(-100)
    word_index.append(-1)
    return input_ids, label_ids, word_index


class PhoBertNER:
    def __init__(self, model_dir: str | Path, max_length: int = 256, device: Optional[str] = None):
        torch, AutoTokenizer, AutoModel = _lazy_imports()
        self.torch = torch
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @classmethod
    def try_load(cls, model_dir: str | Path, max_length: int = 256) -> Optional["PhoBertNER"]:
        if not (Path(model_dir) / "config.json").exists():
            log.info("Chưa có model PhoBERT tại %s -> dùng Rule-only", model_dir)
            return None
        try:
            return cls(model_dir, max_length)
        except ImportError:
            log.warning("Chưa cài torch/transformers -> dùng Rule-only")
            return None

    def _windows(self, n: int, size: int = 160, stride: int = 120):
        start = 0
        while start < n:
            yield start, min(n, start + size)
            if start + size >= n:
                break
            start += stride

    def predict(self, text: str) -> list[dict]:
        torch = self.torch
        toks = tokenize_with_offsets(text)
        words = [t for t, _, _ in toks]
        tags = ["O"] * len(words)
        probs = [0.0] * len(words)
        for ws, we in self._windows(len(words)):
            ids, _, widx = encode_words(self.tokenizer, words[ws:we], None, self.max_length)
            with torch.no_grad():
                logits = self.model(torch.tensor([ids], device=self.device)).logits[0]
            p = torch.softmax(logits, dim=-1)
            conf, pred = p.max(dim=-1)
            for pos, wi in enumerate(widx):
                if wi < 0:
                    continue
                gi = ws + wi
                if conf[pos].item() > probs[gi]:  # cửa sổ chồng lấn: giữ dự đoán tự tin hơn
                    probs[gi] = conf[pos].item()
                    tags[gi] = self.id2label[pred[pos].item()]
        spans = bio_to_spans(toks, tags, probs)
        for sp in spans:
            sp["text"] = text[sp["start"]: sp["end"]]
        return spans


# ---------------------------------------------------------------------------
# Huấn luyện
# ---------------------------------------------------------------------------
def load_bio_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train(train_path: str | Path, eval_path: Optional[str | Path], base_model: str, output_dir: str | Path,
          epochs: int = 5, batch_size: int = 16, lr: float = 3e-5, max_length: int = 256) -> dict:
    torch, AutoTokenizer, AutoModel = _lazy_imports()
    from torch.utils.data import Dataset
    from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

    from ..evaluation.metrics import span_prf

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(
        base_model, num_labels=len(BIO_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    class BioDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.items = []
            for r in rows:
                # cắt văn bản dài thành nhiều đoạn
                for s in range(0, max(1, len(r["tokens"])), 150):
                    w, t = r["tokens"][s:s + 150], r["tags"][s:s + 150]
                    ids, lab, _ = encode_words(tokenizer, w, t, max_length)
                    self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": lab})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    train_ds = BioDataset(load_bio_jsonl(train_path))
    eval_rows = load_bio_jsonl(eval_path) if eval_path else None
    eval_ds = BioDataset(eval_rows) if eval_rows else None

    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(-1)
        gold_seqs, pred_seqs = [], []
        for p_row, l_row in zip(preds, labels):
            g, q = [], []
            for p_i, l_i in zip(p_row, l_row):
                if l_i == -100:
                    continue
                g.append(ID2LABEL[int(l_i)])
                q.append(ID2LABEL[int(p_i)])
            gold_seqs.append(g)
            pred_seqs.append(q)
        return span_prf(gold_seqs, pred_seqs)["micro"]

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="f1" if eval_ds else None,
        logging_steps=50,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics if eval_ds else None,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer.evaluate() if eval_ds else {}
