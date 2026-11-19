"""Tách phần bán cấu trúc (header / khối chi tiết) và phần mô tả tự do theo từng nguồn tin.

Toàn bộ mẫu bố cục nằm trong `rules/source_templates.json`; module này chỉ thông dịch.
Trường lấy từ header được coi là structured field của sàn (source="header"); phần mô tả
được đưa qua PhoBERT NER + Rule Engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from ..rules.loader import RuleSet, load_source_templates
from ..utils.text import canonical_province, collapse_ws


@dataclass
class ParsedListing:
    source: str
    title: Optional[str] = None
    description: str = ""
    header: dict = field(default_factory=dict)    # trường bán cấu trúc lấy từ trang
    location: dict = field(default_factory=dict)  # province/district/ward/street/project/address
    category: Optional[str] = None


def _cut_description(text: str, cfg: dict) -> tuple[str, str]:
    """-> (head, description)."""
    start = cfg["start_after"]
    i = text.find(start)
    if i < 0:
        return text, ""
    head, desc = text[:i], text[i + len(start):]
    for end in cfg["end_before"]:
        j = desc.find(end)
        if j >= 0:
            desc = desc[:j]
    return head, desc.strip()


class SourceParser:
    def __init__(self, rules: RuleSet | None = None):
        self.r = rules or load_source_templates()
        self.sources = self.r["sources"]

    def _rx(self, pattern: str, extra: Optional[dict] = None, flags: int = re.U) -> re.Pattern:
        if extra:
            for k, v in extra.items():
                pattern = pattern.replace("{" + k + "}", v)
        return self.r.compile(pattern, flags)

    def detect(self, text: str, url: str) -> Optional[dict]:
        for s in self.sources:
            m = s["match"]
            if any(u in url for u in m.get("url_contains", [])):
                return s
        for s in self.sources:
            if s["match"].get("text_regex") and self._rx(s["match"]["text_regex"]).search(text):
                return s
        return None

    def parse(self, text: str, url: str = "") -> ParsedListing:
        cfg = self.detect(text, url)
        if cfg is None:
            src = url.split("/")[2] if url.count("/") >= 2 else "unknown"
            return ParsedListing(source=src, title=text[:120], description=text)
        if cfg["name"] == "radanhadat.vn":
            return self._parse_radanhadat(text, url, cfg)
        return self._parse_bds(text, cfg)

    # ------------------------------------------------------------- bds.com.vn
    def _parse_bds(self, text: str, cfg: dict) -> ParsedListing:
        p = ParsedListing(source=cfg["name"])
        head, p.description = _cut_description(text, cfg["description"])
        cats = cfg["categories"]
        alt = "|".join(re.escape(k) for k in sorted(cats, key=len, reverse=True))
        m = self._rx(cfg["category_regex"], {"CATEGORIES": alt}).search(head)
        if m:
            p.title = head[: m.start()].strip()
            p.category = m.group("cat")
            p.header["transaction_type"], p.header["property_type"] = cats[p.category]
            p.location["raw_header_location"] = m.group("loc")
        else:
            p.title = head.split(" Diện tích : ")[0][:200]
        for key, pat in cfg["header_fields"].items():
            fm = self._rx(pat).search(head)
            if fm:
                p.header[key] = fm.group("v").strip()

        loc = cfg["location"]
        tm = self._rx(loc["regex"]).search(text)
        if tm:
            parts = [collapse_ws(x) for x in tm.group("tail").split(loc["separator"])]
            parts = [x for x in parts if x]
            for level, val in zip(loc["levels"], parts):
                p.location[level] = canonical_province(val) if level == "province" else val
        raw_loc = p.location.get("raw_header_location", "")
        ward = p.location.get("ward")
        if ward and "project" not in p.location and not raw_loc.endswith(ward) and f" {ward} " in f" {raw_loc} ":
            after = raw_loc.split(ward, 1)[1].strip()
            if after:
                p.location["project"] = after
        return p

    # ----------------------------------------------------------- radanhadat.vn
    def _parse_radanhadat(self, text: str, url: str, cfg: dict) -> ParsedListing:
        p = ParsedListing(source=cfg["name"])
        head, p.description = _cut_description(text, cfg["description"])
        for t in cfg["transaction"]:
            if self._rx(t["regex"]).search(head):
                p.header["transaction_type"] = t["value"]
                break

        crumbs_and_title = head
        hm = self._rx(cfg["headline_regex"]).search(head)
        if hm:
            crumbs_and_title = head[: hm.start()]
            if "thuận" not in hm.group("price"):
                p.header["price_text"] = hm.group("price")
            if hm.group("ppm"):
                p.header["price_per_m2_text"] = f"{hm.group('ppm')} {hm.group('ppm_unit')}"
            p.header["area"] = hm.group("area")
            if hm.group("area_unit") == "ha":
                p.header["area_multiplier"] = 10000
            if hm.group("bed").isdigit():
                p.header["bedrooms"] = hm.group("bed")
            if hm.group("wc").isdigit():
                p.header["bathrooms"] = hm.group("wc")

        # tiêu đề lặp 2 lần liên tiếp ở cuối breadcrumb
        body = self._rx(cfg["breadcrumb_prefix_regex"]).sub("", crumbs_and_title).strip()
        words = body.split(" ")
        title, breadcrumb = None, body
        for n in range(len(words) // 2, 2, -1):
            if words[-2 * n: -n] == words[-n:]:
                title, breadcrumb = " ".join(words[-n:]), " ".join(words[: -2 * n])
                break
        p.title = title or body[-200:]
        for kw, ptype in cfg["breadcrumb_types"]:
            if kw.lower() in breadcrumb.lower():
                p.header["property_type"] = ptype
                break

        self._detail_block(text, cfg["detail_block"], p)

        ad = cfg["address"]
        am = self._rx(ad["regex"]).search(head)
        if am:
            addr = am.group("addr")
            toks = list(self._rx(ad["province_token_regex"]).finditer(addr))
            if len(toks) >= 2:
                new_addr, old_addr = addr[: toks[0].end()].strip(" ,"), addr[toks[0].end():].strip(" ,")
            else:
                new_addr = old_addr = addr
            p.location["address"] = new_addr
            parts = [x.strip() for x in old_addr.split(",") if x.strip()]
            for level, idx in (("province", -1), ("district", -2), ("ward", -3)):
                if len(parts) >= -idx:
                    p.location[level] = canonical_province(parts[idx]) if level == "province" else parts[idx]
            street = next((x for x in parts if x.lower().startswith(tuple(ad["street_prefixes"]))), None)
            if street:
                p.location["street"] = street
        if p.header.get("project") and "project" not in p.location:
            p.location["project"] = p.header.pop("project")
        return p

    def _detail_block(self, text: str, cfg: dict, p: ParsedListing) -> None:
        i = text.rfind(cfg["start"])
        if i < 0:
            return
        blk = text[i + len("Chi tiết "):]
        keys = sorted(cfg["keys"], key=len, reverse=True)
        rx = re.compile("(?:^| )(" + "|".join(re.escape(k) for k in keys) + ") ")
        parts = list(rx.finditer(blk))
        for j, m in enumerate(parts):
            name = cfg["keys"][m.group(1)]
            end = parts[j + 1].start() if j + 1 < len(parts) else len(blk)
            val = blk[m.end(): end].strip()
            if name == "_skip" or val in cfg["empty_values"]:
                continue
            vr = cfg["value_regex"].get(name)
            if vr:
                vm = self._rx(vr).search(val)
                if not vm:
                    continue
                val = vm.group("v")
                mult = cfg.get("unit_multipliers", {}).get(name, {})
                unit = vm.groupdict().get("u")
                if unit in mult:
                    p.header[f"{name}_multiplier"] = mult[unit]
            p.header.setdefault(name, val)


@lru_cache(maxsize=1)
def _default_parser() -> SourceParser:
    return SourceParser()


def parse_listing(text: str, url: str = "") -> ParsedListing:
    return _default_parser().parse(text, url)
