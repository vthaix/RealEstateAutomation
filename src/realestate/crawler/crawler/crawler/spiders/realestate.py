"""Spider cào tin rao bất động sản theo cấu hình rules/crawl_sites.json.

    scrapy crawl realestate                               # các web enabled, mỗi web 5000 tin
    scrapy crawl realestate -a sites=bds -a limit=100     # chọn web / số tin
    (hoặc từ thư mục gốc dự án: python scripts/crawl.py --sites bds --limit 100)

- Bỏ qua URL đã có trong data/batches (không cào lại tin cũ).
- Dừng mở trang danh sách khi số tin đã lưu + đang tải đạt giới hạn của web đó.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import scrapy

from crawler.items import RealestateItem
from realestate.rules.loader import RULES_DIR


def _norm(url: str) -> str:
    return url.split("?")[0].split("#")[0].rstrip("/")


class RealestateSpider(scrapy.Spider):
    name = "realestate"

    def __init__(self, sites: str | None = None, limit: str | int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = json.loads((RULES_DIR / "crawl_sites.json").read_text(encoding="utf-8"))
        # limit = 0 / "all" / null -> không giới hạn: cào tới khi hết tin
        raw_limit = limit if limit is not None else cfg.get("default_limit")
        self.limit = float("inf") if str(raw_limit).lower() in ("0", "all", "none", "") else int(raw_limit)
        self.limit_text = "không giới hạn" if self.limit == float("inf") else str(int(self.limit))
        self.min_len = int(cfg.get("min_text_length", 200))
        wanted = [s.strip() for s in sites.split(",")] if sites else None
        self.sites = []
        for s in cfg["sites"]:
            if wanted is None and not s.get("enabled", True):
                continue
            if wanted is not None and s["name"] not in wanted:
                continue
            if not s.get("enabled", True):
                self.logger.warning("[%s] đang tắt trong cấu hình (%s) — vẫn chạy vì được chỉ định",
                                    s["name"], s.get("_note", ""))
            self.sites.append(s)
        if not self.sites:
            raise ValueError(f"Không có web nào để cào (sites={sites})")
        self.site_state = {s["name"]: {"scheduled": 0, "done": 0, "saved": 0, "page": s.get("start_page", 1),
                                  "page_inflight": False, "exhausted": False, "blocked": False,
                                  "ok": 0, "bounced": 0, "queue": []} for s in self.sites}
        self.max_pending = int(cfg.get("max_pending_per_site", 16))
        self.block_after = int(cfg.get("block_after_bounces", 10))
        self.seen: set[str] = set()
        self.known: set[str] = set()

    # ------------------------------------------------------------------ setup
    def _load_known_urls(self) -> None:
        """URL đã có trong data/batches -> không cào lại."""
        from pathlib import Path

        from realestate.utils.io import load_raw_listings

        d = Path(self.settings.get("BATCH_DIR"))
        if d.exists():
            try:
                self.known = {_norm(li.url) for li in load_raw_listings(d) if li.url}
            except Exception as e:  # file hỏng không được chặn việc cào
                self.logger.warning("Không đọc được dữ liệu cũ trong %s: %s", d, e)
            # cả file .part (crawler khác đang chạy / lần trước dừng giữa chừng) -> tránh cào trùng
            for part in d.glob("*.jsonl.part"):
                for line in part.read_text(encoding="utf-8", errors="ignore").splitlines():
                    try:
                        self.known.add(_norm(json.loads(line)["url"]))
                    except (ValueError, KeyError):
                        continue  # dòng cuối có thể đang ghi dở
        self.logger.info("Đã có %d URL trong %s -> sẽ bỏ qua", len(self.known), d)

    async def start(self):  # Scrapy >= 2.13
        for r in self.start_requests():
            yield r

    def start_requests(self):
        self._load_known_urls()
        for s in self.sites:
            self.logger.info("[%s] bắt đầu (số tin tối đa: %s), cách tìm URL: %s", s["name"], self.limit_text, s["discovery"])
            if s["discovery"] == "pagination":
                yield self._page_request(s)
            elif s["discovery"] == "sitemap":
                for u in s["sitemap_urls"]:
                    yield scrapy.Request(u, callback=self.parse_sitemap, cb_kwargs={"site": s}, priority=-10)
            else:
                self.logger.error("[%s] discovery không hỗ trợ: %s", s["name"], s["discovery"])

    # ------------------------------------------------------------ quota helpers
    def _capacity(self, name: str) -> int:
        st = self.site_state[name]
        if st["blocked"]:
            return 0
        pending = st["scheduled"] - st["done"]
        return self.limit - st["saved"] - pending

    def on_saved(self, name: str, saved: int) -> None:
        self.site_state[name]["saved"] = saved
        if saved % 100 == 0 or saved == self.limit:
            self.logger.info("[%s] đã lưu %d/%s tin", name, saved, self.limit_text)

    def _detail_request(self, s: dict, url: str) -> scrapy.Request:
        self.site_state[s["name"]]["scheduled"] += 1
        # dont_filter: tự chống trùng bằng self.seen; để mọi phản hồi (kể cả bị đẩy về trang chủ) đều được đếm
        return scrapy.Request(url, callback=self.parse_detail, errback=self.detail_failed,
                              cb_kwargs={"site": s}, priority=10, dont_filter=True)

    def _bounce(self, s: dict, reason: str, url: str) -> None:
        """Ghi nhận phản hồi bị chặn/đẩy đi; chưa có tin nào thành công mà bị liên tục -> coi như web chặn bot."""
        st = self.site_state[s["name"]]
        st["bounced"] += 1
        self.logger.warning("[%s] %s: %s", s["name"], reason, url)
        if not st["blocked"] and st["ok"] == 0 and st["bounced"] >= self.block_after:
            st["blocked"] = True
            self.logger.error("[%s] %d lần liên tiếp bị từ chối/đẩy về trang chủ -> web không cho bot truy cập, "
                              "DỪNG cào web này (không vượt cơ chế chặn).", s["name"], st["bounced"])

    def _new_url(self, url: str) -> bool:
        k = _norm(url)
        if k in self.seen or k in self.known:
            return False
        self.seen.add(k)
        return True

    # ---------------------------------------------------------------- pagination
    def _page_request(self, s: dict) -> scrapy.Request:
        st = self.site_state[s["name"]]
        st["page_inflight"] = True
        return scrapy.Request(s["page_url"].format(page=st["page"]), callback=self.parse_listing,
                              errback=self.page_failed, cb_kwargs={"site": s}, priority=-10)

    def _feed(self, s: dict):
        """Cấp dần URL từ hàng đợi sitemap: tối đa max_pending request đang chờ mỗi web."""
        st = self.site_state[s["name"]]
        while st["queue"] and self._capacity(s["name"]) > 0 and st["scheduled"] - st["done"] < self.max_pending:
            yield self._detail_request(s, st["queue"].pop(0))

    def _maybe_next_page(self, s: dict):
        st = self.site_state[s["name"]]
        if (s["discovery"] == "pagination" and not st["exhausted"] and not st["page_inflight"]
                and self._capacity(s["name"]) > 0):
            st["page"] += 1
            if s.get("max_pages") and st["page"] > s["max_pages"]:  # null -> không giới hạn trang
                st["exhausted"] = True
                self.logger.info("[%s] đã tới max_pages", s["name"])
                return None
            return self._page_request(s)
        return None

    def parse_listing(self, response, site: dict):
        st = self.site_state[site["name"]]
        st["page_inflight"] = False
        cards = response.css(site["listing_selector"])
        if not cards:
            st["exhausted"] = True
            self.logger.info("[%s] trang %s không còn tin -> đã cào hết, dừng phân trang", site["name"], response.url)
            return
        # chốt an toàn: web trả lại đúng trang trước (hay gặp khi vượt trang cuối) -> coi như hết
        sig = tuple(c.css(site.get("listing_link_selector", "a::attr(href)")).get() for c in cards)
        if sig == st.get("last_sig"):
            st["repeat"] = st.get("repeat", 0) + 1
            if st["repeat"] >= 2:
                st["exhausted"] = True
                self.logger.info("[%s] trang lặp lại nội dung trang trước -> đã cào hết", site["name"])
                return
        else:
            st["repeat"] = 0
        st["last_sig"] = sig
        new = 0
        for card in cards:
            href = card.css(site.get("listing_link_selector", "a::attr(href)")).get()
            if not href:
                continue
            url = response.urljoin(href)
            if not url.startswith(("http://", "https://")) or site["domain"] not in url:
                continue
            if self._capacity(site["name"]) <= 0:
                break
            if self._new_url(url):
                new += 1
                yield self._detail_request(site, url)
        self.logger.info("[%s] trang %d: %d tin, %d tin mới", site["name"], st["page"], len(cards), new)
        nxt = self._maybe_next_page(site)
        if nxt:
            yield nxt

    def page_failed(self, failure):
        site = failure.request.cb_kwargs["site"]
        st = self.site_state[site["name"]]
        st["page_inflight"] = False
        self.logger.warning("[%s] lỗi trang danh sách %s: %s", site["name"], failure.request.url, failure.value)
        nxt = self._maybe_next_page(site)  # bỏ qua trang lỗi, sang trang sau
        if nxt:
            yield nxt

    # ------------------------------------------------------------------- sitemap
    def parse_sitemap(self, response, site: dict):
        response.selector.remove_namespaces()
        locs = [x.strip() for x in response.xpath("//loc/text()").getall()]
        pattern = re.compile(site["detail_url_regex"])
        details = 0
        for loc in locs:
            if loc.endswith(".xml"):
                yield scrapy.Request(loc, callback=self.parse_sitemap, cb_kwargs={"site": site}, priority=-10)
            elif pattern.search(loc):
                url = loc if "?" in loc else loc + site.get("detail_url_suffix", "")
                if self._new_url(url):
                    details += 1
                    self.site_state[site["name"]]["queue"].append(url)
        self.logger.info("[%s] sitemap %s: %d URL, %d tin mới", site["name"], response.url, len(locs), details)
        yield from self._feed(site)

    # -------------------------------------------------------------------- detail
    def parse_detail(self, response, site: dict):
        st = self.site_state[site["name"]]
        st["done"] += 1
        redirected_home = bool(response.meta.get("redirect_urls")) and urlparse(response.url).path in ("", "/")
        detail = None if redirected_home else response.css(site["detail_selector"])
        if redirected_home:
            self._bounce(site, "bị chuyển hướng về trang chủ", response.meta["redirect_urls"][0])
        elif detail:
            st["ok"] += 1
            for sel in site.get("exclude_selectors", []):
                for el in detail.css(sel):
                    el.drop()
            # nối văn bản giống hệt bộ dữ liệu gốc để source_templates.json đọc được
            text = " ".join(" ".join(detail.css("::text").getall()).split())
            if len(text) >= self.min_len:
                item = RealestateItem()
                item["url"] = response.url
                item["text"] = text
                item["site"] = site["name"]
                item["crawled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                yield item
            else:
                self.logger.debug("[%s] văn bản quá ngắn: %s", site["name"], response.url)
        else:
            self.logger.warning("[%s] không thấy detail_selector: %s", site["name"], response.url)
        yield from self._feed(site)
        nxt = self._maybe_next_page(site)
        if nxt:
            yield nxt

    def detail_failed(self, failure):
        site = failure.request.cb_kwargs["site"]
        self.site_state[site["name"]]["done"] += 1
        self._bounce(site, f"lỗi tải ({failure.value.__class__.__name__})", failure.request.url)
        yield from self._feed(site)
        nxt = self._maybe_next_page(site)
        if nxt:
            yield nxt

    def closed(self, reason):
        for name, st in self.site_state.items():
            self.logger.info("[%s] kết thúc (%s): lưu %d tin, đã tải %d trang chi tiết, trang danh sách cuối %s%s",
                             name, reason, st["saved"], st["done"], st["page"],
                             " — WEB CHẶN BOT" if st["blocked"] else "")
