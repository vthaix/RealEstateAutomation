import scrapy


class RealestateItem(scrapy.Item):
    url = scrapy.Field()
    text = scrapy.Field()
    site = scrapy.Field()        # tên web trong crawl_sites.json (bds, radanhadat...)
    crawled_at = scrapy.Field()  # ISO time
